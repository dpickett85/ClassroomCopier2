/**
 * mock-classroom-provider — the SQLite-backed implementation of the port.
 *
 * The mock is held to the REAL API's defaults, not to its own convenience. In
 * particular `listCourseWork` without `courseWorkStates` returns PUBLISHED
 * only, exactly as `courses.courseWork.list` does. That is deliberate and it is
 * the single most important line in this file: if the mock were more permissive
 * than the real API, every test would pass while a real adapter silently
 * dropped every Draft and Scheduled post (F8) — a scan-time silent drop the
 * item-level reconciliation invariant cannot detect, because a post that is
 * never scanned never gets a row.
 */
import { createHash } from 'node:crypto'
import type { PrismaClient } from '@prisma/client'
import type { ClassroomProvider } from '../classroom-provider.interface.js'
import {
  LicenseBlockedError,
  NotFoundError,
  PermissionError,
  RateLimitError,
} from '../types.js'
import type {
  AnswerConfig,
  AttachmentRef,
  CourseState,
  CourseWorkMaterialPayload,
  CourseWorkPayload,
  CourseWorkState,
  HealthState,
  ListCourseWorkMaterialsRequest,
  ListCourseWorkRequest,
  ListCoursesRequest,
  Page,
  PageRequest,
  ProviderAttachment,
  ProviderCourse,
  ProviderCourseWork,
  ProviderCourseWorkMaterial,
  ProviderTopic,
  RubricBody,
  ShareMode,
  SourceType,
  WorkType,
} from '../types.js'
import {
  F13_PERSISTENT_429_TITLE,
  F6_TRANSIENT_429_TITLE,
  FIXTURE_KEYS,
} from '../../fixtures/index.js'

/**
 * Run-scoped provider configuration (D25). F12's slow mode lives HERE, not in
 * the seed data: seeded as course data it would mean F4's own throughput budget
 * was measuring a deliberately-slowed course — i.e. measuring its own harness.
 */
export interface MockProviderOptions {
  /** F12 — deterministic per-item delay, supplied by the spec that needs it. */
  perItemDelayMs?: number
  /** Default page size when a caller does not ask for one. */
  defaultPageSize?: number
  /** Test hook: force every list call to page at this size. */
  forcePageSize?: number
}

/**
 * The mock's rate-limit simulation table.
 *
 * APPLY-G — a rule is keyed on a fixture title AND SCOPED TO ITS SOURCE COURSE.
 * Title alone was a global key: any post anywhere with that name 429'd,
 * including a copy of it sitting in a target course from a previous run, so a
 * later transfer using that target as its source tripped a rule intended for
 * one fixture. `sourceCourseId` is checked against the courses this provider
 * instance has actually enumerated, which is the only honest signal available
 * at create time — the create itself carries no provenance, and inventing a
 * provenance field on an API-shaped payload would be worse than the bug.
 */
interface RateLimitRule {
  title: string
  /** The fixture course this rule describes. */
  sourceCourseId: string
  /** 'transient' fails the first N attempts then succeeds (F6). */
  mode: 'transient' | 'attachmentBearing'
  failures?: number
}

const RATE_LIMIT_RULES: RateLimitRule[] = [
  { title: F6_TRANSIENT_429_TITLE, sourceCourseId: FIXTURE_KEYS.F6, mode: 'transient', failures: 1 },
  // D13 — scoped to attachment-bearing creates, so the BARE draft-shell
  // fallback genuinely succeeds. The previous "always 429" definition made the
  // guaranteed shell unreachable: the fallback was issued through the very call
  // that had just refused five times.
  { title: F13_PERSISTENT_429_TITLE, sourceCourseId: FIXTURE_KEYS.F13, mode: 'attachmentBearing' },
]

const DEFAULT_PAGE_SIZE = 200

/** The holding area a copied Drive file lands in — the acting account's Drive,
 *  which is NOT a course post. See MockAttachment.parentType in the schema. */
export const MY_DRIVE_PARENT_TYPE = 'myDrive'

function decodeToken(token: string | null | undefined): number {
  if (!token) return 0
  const n = Number.parseInt(token, 10)
  return Number.isFinite(n) && n >= 0 ? n : 0
}

function parseAnswerConfig(raw: string | null): AnswerConfig | null {
  if (!raw) return null
  try {
    return JSON.parse(raw) as AnswerConfig
  } catch {
    // A malformed answerConfig must not take the scan down. It surfaces as a
    // missing config, and the item still resolves to a terminal outcome.
    return null
  }
}

export function attachmentRefKey(ref: Pick<AttachmentRef, 'id'>): string {
  return ref.id
}

export class MockClassroomProvider implements ClassroomProvider {
  /**
   * APPLY-G — the courses this instance has read from. A rate-limit rule only
   * applies to a run that actually enumerated the fixture course the rule
   * describes; a run copying out of some other course that happens to contain a
   * post with the same title is not that scenario.
   */
  private readonly enumeratedCourses = new Set<string>()

  constructor(
    private readonly prisma: PrismaClient,
    private readonly options: MockProviderOptions = {},
  ) {}

  /* ---------------------------------------------------------------- */

  private async delay(): Promise<void> {
    const ms = this.options.perItemDelayMs ?? 0
    if (ms > 0) await new Promise((resolve) => setTimeout(resolve, ms))
  }

  private pageSize(requested?: number): number {
    return this.options.forcePageSize ?? requested ?? this.options.defaultPageSize ?? DEFAULT_PAGE_SIZE
  }

  private paginate<T>(rows: T[], offset: number, size: number): Page<T> {
    const slice = rows.slice(offset, offset + size)
    const next = offset + size < rows.length ? String(offset + size) : null
    return { items: slice, nextPageToken: next }
  }

  private mapAttachment(row: {
    id: string
    parentType: string
    parentId: string
    kind: string
    title: string
    driveFileId: string | null
    url: string | null
    shareMode: string | null
    sortOrder: number
    ownerAccountId: string | null
  }): ProviderAttachment {
    return {
      id: row.id,
      parentType: row.parentType as SourceType,
      parentId: row.parentId,
      kind: row.kind as ProviderAttachment['kind'],
      title: row.title,
      driveFileId: row.driveFileId,
      url: row.url,
      shareMode: (row.shareMode as ShareMode | null) ?? null,
      sortOrder: row.sortOrder,
      ownerAccountId: row.ownerAccountId,
    }
  }

  private async attachmentsFor(
    parentType: SourceType,
    parentIds: string[],
  ): Promise<Map<string, ProviderAttachment[]>> {
    const rows = await this.prisma.mockAttachment.findMany({
      where: { parentType, parentId: { in: parentIds } },
      // D22 — sortOrder is the total order. Which 20 attachments survive the
      // cap must not depend on whatever the query planner happened to return.
      orderBy: [{ parentId: 'asc' }, { sortOrder: 'asc' }, { id: 'asc' }],
    })
    const map = new Map<string, ProviderAttachment[]>()
    for (const row of rows) {
      const list = map.get(row.parentId) ?? []
      list.push(this.mapAttachment(row))
      map.set(row.parentId, list)
    }
    return map
  }

  /* ---------------------------------------------------------------- *
   * Reads
   * ---------------------------------------------------------------- */

  async listCourses(accountId: string, req: ListCoursesRequest = {}): Promise<Page<ProviderCourse>> {
    const states: CourseState[] = req.courseStates ?? ['ACTIVE']
    const rows = await this.prisma.mockCourse.findMany({
      where: { ownerAccountId: accountId, state: { in: states } },
      orderBy: [{ name: 'asc' }, { id: 'asc' }],
    })
    const mapped: ProviderCourse[] = rows.map((c) => ({
      id: c.id,
      name: c.name,
      section: c.section,
      state: c.state as CourseState,
      isSisShell: c.isSisShell,
      ownerAccountId: c.ownerAccountId,
    }))
    return this.paginate(mapped, decodeToken(req.pageToken), this.pageSize(req.pageSize))
  }

  /** APPLY-B — the port's own way to name a course, so no application module
   *  has to reach into `prisma.mockCourse`. */
  async getCourse(courseId: string): Promise<ProviderCourse | null> {
    const c = await this.prisma.mockCourse.findUnique({ where: { id: courseId } })
    if (!c) return null
    return {
      id: c.id,
      name: c.name,
      section: c.section,
      state: c.state as CourseState,
      isSisShell: c.isSisShell,
      ownerAccountId: c.ownerAccountId,
    }
  }

  /**
   * APPLY-K — two counts, not two paginated drains. `GET /courses` called
   * `enumeratePosts` per course, so a 30-course teacher paid 60+ paginated
   * scans plus attachment and rubric queries to render the selection screen.
   *
   * The state filter matches the enumerator's: a count that quietly excluded
   * Drafts would disagree with the number the scan later produces.
   */
  async countPosts(courseId: string): Promise<number> {
    const [work, materials] = await Promise.all([
      this.prisma.mockCourseWork.count({
        where: { courseId, state: { in: ['DRAFT', 'PUBLISHED'] } },
      }),
      this.prisma.mockCourseWorkMaterial.count({
        where: { courseId, state: { in: ['DRAFT', 'PUBLISHED'] } },
      }),
    ])
    return work + materials
  }

  async listTopics(courseId: string, req: PageRequest = {}): Promise<Page<ProviderTopic>> {
    const rows = await this.prisma.mockTopic.findMany({
      where: { courseId },
      orderBy: [{ sortOrder: 'asc' }, { id: 'asc' }],
    })
    const mapped = rows.map((t) => ({ id: t.id, name: t.name, sortOrder: t.sortOrder }))
    return this.paginate(mapped, decodeToken(req.pageToken), this.pageSize(req.pageSize))
  }

  async createTopic(courseId: string, name: string): Promise<{ topicId: string }> {
    const course = await this.prisma.mockCourse.findUnique({ where: { id: courseId } })
    if (!course) throw new NotFoundError(`Course ${courseId} not found`)
    const existing = await this.prisma.mockTopic.findFirst({ where: { courseId, name } })
    if (existing) return { topicId: existing.id }
    // A hash, not a truncated hex encoding of the name: "Semester 1" and
    // "Semester 2" share their first 16 hex characters, so the naive form
    // collided and the second createTopic threw a unique-constraint error that
    // failed the whole job.
    const id = `topic-${courseId}-${createHash('sha256').update(name).digest('hex').slice(0, 16)}`
    const count = await this.prisma.mockTopic.count({ where: { courseId } })
    await this.prisma.mockTopic.create({ data: { id, courseId, name, sortOrder: count } })
    return { topicId: id }
  }

  async listCourseWork(
    courseId: string,
    req: ListCourseWorkRequest = {},
  ): Promise<Page<ProviderCourseWork>> {
    // Real-API parity (D19): PUBLISHED only unless the caller asks otherwise.
    const states: CourseWorkState[] = req.courseWorkStates ?? ['PUBLISHED']
    this.enumeratedCourses.add(courseId)
    const rows = await this.prisma.mockCourseWork.findMany({
      where: { courseId, state: { in: states } },
      orderBy: [{ creationTime: 'asc' }, { id: 'asc' }],
    })
    const attachments = await this.attachmentsFor(
      'courseWork',
      rows.map((r) => r.id),
    )
    const rubricIds = new Set(
      (
        await this.prisma.mockRubric.findMany({
          where: { courseWorkId: { in: rows.map((r) => r.id) } },
          select: { courseWorkId: true },
        })
      ).map((r) => r.courseWorkId),
    )
    const mapped: ProviderCourseWork[] = rows.map((r) => ({
      id: r.id,
      courseId: r.courseId,
      title: r.title,
      description: r.description,
      workType: r.workType as WorkType,
      state: r.state as CourseWorkState,
      scheduledTime: r.scheduledTime,
      maxPoints: r.maxPoints,
      answerConfig: parseAnswerConfig(r.answerConfig),
      quizFormLink: r.quizFormLink,
      topicId: r.topicId,
      creationTime: r.creationTime,
      attachments: attachments.get(r.id) ?? [],
      hasRubric: rubricIds.has(r.id),
    }))
    return this.paginate(mapped, decodeToken(req.pageToken), this.pageSize(req.pageSize))
  }

  async listCourseWorkMaterials(
    courseId: string,
    req: ListCourseWorkMaterialsRequest = {},
  ): Promise<Page<ProviderCourseWorkMaterial>> {
    // APPLY-D — its own parameter name; same PUBLISHED-only real-API default.
    const states: CourseWorkState[] = req.courseWorkMaterialStates ?? ['PUBLISHED']
    this.enumeratedCourses.add(courseId)
    const rows = await this.prisma.mockCourseWorkMaterial.findMany({
      where: { courseId, state: { in: states } },
      orderBy: [{ creationTime: 'asc' }, { id: 'asc' }],
    })
    const attachments = await this.attachmentsFor(
      'courseWorkMaterial',
      rows.map((r) => r.id),
    )
    const mapped: ProviderCourseWorkMaterial[] = rows.map((r) => ({
      id: r.id,
      courseId: r.courseId,
      title: r.title,
      description: r.description,
      state: r.state as CourseWorkState,
      topicId: r.topicId,
      creationTime: r.creationTime,
      attachments: attachments.get(r.id) ?? [],
    }))
    return this.paginate(mapped, decodeToken(req.pageToken), this.pageSize(req.pageSize))
  }

  /** D20 — one query for the whole batch, not N round-trips. */
  async getAttachmentHealth(refs: AttachmentRef[]): Promise<Map<string, HealthState>> {
    const result = new Map<string, HealthState>()
    if (refs.length === 0) return result
    const rows = await this.prisma.mockAttachment.findMany({
      where: { id: { in: refs.map((r) => r.id) } },
      select: { id: true, driveState: true },
    })
    const byId = new Map(rows.map((r) => [r.id, r.driveState as HealthState]))
    for (const ref of refs) {
result.set(attachmentRefKey(ref), (byId.get(ref.id) ?? 'deleted') as any)    }
    return result
  }

  /**
   * P0-3 — a COPY, and only a copy.
   *
   * This method used to `update` the SOURCE course's attachment row in place —
   * rewriting its `driveFileId`, reassigning its `ownerAccountId` and flipping
   * its `driveState` to healthy. That made the one method named "copy" the only
   * write to the source course anywhere in the system, and the product's whole
   * proposition is that copying is non-destructive. It also healed F3's
   * permission_locked finding within a session, so the scenario stopped being
   * reproducible; and because the engine re-read the same source rows
   * afterwards, a faithful adapter (`drive.files.copy` → a new file, source
   * untouched) would have returned a new id into a caller that ignored it and
   * linked the still-locked original.
   *
   * So: insert a NEW row, in the acting account's Drive rather than on any
   * course post, leave the source exactly as it was, and return the new id for
   * the caller to substitute into the materials payload.
   */
  async copyAttachmentToMyDrive(
    ref: AttachmentRef,
    actingAccountId: string,
  ): Promise<{ newDriveFileId: string }> {
    const row = await this.prisma.mockAttachment.findUnique({ where: { id: ref.id } })
    if (!row) throw new NotFoundError(`Attachment ${ref.id} not found`)
    if (row.driveState === 'deleted' || row.driveState === 'trashed') {
      throw new NotFoundError(`Attachment ${ref.id} is ${row.driveState} and cannot be copied`)
    }

    const newDriveFileId = `${row.driveFileId ?? row.id}-copy-${actingAccountId}`
    const copyRowId = `${row.id}-copy-${actingAccountId}`
    const copy = {
      // NOT the source post: the acting account's Drive. Neither coursework
      // surface ever returns a row with this parentType, so the source post's
      // attachment list is byte-for-byte what it was.
      parentType: MY_DRIVE_PARENT_TYPE,
      parentId: actingAccountId,
      kind: row.kind,
      driveFileId: newDriveFileId,
      url: row.url,
      title: row.title,
      shareMode: row.shareMode,
      driveState: 'healthy',
      ownerAccountId: actingAccountId,
      sortOrder: row.sortOrder,
    }
    await this.prisma.mockAttachment.upsert({
      where: { id: copyRowId },
      create: { id: copyRowId, ...copy },
      update: copy,
    })
    return { newDriveFileId }
  }

  /* ---------------------------------------------------------------- *
   * Writes
   * ---------------------------------------------------------------- */

  /**
   * The rate-limit simulation. `attachmentBearing` refuses only creates that
   * carry materials[], which is what makes the bare draft-shell fallback a
   * genuinely different call with a genuinely different outcome (D13).
   */
  private async enforceRateLimit(title: string, materialCount: number): Promise<void> {
    const rule = RATE_LIMIT_RULES.find((r) => r.title === title)
    if (!rule) return
    // APPLY-G — the rule belongs to a source course, not to a string. A run
    // that never read that course is not the scenario this rule simulates.
    if (!this.enumeratedCourses.has(rule.sourceCourseId)) return

    if (rule.mode === 'attachmentBearing') {
      if (materialCount > 0) {
        throw new RateLimitError('Quota exceeded for attachment-bearing create', 25)
      }
      return
    }

    // Cycle-2 DEFER 1 — the attempt count is persisted in SQLite rather than
    // held in an instance-local Map, so a provider rebuilt mid-run (a fresh
    // `new MockClassroomProvider(...)` over the SAME database) continues the
    // count instead of restarting it at zero and re-issuing 429s the caller
    // already exhausted.
    const key = `${rule.title}:${rule.mode}`
    const seen = await this.readAttemptCount(key)
    if (seen < (rule.failures ?? 1)) {
      await this.incrementAttemptCount(key)
      throw new RateLimitError('Quota exceeded, retry shortly', 25)
    }
  }

  private async readAttemptCount(key: string): Promise<number> {
    const row = await this.prisma.mockRateLimitAttempt.findUnique({ where: { key } })
    return row?.attemptCount ?? 0
  }

  /** Upsert-and-increment in one round trip so two concurrent callers racing
   *  the SAME key still land on distinct counts rather than both reading 0. */
  private async incrementAttemptCount(key: string): Promise<void> {
    await this.prisma.mockRateLimitAttempt.upsert({
      where: { key },
      create: { key, attemptCount: 1 },
      update: { attemptCount: { increment: 1 } },
    })
  }

  private async requireCourse(courseId: string) {
    const course = await this.prisma.mockCourse.findUnique({ where: { id: courseId } })
    if (!course) throw new NotFoundError(`Course ${courseId} not found`)
    if (course.state === 'ARCHIVED') {
      throw new PermissionError(`Course ${courseId} is archived and cannot be written to`)
    }
    return course
  }

  private async writeMaterials(
    parentType: SourceType,
    parentId: string,
    materials: CourseWorkPayload['materials'],
  ): Promise<void> {
    for (const [index, material] of materials.entries()) {
      const id = `${parentId}-att-${index}`
      await this.prisma.mockAttachment.create({
        data: {
          id,
          parentType,
          parentId,
          kind: material.kind,
          title: material.title,
          driveFileId: material.kind === 'driveFile' ? material.driveFileId : null,
          url:
            material.kind === 'link'
              ? material.url
              : material.kind === 'form'
                ? material.formUrl
                : material.kind === 'youTubeVideo'
                  ? `https://youtube.mock/watch?v=${material.videoId}`
                  : null,
          // Copied through verbatim. There is no code path here that can
          // default it, because the union requires it on driveFile and does not
          // define it anywhere else.
          shareMode: material.kind === 'driveFile' ? material.shareMode : null,
          driveState: 'healthy',
          sortOrder: index,
        },
      })
    }
  }

  async createCourseWork(courseId: string, payload: CourseWorkPayload): Promise<{ id: string }> {
    await this.delay()
    await this.requireCourse(courseId)
    await this.enforceRateLimit(payload.title, payload.materials.length)

    const count = await this.prisma.mockCourseWork.count({ where: { courseId } })
    const id = `cw-${courseId}-${count}-${Date.now().toString(36)}`
    await this.prisma.mockCourseWork.create({
      data: {
        id,
        courseId,
        title: payload.title,
        description: payload.description ?? null,
        workType: payload.workType,
        // Literal DRAFT. Dates are structurally absent from the payload type,
        // so "everything lands as a Draft with dates cleared" is not something
        // a caller can violate by forgetting.
        state: 'DRAFT',
        dueDate: null,
        scheduledTime: null,
        maxPoints: payload.maxPoints ?? null,
        answerConfig: payload.answerConfig ? JSON.stringify(payload.answerConfig) : null,
        quizFormLink: payload.quizFormLink ?? null,
        topicId: payload.topicId ?? null,
        creationTime: new Date(),
        createdOrder: count,
      },
    })
    await this.writeMaterials('courseWork', id, payload.materials)
    return { id }
  }

  async createCourseWorkMaterial(
    courseId: string,
    payload: CourseWorkMaterialPayload,
  ): Promise<{ id: string }> {
    await this.delay()
    await this.requireCourse(courseId)
    await this.enforceRateLimit(payload.title, payload.materials.length)

    const count = await this.prisma.mockCourseWorkMaterial.count({ where: { courseId } })
    const id = `cwm-${courseId}-${count}-${Date.now().toString(36)}`
    await this.prisma.mockCourseWorkMaterial.create({
      data: {
        id,
        courseId,
        title: payload.title,
        description: payload.description ?? null,
        state: 'DRAFT',
        topicId: payload.topicId ?? null,
        creationTime: new Date(),
        createdOrder: count,
      },
    })
    await this.writeMaterials('courseWorkMaterial', id, payload.materials)
    return { id }
  }

  async updateCourseWorkDescription(courseWorkId: string, description: string): Promise<void> {
    const row = await this.prisma.mockCourseWork.findUnique({ where: { id: courseWorkId } })
    if (!row) throw new NotFoundError(`CourseWork ${courseWorkId} not found`)
    await this.prisma.mockCourseWork.update({ where: { id: courseWorkId }, data: { description } })
  }

  async updateCourseWorkMaterialDescription(
    materialId: string,
    description: string,
  ): Promise<void> {
    const row = await this.prisma.mockCourseWorkMaterial.findUnique({ where: { id: materialId } })
    if (!row) throw new NotFoundError(`CourseWorkMaterial ${materialId} not found`)
    await this.prisma.mockCourseWorkMaterial.update({
      where: { id: materialId },
      data: { description },
    })
  }

  /* ---------------------------------------------------------------- *
   * Rubrics (D23) — get-then-create, two calls, two failure surfaces
   * ---------------------------------------------------------------- */

  async getRubric(courseWorkId: string): Promise<RubricBody | null> {
    const rubric = await this.prisma.mockRubric.findUnique({
      where: { courseWorkId },
      include: { criteria: { include: { levels: true }, orderBy: { sortOrder: 'asc' } } },
    })
    if (!rubric) return null
    return {
      criteria: rubric.criteria.map((c) => ({
        title: c.title,
        description: c.description,
        sortOrder: c.sortOrder,
        levels: [...c.levels]
          .sort((a, b) => a.sortOrder - b.sortOrder)
          .map((l) => ({
            title: l.title,
            description: l.description,
            points: l.points,
            sortOrder: l.sortOrder,
          })),
      })),
    }
  }

  /** The licence denial arrives on the CREATE, which is where the real API
   *  raises it — not on the read. */
  async createRubric(targetCourseWorkId: string, rubric: RubricBody): Promise<{ id: string }> {
    const courseWork = await this.prisma.mockCourseWork.findUnique({
      where: { id: targetCourseWorkId },
      include: { course: true },
    })
    if (!courseWork) throw new NotFoundError(`CourseWork ${targetCourseWorkId} not found`)
    if (!courseWork.course.rubricsLicensed) {
      throw new LicenseBlockedError(
        `Rubrics are not available on course ${courseWork.courseId}'s licence tier`,
      )
    }

    const rubricId = `rubric-${targetCourseWorkId}`
    await this.prisma.mockRubric.create({
      data: { id: rubricId, courseWorkId: targetCourseWorkId, licenseBlocked: false },
    })
    for (const [ci, criterion] of rubric.criteria.entries()) {
      const criterionId = `${rubricId}-c${ci}`
      await this.prisma.mockRubricCriterion.create({
        data: {
          id: criterionId,
          rubricId,
          title: criterion.title,
          description: criterion.description,
          sortOrder: criterion.sortOrder,
        },
      })
      for (const [li, level] of criterion.levels.entries()) {
        await this.prisma.mockRubricLevel.create({
          data: {
            id: `${criterionId}-l${li}`,
            criterionId,
            title: level.title,
            description: level.description,
            points: level.points,
            sortOrder: level.sortOrder,
          },
        })
      }
    }
    return { id: rubricId }
  }
}
