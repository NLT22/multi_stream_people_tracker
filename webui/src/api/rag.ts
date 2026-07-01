// Fetch wrappers for the RAG query API (src/rag/api.py, FastAPI on :8077).
// This is the first real backend the webUI talks to — every other view is still
// seeded from src/data/*. Base URL is overridable for prod via VITE_RAG_API.

const API_BASE: string =
  (import.meta as { env?: Record<string, string> }).env?.VITE_RAG_API ?? 'http://localhost:8077'

export interface ToolCall { tool: string; args: Record<string, unknown>; result: unknown }
export interface AskResult { answer: string; tool_calls: ToolCall[]; llm_disabled?: boolean }
export interface PersonCandidate { global_id: number; score: number }
export interface ZoneRank { zone: string; value: number }
export interface TimelineInterval { cam: number; zone: string; t_start: number; t_end: number }
export interface BevPoint { t: number; x: number; y: number }
export interface RunInfo { run_id: string; scene: string; env: string; fps: number; n_cams: number; n_gids: number }

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`)
  if (!r.ok) throw new Error(`${path} -> ${r.status}`)
  return r.json() as Promise<T>
}

// One backend can hold multiple runs; `runId` selects which (matches the UI dataset
// toggle). `&run_id=` appended when present; omitted -> the store's default run.
const rid = (runId?: string) => (runId ? `&run_id=${encodeURIComponent(runId)}` : '')

// Info for a specific run (or the default) served by the backend.
export const getRunInfo = (runId?: string) =>
  getJSON<RunInfo[]>('/runs').then((rs) => rs.find((r) => r.run_id === runId) ?? rs[0])

export async function ask(question: string, image?: File | null, runId?: string): Promise<AskResult> {
  const fd = new FormData()
  fd.append('question', question)
  if (runId) fd.append('run_id', runId)
  if (image) fd.append('file', image)
  const r = await fetch(`${API_BASE}/ask`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error(`/ask -> ${r.status}`)
  return r.json() as Promise<AskResult>
}

export async function searchImage(image: File, k = 5, runId?: string): Promise<PersonCandidate[]> {
  const fd = new FormData()
  fd.append('file', image)
  const r = await fetch(`${API_BASE}/search/image?k=${k}${rid(runId)}`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error(`/search/image -> ${r.status}`)
  const j = (await r.json()) as { candidates: PersonCandidate[] }
  return j.candidates
}

export const topZones = (metric = 'footfall', k = 5, runId?: string) =>
  getJSON<{ metric: string; top: ZoneRank[] }>(`/zones/top?metric=${metric}&k=${k}${rid(runId)}`)

export const personTimeline = (gid: number, runId?: string) =>
  getJSON<{ global_id: number; intervals: TimelineInterval[] }>(`/person/${gid}/timeline?_${rid(runId)}`)

export const personTrajectory = (gid: number, step = 4, runId?: string) =>
  getJSON<{ global_id: number; points: BevPoint[] }>(`/person/${gid}/trajectory?step=${step}${rid(runId)}`)

export const personDwell = (gid: number, runId?: string) =>
  getJSON<{ global_id: number; dwell: { zone: string; seconds: number; visits: number }[] }>(
    `/person/${gid}/dwell?_${rid(runId)}`)

export const apiBase = API_BASE
