/** 관제 API 클라이언트.
 *
 * 타입은 `src/firstlight/events/models.py` 의 `Event.to_public_dict()` 와
 * `api/main.py` 의 `/api/summary` 응답을 그대로 반영한다. 백엔드에서 필드가
 * 바뀌면 여기도 바뀌어야 한다.
 */

export type Tier = "FLARE" | "GLOW" | "SPARK"
export type EventLabel = "unlabelled" | "confirmed" | "false_positive"

export interface FireEvent {
  event_id: number
  track_id: number
  tier: Tier
  score: number
  timestamp: number
  site: string
  camera: string

  bbox: [number, number, number, number]
  confidence: number
  n_observations: number

  /** false 면 좌표 필드는 의미가 없다 — 지오레퍼런싱이 발행을 거절한 것이다. */
  geo_ok: boolean
  lat: number | null
  lon: number | null
  elevation_m: number | null
  range_m: number | null
  depression_deg: number | null
  cep50_m: number | null
  cep90_m: number | null
  geo_reject_reason: string | null

  /** 탐지 시점 크롭 이미지. 실시간 모드는 파일명, 데모는 data URI 다. */
  snapshot: string | null

  /** 시퀀스 검증 2조건 (작품설명서 Ⅱ-1). 등급의 직접적 근거다. */
  flow_ok: boolean
  area_ok: boolean
  criteria: {
    n_satisfied?: number
    enough_frames?: boolean
    flow_from_proxy?: boolean
    flow_upward_ratio?: number
    flow_divergence?: number
    rise_rate?: number
    area_growth_rate?: number
    area_monotonicity?: number
    reasons?: string[]
  }

  label: EventLabel
  features: Record<string, number | string>
  /** 특징별 로짓 기여도. 같은 등급 안의 우선순위를 매기는 보조 신호다. */
  explanation: Record<string, number>

  created_at: number
  tier_colour: string
  tier_label_ko: string
}

export interface ResponseUnit {
  name: string
  lat: number
  lon: number
}

export interface SiteInfo {
  name: string
  label: string
  lat: number
  lon: number
  bbox: [number, number, number, number]
  response_units: ResponseUnit[]
}

export interface Summary {
  counts: Partial<Record<Tier, number>>
  total: number
  labelled: number
  queue: number
  scorer: {
    fitted: boolean
    n_train: number
    mode: "dense" | "sparse"
    tau_high: number
    tau_low: number
  }
}

async function json<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  events: (limit = 300) => json<FireEvent[]>(`/api/events?limit=${limit}`),

  summary: () => json<Summary>("/api/summary"),

  site: () => json<SiteInfo>("/api/site"),

  setLabel: (eventId: number, label: EventLabel) =>
    json<{ ok: boolean }>(`/api/events/${eventId}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    }),

  retrain: () =>
    json<{ ok: boolean; n_train: number; n_positive: number; n_negative: number }>(
      "/api/scorer/retrain",
      { method: "POST" },
    ),
}

/** 스냅샷을 화면에 걸 수 있는 URL 로 바꾼다.
 *
 * 데모는 이미 data URI 라 그대로 쓰고, 실시간 모드는 파일명이라 API 경로를
 * 붙인다. 없으면 null — 호출부가 자리를 비워 둔다. */
export function snapshotUrl(snapshot: string | null): string | null {
  if (!snapshot) return null
  return snapshot.startsWith("data:") ? snapshot : `/api/snapshots/${snapshot}`
}

/** 등급별 유틸리티 클래스. 소개서 §8 색을 Tailwind 토큰으로 등록해 두었다. */
export const TIER_STYLES: Record<Tier, { text: string; bg: string; border: string; ring: string }> = {
  FLARE: {
    text: "text-flare",
    bg: "bg-flare/10",
    border: "border-flare/35",
    ring: "bg-flare",
  },
  GLOW: {
    text: "text-glow",
    bg: "bg-glow/10",
    border: "border-glow/35",
    ring: "bg-glow",
  },
  SPARK: {
    text: "text-spark",
    bg: "bg-spark/10",
    border: "border-spark/35",
    ring: "bg-spark",
  },
}

export const TIER_ORDER: Tier[] = ["FLARE", "GLOW", "SPARK"]
