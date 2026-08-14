import { Check, X } from "lucide-react"

import { TierBadge } from "@/components/tier-badge"
import { Separator } from "@/components/ui/separator"
import { ResponseTimeline } from "@/components/response-timeline"
import { snapshotUrl, type FireEvent, type ResponseStatus } from "@/lib/api"
import { demoSequenceGif } from "@/lib/demo"
import { cn } from "@/lib/utils"

/** 특징 이름을 사람이 읽을 수 있는 말로. `verify/features.py` 와 대응한다. */
const FEATURE_KO: Record<string, string> = {
  persistence: "지속성",
  area_growth_rate: "면적 성장률",
  area_monotonicity: "면적 단조성",
  centroid_rise_rate: "상승 속도",
  translation_over_growth: "이동/성장 비",
  aspect_growth_rate: "세로 신장률",
  intensity_flicker: "밝기 변동",
  edge_softness: "경계 흐림",
  confidence_mean: "평균 신뢰도",
  confidence_slope: "신뢰도 추세",
  flow_divergence: "광류 발산",
  flow_upward_ratio: "상승 벡터 비율",
  flow_translation_mag: "잔차 평행이동",
}

function Condition({
  met,
  label,
  detail,
}: {
  met: boolean
  label: string
  detail: string
}) {
  return (
    <div className="flex items-start gap-2">
      {/* 통과 여부를 색만이 아니라 기호로도 표시한다 */}
      {met ? (
        <Check className="text-flare mt-0.5 size-4 shrink-0" aria-hidden />
      ) : (
        <X className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden />
      )}
      <div className="min-w-0">
        <div className={cn("text-sm", met ? "font-medium" : "text-muted-foreground")}>
          {label}
          <span className="sr-only">{met ? " — 통과" : " — 미달"}</span>
        </div>
        <div className="text-muted-foreground font-mono text-xs tabular-nums">
          {detail}
        </div>
      </div>
    </div>
  )
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono tabular-nums">{value}</span>
    </div>
  )
}

export function EventDetail({
  event,
  onAdvanceResponse,
  responsePending,
}: {
  event: FireEvent | null
  onAdvanceResponse?: (status: ResponseStatus) => void
  responsePending?: boolean
}) {
  if (!event) {
    return (
      <p className="text-muted-foreground text-sm">
        목록이나 지도에서 이벤트를 선택하세요.
      </p>
    )
  }

  const snapshot = snapshotUrl(event.snapshot)
  // 데모는 키로 조회, 실시간은 파일명 → API 경로.
  const sequenceGif =
    demoSequenceGif(event.sequence_gif_key) ?? snapshotUrl(event.sequence_gif ?? null)

  // 기여도 절댓값 상위 5개 — 이 등급이 나온 이유의 대부분이다.
  const contributions = Object.entries(event.explanation ?? {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 5)
  const peak = Math.max(...contributions.map(([, v]) => Math.abs(v)), 0.001)

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <TierBadge tier={event.tier} labelKo={event.tier_label_ko} />
        <span className="text-muted-foreground font-mono text-xs tabular-nums">
          #{event.event_id} · 트랙 {event.track_id}
        </span>
      </div>

      {(snapshot || sequenceGif) && (
        <figure className="space-y-1.5">
          <div className="grid gap-2" style={{ gridTemplateColumns: sequenceGif && snapshot ? "1fr 1fr" : "1fr" }}>
            {snapshot && (
              <img
                src={snapshot}
                alt={`탐지 지점 크롭 — ${event.tier}, 관측 ${event.n_observations}회`}
                className="bg-muted max-h-56 w-full rounded-lg border object-contain"
              />
            )}
            {sequenceGif && (
              <img
                src={sequenceGif}
                alt="시퀀스 GIF — 트랙 전체의 형태 변화와 상승 운동"
                className="bg-muted max-h-56 w-full rounded-lg border object-contain"
              />
            )}
          </div>
          <figcaption className="text-muted-foreground text-xs">
            {sequenceGif
              ? "왼쪽 탐지 시점 크롭 · 오른쪽 시퀀스 GIF(고정 크롭이라 연기가 자라는 것이 보인다)"
              : "탐지 시점 크롭 · 박스는 등급 색 · 주변 맥락 포함"}
          </figcaption>
        </figure>
      )}

      <div className="space-y-1.5">
        <Row label="검증 점수" value={event.score.toFixed(3)} />
        <Row label="관측 횟수" value={`${event.n_observations}회`} />
        {event.geo_ok && event.lat !== null && event.lon !== null ? (
          <>
            <Row
              label="추정 발화점"
              value={`${event.lat.toFixed(6)}N ${event.lon.toFixed(6)}E`}
            />
            <Row label="표고" value={`${Math.round(event.elevation_m ?? 0)} m`} />
            <Row
              label="오차반경"
              value={`CEP50 ${Math.round(event.cep50_m ?? 0)} m / CEP90 ${Math.round(
                event.cep90_m ?? 0,
              )} m`}
            />
          </>
        ) : (
          <Row
            label="좌표"
            value={<span className="text-glow">미발행 ({event.geo_reject_reason})</span>}
          />
        )}
      </div>

      {onAdvanceResponse && (
        <>
          <Separator />
          <ResponseTimeline
            event={event}
            onAdvance={onAdvanceResponse}
            pending={Boolean(responsePending)}
          />
        </>
      )}

      <Separator />

      {/* 작품설명서 Ⅱ-1: 등급은 두 조건 중 몇 개를 만족했는가로 정해진다.
          관제 요원이 "왜 GLOW냐"고 물었을 때 답이 되는 부분이다. */}
      <div>
        <h3 className="mb-2.5 text-sm font-medium">
          시퀀스 검증
          <span className="text-muted-foreground ml-1.5 font-mono text-xs">
            {event.criteria?.n_satisfied ?? 0}/2 조건
          </span>
        </h3>
        <div className="space-y-2">
          <Condition
            met={event.flow_ok}
            label="광류 — 상승·확산 운동"
            detail={
              event.criteria?.flow_from_proxy
                ? `기하 대용값 · 상승률 ${(event.criteria?.rise_rate ?? 0).toExponential(1)}`
                : `상승 벡터 ${((event.criteria?.flow_upward_ratio ?? 0) * 100).toFixed(0)}% · 발산 ${(event.criteria?.flow_divergence ?? 0).toFixed(2)}`
            }
          />
          <Condition
            met={event.area_ok}
            label="마스크 면적 — 단조 증가"
            detail={`성장률 ${(event.criteria?.area_growth_rate ?? 0).toExponential(1)} · 단조성 ${((event.criteria?.area_monotonicity ?? 0) * 100).toFixed(0)}%`}
          />
          {event.criteria?.enough_frames === false && (
            <p className="text-glow text-xs">
              관측 {event.n_observations}회 — 시퀀스 검증에 필요한 프레임 부족으로 강등
            </p>
          )}
        </div>
      </div>

      <Separator />

      <div>
        <h3 className="mb-2.5 text-sm font-medium">
          보조 신호
          <span className="text-muted-foreground ml-1.5 text-xs font-normal">
            등급을 정하지는 않음
          </span>
        </h3>
        {contributions.length === 0 ? (
          <p className="text-muted-foreground text-sm">설명 정보가 없습니다.</p>
        ) : (
          <ul className="space-y-2">
            {contributions.map(([name, value]) => {
              const positive = value > 0
              return (
                <li key={name} className="space-y-1">
                  <div className="flex items-baseline justify-between gap-3 text-xs">
                    <span>{FEATURE_KO[name] ?? name}</span>
                    <span
                      className={cn(
                        "font-mono tabular-nums",
                        positive ? "text-flare" : "text-muted-foreground",
                      )}
                    >
                      {positive ? "+" : ""}
                      {value.toFixed(2)}
                    </span>
                  </div>
                  {/* 0을 가운데 두고 좌우로 뻗는 막대 — 부호가 형태로 읽힌다 */}
                  <div className="bg-muted relative h-1.5 overflow-hidden rounded-full">
                    <div
                      className={cn(
                        "absolute top-0 h-full",
                        positive ? "bg-flare" : "bg-spark",
                      )}
                      style={{
                        left: positive ? "50%" : undefined,
                        right: positive ? undefined : "50%",
                        width: `${(Math.abs(value) / peak) * 50}%`,
                      }}
                    />
                    <div className="bg-border absolute top-0 left-1/2 h-full w-px" />
                  </div>
                </li>
              )
            })}
          </ul>
        )}
        <p className="text-muted-foreground mt-3 text-xs">
          양수는 연기 쪽, 음수는 오탐 쪽으로 민 값입니다.
        </p>
      </div>
    </div>
  )
}
