import { Check, MapPinOff, Truck, X } from "lucide-react"

import { TierBadge } from "@/components/tier-badge"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  snapshotUrl,
  TIER_STYLES,
  type EventLabel,
  type FireEvent,
} from "@/lib/api"
import { cn } from "@/lib/utils"

/** 지오레퍼런싱 거절 사유를 사람 말로 옮긴다.
 *  `geo/raycast.py` 의 RejectReason 과 대응한다. */
const REJECT_REASON_KO: Record<string, string> = {
  grazing: "부각이 너무 낮음 — 오차가 폭발하는 각도",
  no_intersection: "지형과 만나지 않음 — 지평선 위를 봄",
  out_of_dem: "DEM 범위를 벗어남",
  origin_below: "드론이 지형면 아래 — 고도·기준면 오류",
  cep_too_large: "오차반경이 상한 초과",
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono tabular-nums">{value}</span>
    </div>
  )
}

export function EventCard({
  event,
  selected,
  onSelect,
  onLabel,
  pending,
}: {
  event: FireEvent
  selected: boolean
  onSelect: () => void
  onLabel: (label: EventLabel) => void
  pending: boolean
}) {
  const style = TIER_STYLES[event.tier]
  const labelled = event.label !== "unlabelled"
  const snapshot = snapshotUrl(event.snapshot)

  return (
    <Card
      onClick={onSelect}
      data-selected={selected}
      className={cn(
        "cursor-pointer gap-0 border-l-4 p-3.5 transition-colors",
        "hover:bg-accent/50 data-[selected=true]:ring-ring/60 data-[selected=true]:ring-2",
        style.border.replace("/35", "/70"),
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <TierBadge tier={event.tier} labelKo={event.tier_label_ko} />
        <span className="text-muted-foreground font-mono text-xs tabular-nums">
          점수 {event.score.toFixed(2)}
        </span>
        <span className="text-muted-foreground text-xs">
          관측 {event.n_observations}회
        </span>
        {/* 대응이 시작된 건은 큐에서 바로 구분돼야 한다 — 이미 출동한
            현장을 다시 판정하려 들면 안 된다. */}
        {event.response !== "none" && (
          <Badge
            variant="outline"
            className="border-flare/40 text-flare gap-1 text-[11px]"
          >
            <Truck className="size-3" aria-hidden />
            {event.response_label_ko}
          </Badge>
        )}
        <span className="text-muted-foreground ml-auto font-mono text-xs tabular-nums">
          #{event.event_id}
        </span>
      </div>

      {/* 탐지 시점 크롭. 좌표와 점수만으로는 "실제 연기 / 오탐"을 고를 수
          없다 — 요원이 실제로 보는 것은 그 순간 그 자리의 그림이다. */}
      {snapshot && (
        <div className="mt-2.5 flex gap-2.5">
          <img
            src={snapshot}
            alt={`탐지 지점 크롭 — ${event.tier}, 신뢰도 ${event.confidence.toFixed(2)}`}
            loading="lazy"
            className="bg-muted size-20 shrink-0 rounded-md border object-cover"
          />
          <div className="min-w-0 flex-1 self-center">
            {event.geo_ok && event.lat !== null && event.lon !== null ? (
              <div className="font-mono text-[13px] tabular-nums">
                {event.lat.toFixed(5)}N {event.lon.toFixed(5)}E
                <span className="text-muted-foreground ml-1.5">
                  ± {Math.round(event.cep50_m ?? 0)}m
                </span>
              </div>
            ) : (
              <div className="text-glow flex items-start gap-1.5 text-[13px]">
                <MapPinOff className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                <span>
                  좌표 미발행
                  <span className="text-muted-foreground">
                    {" — "}
                    {REJECT_REASON_KO[event.geo_reject_reason ?? ""] ??
                      event.geo_reject_reason ??
                      "사유 미상"}
                  </span>
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 스냅샷이 있으면 위쪽 블록이 좌표를 이미 보여준다 */}
      {!snapshot &&
        (event.geo_ok && event.lat !== null && event.lon !== null ? (
          <div className="mt-2 font-mono text-[13px] tabular-nums">
            {event.lat.toFixed(5)}N {event.lon.toFixed(5)}E
            <span className="text-muted-foreground ml-1.5">
              ± {Math.round(event.cep50_m ?? 0)}m
            </span>
          </div>
        ) : (
          <div className="text-glow mt-2 flex items-start gap-1.5 text-[13px]">
            <MapPinOff className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            <span>
              좌표 미발행
              <span className="text-muted-foreground">
                {" — "}
                {REJECT_REASON_KO[event.geo_reject_reason ?? ""] ??
                  event.geo_reject_reason ??
                  "사유 미상"}
              </span>
            </span>
          </div>
        ))}

      <div className="text-muted-foreground mt-2 flex flex-wrap gap-x-3.5 gap-y-1 text-xs">
        <Field label="t" value={`${event.timestamp.toFixed(0)}s`} />
        <Field label="신뢰도" value={event.confidence.toFixed(2)} />
        {event.range_m !== null && (
          <Field label="사거리" value={`${Math.round(event.range_m).toLocaleString()}m`} />
        )}
        {event.depression_deg !== null && (
          <Field label="부각" value={`${event.depression_deg.toFixed(0)}°`} />
        )}
      </div>

      {labelled ? (
        <div className="text-muted-foreground mt-3 flex items-center gap-1.5 text-xs">
          {event.label === "confirmed" ? (
            <Check className="text-flare size-3.5" aria-hidden />
          ) : (
            <X className="size-3.5" aria-hidden />
          )}
          판정: {event.label === "confirmed" ? "실제 연기" : "오탐"}
        </div>
      ) : (
        <div className="mt-3 flex gap-2" onClick={(e) => e.stopPropagation()}>
          <Button
            size="sm"
            disabled={pending}
            onClick={() => onLabel("confirmed")}
            className="h-7 px-2.5 text-xs"
          >
            <Check aria-hidden />
            실제 연기
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={pending}
            onClick={() => onLabel("false_positive")}
            className="h-7 px-2.5 text-xs"
          >
            <X aria-hidden />
            오탐
          </Button>
        </div>
      )}
    </Card>
  )
}
