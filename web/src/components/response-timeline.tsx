import { Check, Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  RESPONSE_LABEL_KO,
  RESPONSE_ORDER,
  RESPONSE_STEPS,
  type FireEvent,
  type ResponseStatus,
} from "@/lib/api"
import { cn } from "@/lib/utils"

/** 대응 이력 — 작품설명서 「이력 타임라인: 경보 이력 및 대응 상태(접수-출동-진화)」.
 *
 * 판정("연기가 맞았나")과 다른 축이다. 이쪽은 "그래서 어떻게 됐나"를 기록한다.
 * 오탐으로 판정한 이벤트에도 접수 이력이 남을 수 있어야 하므로 필드를 나눴다.
 *
 * 각 단계에 **시각**을 남기는 것이 핵심이다. 이 시스템이 줄이려는 값이
 * 대응 지연이므로, 접수→출동 간격이 측정 가능해야 개선했는지 알 수 있다.
 * 되돌리기는 막는다 — 순서가 뒤집힌 이력은 그 측정을 무의미하게 만든다.
 */
export function ResponseTimeline({
  event,
  onAdvance,
  pending,
}: {
  event: FireEvent
  onAdvance: (status: ResponseStatus) => void
  pending: boolean
}) {
  const currentIndex = RESPONSE_ORDER.indexOf(event.response)
  const timeOf = (status: ResponseStatus) =>
    event.response_history?.find((h) => h.status === status)?.at ?? null

  return (
    <div>
      <h3 className="mb-2.5 text-sm font-medium">
        대응 이력
        {event.response !== "none" && (
          <span className="text-muted-foreground ml-1.5 text-xs font-normal">
            현재 {event.response_label_ko ?? RESPONSE_LABEL_KO[event.response]}
          </span>
        )}
      </h3>

      <ol className="space-y-0">
        {RESPONSE_STEPS.map((step, i) => {
          const stepIndex = RESPONSE_ORDER.indexOf(step.value)
          const done = currentIndex >= stepIndex
          const isNext = currentIndex === stepIndex - 1
          const at = timeOf(step.value)
          const last = i === RESPONSE_STEPS.length - 1

          return (
            <li key={step.value} className="flex gap-3">
              {/* 진행 표시선 */}
              <div className="flex flex-col items-center">
                <span
                  className={cn(
                    "grid size-5 shrink-0 place-items-center rounded-full border text-[10px]",
                    done
                      ? "bg-flare border-flare text-white"
                      : "border-border text-muted-foreground",
                  )}
                  aria-hidden
                >
                  {done ? <Check className="size-3" /> : i + 1}
                </span>
                {!last && (
                  <span
                    className={cn(
                      "w-px flex-1",
                      done ? "bg-flare/50" : "bg-border",
                    )}
                  />
                )}
              </div>

              <div className={cn("min-w-0 flex-1", last ? "pb-0" : "pb-4")}>
                <div className="flex items-baseline justify-between gap-3">
                  <span
                    className={cn(
                      "text-sm",
                      done ? "font-medium" : "text-muted-foreground",
                    )}
                  >
                    {step.label}
                    <span className="sr-only">{done ? " 완료" : " 미완료"}</span>
                  </span>

                  {at !== null ? (
                    <time className="text-muted-foreground font-mono text-xs tabular-nums">
                      {new Date(at * 1000).toLocaleTimeString("ko-KR", {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </time>
                  ) : isNext ? (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={pending}
                      onClick={() => onAdvance(step.value)}
                      className="h-6 px-2 text-xs"
                    >
                      {pending && <Loader2 className="animate-spin" aria-hidden />}
                      {step.label} 기록
                    </Button>
                  ) : null}
                </div>
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
