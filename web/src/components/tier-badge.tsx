import { Badge } from "@/components/ui/badge"
import { TIER_STYLES, type Tier } from "@/lib/api"
import { cn } from "@/lib/utils"

/** 등급 배지 — FLARE 타오름 / GLOW 어른거림 / SPARK 스침.
 *
 * 색만으로 구분하지 않는다. 등급 이름을 항상 함께 적어 색각 이상이나
 * 흑백 출력에서도 읽힌다 — 경보 시스템에서 색은 보조 신호여야 한다.
 */
export function TierBadge({
  tier,
  labelKo,
  className,
}: {
  tier: Tier
  labelKo?: string
  className?: string
}) {
  const style = TIER_STYLES[tier]
  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1.5 font-mono text-[11px] font-bold tracking-wider",
        style.text,
        style.bg,
        style.border,
        className,
      )}
    >
      <span className={cn("size-1.5 rounded-full", style.ring)} aria-hidden />
      {tier}
      {labelKo ? (
        <span className="font-sans font-medium tracking-normal opacity-70">
          {labelKo}
        </span>
      ) : null}
    </Badge>
  )
}
