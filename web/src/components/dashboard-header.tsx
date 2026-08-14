import { FlaskConical, Monitor, Moon, Sun } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { TIER_STYLES, TIER_ORDER, type Summary } from "@/lib/api"
import { IS_DEMO } from "@/lib/demo"
import { useTheme } from "@/lib/theme"
import { cn } from "@/lib/utils"

const TIER_KO = { FLARE: "타오름", GLOW: "어른거림", SPARK: "스침" } as const

function Stat({
  value,
  label,
  hint,
  className,
}: {
  value: number
  label: string
  hint: string
  className?: string
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="cursor-default text-right">
          <div
            className={cn(
              "font-mono text-xl leading-none font-bold tabular-nums",
              className,
            )}
          >
            {value}
          </div>
          <div className="text-muted-foreground mt-1 text-[11px] tracking-wide">
            {label}
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  )
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const next = theme === "system" ? "light" : theme === "light" ? "dark" : "system"
  const Icon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor
  const labels = { system: "시스템 설정", light: "밝은 테마", dark: "어두운 테마" }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setTheme(next)}
          aria-label={`테마 전환 (현재 ${labels[theme]})`}
        >
          <Icon />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{labels[theme]}</TooltipContent>
    </Tooltip>
  )
}

/** 실시간 연결 표시.
 *
 * 끊긴 채로 조용히 멈춘 관제 화면은 빈 화면보다 위험하다 — 요원은 "새 경보가
 * 없다"고 읽지만 실제로는 "못 받고 있다"이기 때문이다. 상태를 항상 보인다.
 *
 * 데모 배포에서는 "실시간"이라고 말하면 안 된다. 고정된 스냅샷임을 밝힌다. */
function ConnectionDot({ connected }: { connected: boolean }) {
  if (IS_DEMO) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="border-dawn/40 text-dawn cursor-default gap-1.5 text-[11px]"
          >
            <FlaskConical className="size-3" aria-hidden />
            데모 스냅샷
          </Badge>
        </TooltipTrigger>
        <TooltipContent className="max-w-72">
          실제 FIgLib 시퀀스를 파이프라인에 통과시킨 결과를 고정해 둔 화면입니다.
          실시간 수신이 아니며, 판정을 눌러도 저장되지 않습니다.
        </TooltipContent>
      </Tooltip>
    )
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex cursor-default items-center gap-1.5">
          <span
            className={cn(
              "size-2 rounded-full",
              connected ? "bg-flare" : "bg-muted-foreground",
            )}
            aria-hidden
          />
          <span className="text-muted-foreground text-[11px]">
            {connected ? "실시간" : "재연결 중"}
          </span>
        </div>
      </TooltipTrigger>
      <TooltipContent>
        {connected
          ? "WebSocket 연결됨 — 경보가 즉시 표시됩니다"
          : "연결이 끊겨 5초 폴링으로 동작 중입니다"}
      </TooltipContent>
    </Tooltip>
  )
}

export function DashboardHeader({
  summary,
  connected,
}: {
  summary: Summary | null
  connected: boolean
}) {
  return (
    <header className="bg-background/80 sticky top-0 z-[600] border-b backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-5 gap-y-3 px-5 py-3.5">
        <div className="flex items-baseline gap-3">
          <span className="text-dawn text-lg font-extrabold tracking-[0.2em]">
            FIRSTLIGHT
          </span>
          <span className="text-muted-foreground hidden text-xs sm:inline">
            가장 먼저 보는 눈 · 관제
          </span>
        </div>
        <ConnectionDot connected={connected} />

        <div className="ml-auto flex items-center gap-5">
          {TIER_ORDER.map((tier) => (
            <Stat
              key={tier}
              value={summary?.counts[tier] ?? 0}
              label={tier}
              hint={`${TIER_KO[tier]} — ${
                tier === "FLARE"
                  ? "자동 경보 + 좌표 확정"
                  : tier === "GLOW"
                    ? "사람이 확인해야 하는 큐"
                    : "알림 없이 로그만, 재학습 데이터"
              }`}
              className={TIER_STYLES[tier].text}
            />
          ))}
          <Separator orientation="vertical" className="h-8" />
          <Stat
            value={summary?.labelled ?? 0}
            label="판정됨"
            hint="관제 요원이 판정한 이벤트 수 — 재학습 입력이 된다"
          />
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
