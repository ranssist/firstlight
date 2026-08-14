import { useCallback, useEffect, useMemo, useState } from "react"
import { Brain, RefreshCw } from "lucide-react"
import { toast } from "sonner"

import { DashboardHeader } from "@/components/dashboard-header"
import { EventCard } from "@/components/event-card"
import { EventDetail } from "@/components/event-detail"
import { EventMap } from "@/components/event-map"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TooltipProvider } from "@/components/ui/tooltip"
import {
  api,
  RESPONSE_LABEL_KO,
  TIER_ORDER,
  type EventLabel,
  type FireEvent,
  type ResponseStatus,
  type SiteInfo,
  type Summary,
  type Tier,
} from "@/lib/api"
import { demoEvents, demoSite, demoSummary, IS_DEMO } from "@/lib/demo"
import { useEventStream } from "@/lib/use-event-stream"

/** WebSocket 이 끊겼을 때만 쓰는 폴백 주기. 연결돼 있으면 폴링하지 않는다. */
const FALLBACK_POLL_MS = 5000

export default function App() {
  const [events, setEvents] = useState<FireEvent[] | null>(null)
  const [summary, setSummary] = useState<Summary | null>(null)
  const [site, setSite] = useState<SiteInfo | null>(null)
  const [filter, setFilter] = useState<Tier | "ALL">("ALL")
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [pendingId, setPendingId] = useState<number | null>(null)
  const [retraining, setRetraining] = useState(false)
  const [responsePending, setResponsePending] = useState(false)

  // 이벤트는 WebSocket 이 밀어주고, 집계는 REST 로 가져온다.
  // 데모 배포에는 서버가 없으므로 소켓을 아예 열지 않는다.
  const connected = useEventStream(IS_DEMO ? null : setEvents)

  const refreshSummary = useCallback(async () => {
    try {
      setSummary(await api.summary())
    } catch {
      // 요약이 잠깐 늦는 것은 화면을 막을 이유가 아니다.
    }
  }, [])

  const refresh = useCallback(async () => {
    try {
      const [nextEvents, nextSummary] = await Promise.all([
        api.events(),
        api.summary(),
      ])
      setEvents(nextEvents)
      setSummary(nextSummary)
    } catch (error) {
      toast.error("관제 서버에 연결할 수 없습니다", {
        description: error instanceof Error ? error.message : undefined,
      })
    }
  }, [])

  useEffect(() => {
    if (IS_DEMO) {
      setEvents(demoEvents)
      setSummary(demoSummary)
      setSite(demoSite)
      return
    }
    void refresh()
    void api.site().then(setSite).catch(() => setSite(null))
  }, [refresh])

  useEffect(() => {
    if (IS_DEMO) return
    // 소켓이 살아 있으면 이벤트는 푸시로 오므로 집계만 갱신한다.
    const timer = setInterval(
      () => void (connected ? refreshSummary() : refresh()),
      FALLBACK_POLL_MS,
    )
    return () => clearInterval(timer)
  }, [connected, refresh, refreshSummary])

  const shown = useMemo(
    () => (events ?? []).filter((e) => filter === "ALL" || e.tier === filter),
    [events, filter],
  )
  const selected = useMemo(
    () => (events ?? []).find((e) => e.event_id === selectedId) ?? null,
    [events, selectedId],
  )

  const handleLabel = async (eventId: number, label: EventLabel) => {
    // 데모에는 서버가 없다. 화면은 실제와 같이 반응시키되, 저장된 척은 하지 않는다.
    if (IS_DEMO) {
      setEvents((prev) =>
        (prev ?? []).map((e) => (e.event_id === eventId ? { ...e, label } : e)),
      )
      setSummary((prev) =>
        prev ? { ...prev, labelled: prev.labelled + 1 } : prev,
      )
      toast.success(label === "confirmed" ? "실제 연기로 판정" : "오탐으로 판정", {
        description: "데모에서는 저장되지 않습니다 — 새로고침하면 되돌아갑니다.",
      })
      return
    }

    setPendingId(eventId)
    try {
      await api.setLabel(eventId, label)
      toast.success(label === "confirmed" ? "실제 연기로 판정" : "오탐으로 판정", {
        description: "이 판정은 스코어러 재학습 입력이 됩니다.",
      })
      await refresh()
    } catch (error) {
      toast.error("판정을 저장하지 못했습니다", {
        description: error instanceof Error ? error.message : undefined,
      })
    } finally {
      setPendingId(null)
    }
  }

  const handleAdvanceResponse = async (
    eventId: number,
    status: ResponseStatus,
  ) => {
    const stamp = Date.now() / 1000
    const apply = (e: FireEvent) =>
      e.event_id === eventId
        ? {
            ...e,
            response: status,
            response_label_ko: RESPONSE_LABEL_KO[status],
            response_history: [...(e.response_history ?? []), { status, at: stamp }],
          }
        : e

    if (IS_DEMO) {
      setEvents((prev) => (prev ?? []).map(apply))
      toast.success(`대응 상태: ${RESPONSE_LABEL_KO[status]}`, {
        description: "데모에서는 저장되지 않습니다.",
      })
      return
    }

    setResponsePending(true)
    try {
      await api.setResponse(eventId, status)
      setEvents((prev) => (prev ?? []).map(apply))
      await refresh()
    } catch (error) {
      toast.error("대응 상태를 기록하지 못했습니다", {
        description: error instanceof Error ? error.message : undefined,
      })
    } finally {
      setResponsePending(false)
    }
  }

  const handleRetrain = async () => {
    if (IS_DEMO) {
      // 실제 서버가 거절하는 것과 같은 사유를 보인다 — 데모라고 성공한
      // 척하면 "쓸수록 오탐이 준다"는 주장을 검증 없이 보여주는 셈이다.
      toast.warning("재학습하지 않았습니다", {
        description:
          "클래스당 최소 8개가 필요합니다. 표본이 모자라면 서버가 거절합니다.",
      })
      return
    }

    setRetraining(true)
    try {
      const result = await api.retrain()
      toast.success("스코어러 재학습 완료", {
        description: `표본 ${result.n_train}개 (양성 ${result.n_positive} / 음성 ${result.n_negative})`,
      })
      await refresh()
    } catch (error) {
      // 표본이 모자라면 백엔드가 거절한다 — 조용히 퇴화한 모델을 만드는
      // 것보다 낫다. 사유를 그대로 보여준다.
      toast.warning("재학습하지 않았습니다", {
        description: error instanceof Error ? error.message : undefined,
      })
    } finally {
      setRetraining(false)
    }
  }

  const scorer = summary?.scorer

  return (
    <TooltipProvider delayDuration={200}>
      <div className="bg-background text-foreground min-h-svh">
        <DashboardHeader summary={summary} connected={connected} />

        <main className="mx-auto grid max-w-[1400px] gap-4 p-5 lg:grid-cols-[1.25fr_1fr]">
          {/* ── 왼쪽: 지도 + 선택 이벤트 ─────────────────────────── */}
          <div className="flex flex-col gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  지도 · 추정 발화점과 오차반경
                </CardTitle>
              </CardHeader>
              <CardContent>
                <EventMap
                  events={events ?? []}
                  site={site}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                />
              </CardContent>
            </Card>

            <Card className="flex-1">
              <CardHeader>
                <CardTitle className="text-sm">선택한 이벤트</CardTitle>
              </CardHeader>
              <CardContent>
                <EventDetail
                  event={selected}
                  responsePending={responsePending}
                  onAdvanceResponse={
                    selected
                      ? (status) => void handleAdvanceResponse(selected.event_id, status)
                      : undefined
                  }
                />
              </CardContent>
            </Card>
          </div>

          {/* ── 오른쪽: 이벤트 큐 ────────────────────────────────── */}
          <Card className="flex max-h-[calc(100svh-7rem)] flex-col lg:sticky lg:top-[4.5rem]">
            <CardHeader className="gap-3">
              <div className="flex items-center justify-between gap-3">
                <CardTitle className="text-sm">이벤트 큐</CardTitle>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRetrain}
                  disabled={retraining}
                  className="h-7 text-xs"
                >
                  {retraining ? (
                    <RefreshCw className="animate-spin" aria-hidden />
                  ) : (
                    <Brain aria-hidden />
                  )}
                  스코어러 재학습
                </Button>
              </div>

              <Tabs
                value={filter}
                onValueChange={(value) => setFilter(value as Tier | "ALL")}
              >
                <TabsList className="w-full">
                  <TabsTrigger value="ALL" className="text-xs">
                    전체
                  </TabsTrigger>
                  {TIER_ORDER.map((tier) => (
                    <TabsTrigger key={tier} value={tier} className="text-xs">
                      {tier}
                      <span className="text-muted-foreground ml-1 font-mono tabular-nums">
                        {summary?.counts[tier] ?? 0}
                      </span>
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>
            </CardHeader>

            <CardContent className="min-h-0 flex-1 p-0">
              <ScrollArea className="h-full">
                <div className="flex flex-col gap-2 px-6 pb-4">
                  {events === null ? (
                    Array.from({ length: 4 }).map((_, i) => (
                      <Skeleton key={i} className="h-32 w-full" />
                    ))
                  ) : shown.length === 0 ? (
                    <p className="text-muted-foreground py-8 text-center text-sm">
                      이벤트가 없습니다.
                    </p>
                  ) : (
                    shown.map((event) => (
                      <EventCard
                        key={event.event_id}
                        event={event}
                        selected={event.event_id === selectedId}
                        pending={pendingId === event.event_id}
                        onSelect={() => setSelectedId(event.event_id)}
                        onLabel={(label) => void handleLabel(event.event_id, label)}
                      />
                    ))
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </main>

        <footer className="text-muted-foreground mx-auto flex max-w-[1400px] flex-wrap gap-x-6 gap-y-1.5 border-t px-5 py-4 text-xs">
          <span>
            스코어러:{" "}
            {scorer
              ? scorer.fitted
                ? `학습됨 (표본 ${scorer.n_train}, ${scorer.mode})`
                : `사전값 — 아직 학습되지 않음 (${scorer.mode})`
              : "—"}
          </span>
          <span>FLARE 자동 경보 · GLOW 사람 확인 · SPARK 로그만</span>
          <span>좌표를 못 내면 FLARE 로 올리지 않습니다</span>
        </footer>
      </div>
    </TooltipProvider>
  )
}
