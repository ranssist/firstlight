import { useEffect, useRef, useState } from "react"

import type { FireEvent } from "@/lib/api"

/** 실시간 이벤트 스트림 — 작품설명서 표 Ⅱ-1 「WebSocket」.
 *
 * 폴링 대신 서버가 밀어준다. 다만 WebSocket 은 끊긴다 — 프록시 타임아웃,
 * 서버 재시작, 노트북 절전. 끊긴 채로 조용히 멈춘 관제 화면은 빈 화면보다
 * 위험하므로, 연결 상태를 밖으로 내보내 UI 가 표시하게 하고 자동 재연결한다.
 *
 * `onMessage` 로 받은 이벤트를 넘긴다. 재연결 대기 중에는 호출부가 폴링으로
 * 물러설 수 있도록 `connected` 를 false 로 유지한다.
 */
export function useEventStream(onEvents: ((events: FireEvent[]) => void) | null) {
  const [connected, setConnected] = useState(false)
  // onEvents 가 매 렌더 새 함수여도 소켓을 다시 열지 않도록 ref 에 담는다.
  const handler = useRef(onEvents)
  handler.current = onEvents

  useEffect(() => {
    // null 이면 서버가 없는 정적 배포다 — 소켓을 열지 않는다.
    if (handler.current === null) return

    let socket: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let closed = false
    let backoffMs = 1000

    const connect = () => {
      if (closed) return
      const scheme = location.protocol === "https:" ? "wss:" : "ws:"
      socket = new WebSocket(`${scheme}//${location.host}/api/ws`)

      socket.onopen = () => {
        setConnected(true)
        backoffMs = 1000
      }
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data)
          if (payload.type === "events") handler.current?.(payload.events)
        } catch {
          // 형식이 깨진 프레임 하나 때문에 스트림을 끊지는 않는다.
        }
      }
      socket.onclose = () => {
        setConnected(false)
        if (closed) return
        retry = setTimeout(connect, backoffMs)
        backoffMs = Math.min(backoffMs * 2, 15000)
      }
      socket.onerror = () => socket?.close()
    }

    connect()
    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      socket?.close()
    }
  }, [])

  return connected
}
