/** 데모 모드 — 백엔드 없이 도는 정적 배포용.
 *
 * 아티팩트/정적 호스팅에는 FastAPI 가 없다. 실제 파이프라인이 만든 이벤트를
 * 빌드 시점에 구워 넣고, 판정·필터·지도는 그대로 동작시킨다. 라벨링만
 * 메모리에 머문다 — 새로고침하면 사라진다.
 *
 * `scripts/build_demo_data.py` 가 데이터를 만든다. 지어낸 값이 아니라
 * FIgLib 시퀀스를 실제로 돌린 결과다.
 */

import type { FireEvent, SiteInfo, Summary } from "@/lib/api"

import eventsJson from "@/demo/events.json"
import summaryJson from "@/demo/summary.json"
import siteJson from "@/demo/site.json"
import hillshadeJson from "@/demo/hillshade.json"

export const IS_DEMO = import.meta.env.VITE_DEMO === "1"

export const demoEvents = eventsJson as unknown as FireEvent[]
export const demoSummary = summaryJson as unknown as Summary
export const demoSite = siteJson as unknown as SiteInfo

/** DEM 에서 만든 음영기복도. 아티팩트 CSP 가 외부 타일을 막으므로
 *  지도 배경을 번들에 넣는다. 산불에는 도로망보다 지형이 맞기도 하다. */
export const demoHillshade = hillshadeJson as unknown as {
  image: string
  bounds: [[number, number], [number, number]]
}
