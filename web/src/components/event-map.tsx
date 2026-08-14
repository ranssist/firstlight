import { useEffect } from "react"
import {
  Circle,
  CircleMarker,
  ImageOverlay,
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  useMap,
} from "react-leaflet"
import L from "leaflet"
import "leaflet/dist/leaflet.css"

import type { FireEvent, SiteInfo } from "@/lib/api"
import { demoHillshade, IS_DEMO } from "@/lib/demo"

/** 추정 발화점과 오차반경 — 작품설명서 표 Ⅱ-1 「React + Leaflet(지도)」.
 *
 * 원 두 개는 각각 CEP50(채움)과 CEP90(점선)이다. 점 하나만 찍으면 좌표가
 * 실제보다 확실해 보이는데, 이 시스템에서 불확실성은 감출 것이 아니라
 * 보여줄 값이다. Leaflet 의 Circle 은 **미터 단위 반경**을 그대로 받으므로
 * CEP 를 지도 축척에 맞춰 정확히 그린다 (자체 SVG 로는 근사만 됐다).
 *
 * 타일은 OpenStreetMap 을 쓴다. 관제실이 폐쇄망이면 여기를 VWorld 또는
 * 사내 타일 서버로 바꾸면 된다 — `TILE_URL` 한 줄이다.
 */

const TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
const TILE_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'

/** 진화대 아이콘 — Leaflet 기본 마커는 번들러에서 경로가 깨진다. */
const unitIcon = L.divIcon({
  className: "",
  html: `<div style="
    width:22px;height:22px;border-radius:50%;
    background:var(--dawn);border:2px solid var(--background);
    box-shadow:0 1px 4px rgba(0,0,0,.5);
    display:grid;place-items:center;font-size:11px;color:#fff;font-weight:700;
  ">진</div>`,
  iconSize: [22, 22],
  iconAnchor: [11, 11],
})

/** 이벤트가 바뀌면 보이는 범위를 다시 맞춘다. */
function FitBounds({ points }: { points: FireEvent[] }) {
  const map = useMap()
  useEffect(() => {
    const located = points.filter((p) => p.geo_ok && p.lat !== null)
    if (located.length === 0) return
    const bounds = L.latLngBounds(
      located.map((p) => [p.lat as number, p.lon as number]),
    )
    map.fitBounds(bounds.pad(0.35), { animate: false, maxZoom: 15 })
  }, [map, points])
  return null
}

export function EventMap({
  events,
  site,
  selectedId,
  onSelect,
}: {
  events: FireEvent[]
  site: SiteInfo | null
  selectedId: number | null
  onSelect: (id: number) => void
}) {
  const located = events.filter(
    (e): e is FireEvent & { lat: number; lon: number } =>
      e.geo_ok && e.lat !== null && e.lon !== null,
  )
  const centre: [number, number] = site
    ? [site.lat, site.lon]
    : located.length
      ? [located[0].lat, located[0].lon]
      : [36.4127, 128.7043]

  return (
    <div className="relative h-[320px] w-full overflow-hidden rounded-lg border">
      <MapContainer
        center={centre}
        zoom={13}
        scrollWheelZoom
        className="size-full"
        // Leaflet 기본 배경은 흰색이라 다크 테마에서 튄다.
        style={{ background: "var(--muted)" }}
      >
        {/* 데모 배포는 외부 타일을 받을 수 없다 (CSP). 대신 이 현장의 DEM 으로
            구운 음영기복도를 깐다 — 산불 조기탐지에서 읽어야 할 배경은
            도로망이 아니라 능선과 계곡이라 오히려 맞는 배경이다. */}
        {IS_DEMO ? (
          <ImageOverlay url={demoHillshade.image} bounds={demoHillshade.bounds} />
        ) : (
          <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} maxZoom={19} />
        )}
        <FitBounds points={events} />

        {/* 최근접 진화대 */}
        {(site?.response_units ?? []).map((unit) => (
          <Marker key={unit.name} position={[unit.lat, unit.lon]} icon={unitIcon}>
            <Popup>
              <b>{unit.name}</b>
              <br />
              진화대
            </Popup>
          </Marker>
        ))}

        {/* 탐지 이벤트 — CEP90 점선, CEP50 채움, 중심점 */}
        {located.map((event) => {
          const selected = selectedId === event.event_id
          return (
            <div key={event.event_id}>
              <Circle
                center={[event.lat, event.lon]}
                radius={event.cep90_m ?? 60}
                pathOptions={{
                  color: event.tier_colour,
                  weight: 1,
                  opacity: 0.5,
                  dashArray: "4 4",
                  fillOpacity: 0.05,
                }}
              />
              <Circle
                center={[event.lat, event.lon]}
                radius={event.cep50_m ?? 30}
                pathOptions={{
                  color: event.tier_colour,
                  weight: 1,
                  fillColor: event.tier_colour,
                  fillOpacity: 0.2,
                }}
              />
              <CircleMarker
                center={[event.lat, event.lon]}
                radius={selected ? 8 : 5}
                pathOptions={{
                  color: "#ffffff",
                  weight: 2,
                  fillColor: event.tier_colour,
                  fillOpacity: 1,
                }}
                eventHandlers={{ click: () => onSelect(event.event_id) }}
              >
                <Popup>
                  <b>{event.tier}</b> · {event.tier_label_ko}
                  <br />
                  {event.lat.toFixed(5)}N {event.lon.toFixed(5)}E
                  <br />
                  오차반경 {Math.round(event.cep50_m ?? 0)} m
                </Popup>
              </CircleMarker>
            </div>
          )
        })}
      </MapContainer>

      {located.length === 0 && (
        <div className="bg-background/85 text-muted-foreground pointer-events-none absolute inset-x-0 bottom-0 z-[500] px-3 py-2 text-center text-xs">
          좌표가 발행된 이벤트가 없습니다
        </div>
      )}
    </div>
  )
}
