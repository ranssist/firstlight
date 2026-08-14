"""관제 대시보드 백엔드.

프런트엔드는 `web/` 의 React + shadcn/ui 앱이고, 이 서버는 빌드 산출물
(`web/dist`)을 서빙한다.

개발할 때는 두 프로세스를 나란히 띄운다:
    uv run firstlight serve          # API + 빌드된 앱  (:8000)
    cd web && npm run dev            # HMR 개발 서버     (:5173)
Vite 설정이 `/api` 를 :8000 으로 프록시하므로, 개발 서버에서도 실제
이벤트 DB 를 그대로 본다.

환류 루프가 여기서 닫힌다:
    GLOW 큐 → 사람이 확인/오탐 판정 → POST /events/{id}/label
    → 라벨 누적 → POST /scorer/retrain → 가중치 갱신
"""

# `from __future__ import annotations` 를 쓰지 않는다.
#
# 그걸 켜면 함수 주석이 전부 문자열이 되고, FastAPI 는 라우트 시그니처를
# `get_type_hints()` 로 **모듈 전역에서** 되살린다. fastapi 를 함수 안에서
# 지연 임포트하는 이 모듈에서는 `WebSocket` 이 전역에 없으므로 해석에
# 실패하고, WebSocket 인자가 주입되지 않아 연결이 곧바로 끊긴다.
# (실제로 대시보드가 "재연결 중"에서 멈춰 이 원인을 찾는 데 시간을 썼다.)
#
# 주석을 실제 객체로 두면 def 시점에 지역 스코프에서 해석되므로 문제가 없다.
import json
from pathlib import Path

from firstlight.events.models import EventLabel
from firstlight.events.store import EventStore
from firstlight.verify.features import VerifierMode
from firstlight.verify.scorer import SequenceScorer

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
DIST_DIR = WEB_DIR / "dist"
DEFAULT_SCORER_PATH = Path("models/scorer_sparse.json")
SNAPSHOT_DIR = Path("data/snapshots")

BUILD_MISSING_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>FIRSTLIGHT 관제</title>
<style>
  body { font-family: system-ui, "Malgun Gothic", sans-serif; background:#0B1026;
         color:#E8ECF7; display:grid; place-items:center; min-height:100vh; margin:0; }
  main { max-width: 34rem; padding: 2rem; line-height: 1.7; }
  h1 { letter-spacing:.18em; color:#FF6B35; font-size:1.1rem; }
  code { background:#1B2244; padding:.15rem .4rem; border-radius:4px;
         font-family: Consolas, monospace; }
  pre { background:#141A33; padding:1rem; border-radius:8px; overflow-x:auto; }
</style></head>
<body><main>
  <h1>FIRSTLIGHT</h1>
  <p>대시보드가 아직 빌드되지 않았습니다. <code>web/dist</code> 가 없습니다.</p>
  <pre>cd web
npm install
npm run build</pre>
  <p>API 는 정상 동작 중입니다 — <code>/api/docs</code> 에서 확인할 수 있습니다.</p>
</main></body></html>
"""


def create_app(
    db_path: str = "data/events.db",
    scorer_path: Path = DEFAULT_SCORER_PATH,
    site_name: str | None = None,
):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse

    store = EventStore(db_path)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        store.close()

    app = FastAPI(title="FIRSTLIGHT 관제", docs_url="/api/docs", lifespan=lifespan)
    app.state.store = store
    app.state.scorer_path = Path(scorer_path)

    # ------------------------------------------------------------ 정적

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        path = DIST_DIR / "index.html"
        if not path.exists():
            # 빌드가 없다고 API 까지 죽이지는 않는다. 무엇을 해야 하는지 알려준다.
            return BUILD_MISSING_HTML
        return path.read_text(encoding="utf-8")

    @app.get("/api/snapshots/{filename}")
    def snapshot(filename: str):
        """탐지 시점 크롭 이미지 (작품설명서의 "연기 스냅샷").

        경로 조작을 막기 위해 파일명만 받고 디렉터리 구분자는 거부한다 —
        사용자 입력이 파일 경로가 되는 지점이라 반드시 막아야 한다.
        """
        from fastapi.responses import FileResponse

        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(400, "잘못된 파일명")
        path = (SNAPSHOT_DIR / filename).resolve()
        if not path.is_file() or SNAPSHOT_DIR.resolve() not in path.parents:
            raise HTTPException(404, "스냅샷이 없다")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/favicon.svg")
    def favicon():
        from fastapi.responses import FileResponse

        path = DIST_DIR / "favicon.svg"
        if not path.exists():
            raise HTTPException(404, "favicon 이 없다")
        return FileResponse(path, media_type="image/svg+xml")

    # ------------------------------------------------------------ 이벤트

    @app.get("/api/events")
    def list_events(
        tier: str | None = None, label: str | None = None, limit: int = 200
    ) -> JSONResponse:
        events = store.list(tier=tier, label=label, site=site_name, limit=limit)
        return JSONResponse([e.to_public_dict() for e in events])

    @app.get("/api/events/{event_id}")
    def get_event(event_id: int) -> JSONResponse:
        event = store.get(event_id)
        if event is None:
            raise HTTPException(404, "이벤트를 찾을 수 없다")
        return JSONResponse(event.to_public_dict())

    @app.post("/api/events/{event_id}/label")
    def set_label(event_id: int, payload: dict) -> JSONResponse:
        """관제 요원의 판정. 이 호출이 재학습 데이터를 만든다."""
        raw = payload.get("label", "")
        try:
            label = EventLabel(raw)
        except ValueError as exc:
            raise HTTPException(
                400, f"label 은 {[e.value for e in EventLabel]} 중 하나여야 한다"
            ) from exc
        if not store.set_label(event_id, label):
            raise HTTPException(404, "이벤트를 찾을 수 없다")

        _append_label_log(event_id, label)
        return JSONResponse({"ok": True, "event_id": event_id, "label": label.value})

    @app.get("/api/site")
    def site_info() -> JSONResponse:
        """지도 초기화에 필요한 현장 정보.

        작품설명서 「관제 대시보드 화면 설계」의 지도 패널은 탐지 이벤트
        외에 **최근접 진화대 위치**도 겹쳐 표시하도록 되어 있다.
        """
        from firstlight.config import SiteConfig

        if not site_name:
            return JSONResponse({"name": None, "response_units": []})
        try:
            cfg = SiteConfig.load(site_name)
        except FileNotFoundError:
            return JSONResponse({"name": site_name, "response_units": []})
        return JSONResponse(
            {
                "name": cfg.name,
                "label": cfg.label,
                "lat": cfg.lat,
                "lon": cfg.lon,
                "bbox": list(cfg.bbox),
                "response_units": cfg.response_units,
            }
        )

    @app.websocket("/api/ws")
    async def events_socket(websocket: WebSocket) -> None:
        """실시간 이벤트 푸시 (작품설명서 표 Ⅱ-1 「WebSocket」).

        폴링을 대체한다. 관제 화면에서 5초 지연은 30분 골든타임 대비
        작지만, 경보 시스템이 "새로고침을 기다린다"는 인상을 주면 안 된다.

        서버가 이벤트 총계를 감시하다가 바뀌면 밀어준다. DB 가 sqlite 라
        트리거를 걸 수 없어 짧은 주기로 확인하되, **변화가 있을 때만**
        전송한다 — 연결 하나당 초당 한 번 정수 하나를 읽는 비용이다.
        """
        import asyncio
        import logging

        from starlette.websockets import WebSocketDisconnect

        await websocket.accept()
        last_seen = -1
        try:
            while True:
                events = store.list(site=site_name, limit=300)
                marker = events[0].event_id if events else 0
                fingerprint = (marker, len(events))
                if fingerprint != last_seen:
                    last_seen = fingerprint
                    await websocket.send_json(
                        {
                            "type": "events",
                            "events": [e.to_public_dict() for e in events],
                        }
                    )
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return                       # 클라이언트가 떠난 것 — 정상 흐름
        except Exception:
            # 여기를 조용히 삼키면 대시보드가 "재연결 중"에서 영원히 멈춘 채
            # 이유를 알 수 없게 된다. 실제로 그렇게 한 번 막혔다.
            logging.getLogger("firstlight.api").exception("이벤트 스트림 실패")
            raise

    @app.post("/api/events/{event_id}/response")
    def set_response(event_id: int, payload: dict) -> JSONResponse:
        """대응 상태 갱신 (접수-출동-진화).

        되돌리기는 막는다 — 이력 타임라인은 대응 지연을 재기 위한 기록이라
        순서가 뒤집히면 의미가 없다. 잘못 눌렀으면 로그로 남는 편이 맞다.
        """
        from firstlight.events.models import ResponseStatus

        raw = payload.get("response", "")
        try:
            status = ResponseStatus(raw)
        except ValueError as exc:
            raise HTTPException(
                400, f"response 는 {[s.value for s in ResponseStatus]} 중 하나여야 한다"
            ) from exc

        event = store.get(event_id)
        if event is None:
            raise HTTPException(404, "이벤트를 찾을 수 없다")
        if status.order <= event.response.order:
            raise HTTPException(
                400,
                f"대응 상태는 되돌릴 수 없다 (현재 {event.response.label_ko} "
                f"→ 요청 {status.label_ko})",
            )

        store.set_response(event_id, status)
        return JSONResponse(
            {"ok": True, "event_id": event_id, "response": status.value}
        )

    @app.get("/api/summary")
    def summary() -> JSONResponse:
        counts = store.counts_by_tier(site=site_name)
        labelled = store.labelled_events()
        scorer = _load_scorer(app.state.scorer_path)
        return JSONResponse(
            {
                "counts": counts,
                "total": sum(counts.values()),
                "labelled": len(labelled),
                "queue": counts.get("GLOW", 0),
                "scorer": {
                    "fitted": scorer.is_fitted,
                    "n_train": scorer.n_train,
                    "mode": scorer.mode.value,
                    "tau_high": scorer.tau_high,
                    "tau_low": scorer.tau_low,
                },
            }
        )

    # ------------------------------------------------------------ 재학습

    @app.post("/api/scorer/retrain")
    def retrain() -> JSONResponse:
        """누적된 라벨로 스코어러를 다시 맞춘다.

        로지스틱 회귀라 CPU 에서 1초 안에 끝난다. 소개서 §4③ 의 "쓸수록
        오탐이 줄어드는 구조"가 이 엔드포인트다.
        """
        from firstlight.verify.features import TrackFeatures

        events = store.labelled_events()
        features, labels = [], []
        for event in events:
            if not event.features:
                continue
            data = dict(event.features)
            data.pop("mode", None)
            features.append(TrackFeatures(**data))
            labels.append(1 if event.label is EventLabel.CONFIRMED else 0)

        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos < 8 or n_neg < 8:
            raise HTTPException(
                400,
                f"클래스당 최소 8개가 필요하다 (양성 {n_pos} / 음성 {n_neg}). "
                f"GLOW 큐를 더 판정해야 한다.",
            )

        scorer = _load_scorer(app.state.scorer_path)
        scorer.fit(features, labels)
        scorer.save(app.state.scorer_path)
        return JSONResponse(
            {
                "ok": True,
                "n_train": scorer.n_train,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "weights": scorer.weights,
            }
        )

    # 정적 자산은 **맨 마지막에** 붙인다. 라우트는 등록 순서대로 매칭되므로
    # 위의 /api 경로들이 항상 먼저 잡힌다.
    if (DIST_DIR / "assets").is_dir():
        import mimetypes

        from fastapi.staticfiles import StaticFiles

        # Windows 레지스트리에는 woff2 가 없어 application/octet-stream 으로
        # 나간다. 브라우저는 매직 바이트로 알아채지만, 프록시 캐싱과
        # CSP font-src 검사는 Content-Type 을 본다.
        mimetypes.add_type("font/woff2", ".woff2")
        mimetypes.add_type("font/woff", ".woff")

        app.mount(
            "/assets",
            StaticFiles(directory=DIST_DIR / "assets"),
            name="assets",
        )

    return app


# --------------------------------------------------------------------------


def _load_scorer(path: Path) -> SequenceScorer:
    if path.exists():
        return SequenceScorer.load(path)
    return SequenceScorer(mode=VerifierMode.SPARSE)


def _append_label_log(event_id: int, label: EventLabel) -> None:
    """라벨을 추가 전용 로그에도 남긴다.

    DB 가 날아가도 사람이 들인 판정 노력은 남아야 한다. 사람의 라벨은
    이 시스템에서 가장 비싼 자원이다.
    """
    path = Path("data/labels.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event_id": event_id, "label": label.value},
                            ensure_ascii=False) + "\n")
