"""탐지 스냅샷 — 작품설명서 「관제 대시보드 화면 설계」의 "연기 스냅샷".

요원의 1클릭 판정이 성립하려면 그 시점 그 자리의 그림이 있어야 한다.
크롭이 엉뚱한 곳을 잘라내면 판정 근거가 통째로 틀어지므로 기하를 검증한다.
"""

import numpy as np
import pytest

from firstlight.events.snapshot import OUTPUT_WIDTH, crop_detection, save_snapshot


def frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """위치를 확인할 수 있도록 좌표가 값에 인코딩된 프레임."""
    img = np.zeros((height, width, 3), np.uint8)
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    img[:, :, 0] = xs[None, :]                       # B 채널 = x 위치
    ys = np.linspace(0, 255, height, dtype=np.uint8)
    img[:, :, 1] = ys[:, None]                       # G 채널 = y 위치
    return img


# ------------------------------------------------------------------ 크롭


def test_output_is_square_and_fixed_size():
    """카드에서 크기가 들쭉날쭉하면 목록을 훑기 어렵다."""
    crop = crop_detection(frame(), (100, 100, 140, 180))
    assert crop.shape == (OUTPUT_WIDTH, OUTPUT_WIDTH, 3)


def test_crop_includes_context_around_box():
    """박스만 자르면 안개인지 연기인지 알 수 없다 — 주변이 필요하다."""
    box = (600, 300, 640, 340)          # 40px 박스
    crop = crop_detection(frame(), box, draw_box=False)

    # B 채널이 x 위치를 담고 있으므로 크롭이 덮은 x 범위를 되읽을 수 있다.
    covered = crop[:, :, 0]
    x_span = (covered.max() - covered.min()) / 255.0 * 1280
    assert x_span > 60, f"맥락 없이 박스만 잘렸다 (x 범위 {x_span:.0f}px)"


def test_crop_centres_on_detection():
    """탐지가 크롭 한가운데 있어야 요원이 어디를 볼지 헤매지 않는다."""
    crop = crop_detection(frame(), (600, 300, 660, 360), draw_box=False)
    centre_x = crop[OUTPUT_WIDTH // 2, OUTPUT_WIDTH // 2, 0] / 255.0 * 1280
    assert centre_x == pytest.approx(630, abs=40)


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 30, 30),                  # 좌상단 모서리
        (1250, 690, 1280, 720),          # 우하단 모서리
        (0, 350, 40, 390),               # 좌측 가장자리
    ],
)
def test_edge_detections_still_produce_full_crop(box):
    """가장자리 탐지도 잘리지 않고 프레임 안으로 밀어 넣어야 한다."""
    crop = crop_detection(frame(), box)
    assert crop is not None
    assert crop.shape == (OUTPUT_WIDTH, OUTPUT_WIDTH, 3)


def test_tier_colour_is_drawn():
    """박스 색이 등급을 나타낸다 — 어디를 봐야 하는지 표시다."""
    plain = crop_detection(frame(), (600, 300, 660, 360), draw_box=False)
    marked = crop_detection(frame(), (600, 300, 660, 360), tier="FLARE")
    assert not np.array_equal(plain, marked), "박스가 그려지지 않았다"


def test_degenerate_box_is_rejected():
    """비정상 박스로 파이프라인이 죽으면 안 된다."""
    assert crop_detection(frame(), (5, 5, 5, 5)) is not None   # 최소 크롭으로 확대
    assert crop_detection(np.zeros((4, 4, 3), np.uint8), (0, 0, 2, 2)) is None


# ------------------------------------------------------------------ 저장


def test_save_writes_jpeg_and_returns_filename(tmp_path):
    name = save_snapshot(frame(), (600, 300, 660, 360), tmp_path, "uiseong-t1-f7")
    assert name == "uiseong-t1-f7.jpg"

    path = tmp_path / name
    assert path.is_file()
    assert path.read_bytes()[:2] == b"\xff\xd8"      # JPEG 매직 바이트


def test_each_event_gets_its_own_file(tmp_path):
    """트랙 단위로 덮어쓰면 과거 이벤트가 나중 프레임 그림을 가리킨다."""
    a = save_snapshot(frame(), (600, 300, 660, 360), tmp_path, "uiseong-t1-f1")
    b = save_snapshot(frame(), (600, 300, 700, 400), tmp_path, "uiseong-t1-f9")
    assert a != b
    assert (tmp_path / a).is_file() and (tmp_path / b).is_file()


# -------------------------------------------------------------------- API


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import firstlight.api.main as api
    from firstlight.api.main import create_app

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    monkeypatch.setattr(api, "SNAPSHOT_DIR", snapshots)
    save_snapshot(frame(), (600, 300, 660, 360), snapshots, "demo")
    return TestClient(create_app(str(tmp_path / "events.db"), tmp_path / "s.json"))


def test_snapshot_endpoint_serves_image(client):
    response = client.get("/api/snapshots/demo.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_missing_snapshot_is_404(client):
    assert client.get("/api/snapshots/없는파일.jpg").status_code == 404


@pytest.mark.parametrize(
    "attack",
    ["../../pyproject.toml", "..%2F..%2Fetc%2Fpasswd", "sub/dir.jpg"],
)
def test_path_traversal_is_rejected(client, attack):
    """사용자 입력이 파일 경로가 되는 지점 — 반드시 막아야 한다."""
    assert client.get(f"/api/snapshots/{attack}").status_code in (400, 404)
