"""수치표고모델 — Copernicus GLO-30 COG 를 메모리에 올리고 벡터화 표고 조회.

왜 메모리에 통째로 올리는가:
    광선-지형 교차는 광선 하나당 수백 회의 표고 조회를 한다. 몬테카를로
    오차전파는 여기에 200배를 곱한다. 파일 I/O 를 그때마다 하면 수십만 회
    윈도우 읽기가 되어 못 쓴다. AOI 영역(예: 0.3°x0.3°)은 30m 격자에서
    1100x1100 float32 ≈ 5MB 에 불과하므로 그냥 상주시킨다.

수직 기준면 (조용히 결과를 망치는 함정):
    Copernicus DEM 의 표고는 **EGM2008 지오이드 기준(정표고)** 이다.
    반면 GNSS/드론 텔레메트리의 고도는 기종·설정에 따라 **WGS84 타원체고**
    인 경우가 흔하다. 한반도에서 지오이드고는 약 +25m 이므로, 이를 맞추지
    않으면 드론 고도에 25m 계통오차가 실린다. 부각이 낮은 광선에서는 이
    25m 가 지상에서 수백 m 로 증폭된다.

    `geoid_offset_m` 로 이 차이를 명시적으로 넣게 했다. 기본값 0.0 은
    "텔레메트리 고도가 이미 정표고(MSL)"라는 뜻이다. 타원체고를 쓴다면
    한국 기준 약 -25.0 을 넣어야 한다 (타원체고 - 지오이드고 = 정표고).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Copernicus DEM 30m — AWS Open Data, 인증 불필요.
COP_DEM_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    """정수 남서쪽 모서리 좌표 → GLO-30 타일 이름."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def tile_url(lat: int, lon: int) -> str:
    name = tile_name(lat, lon)
    return f"{COP_DEM_BASE}/{name}/{name}.tif"


def tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """bbox (min_lon, min_lat, max_lon, max_lat) 를 덮는 1°x1° 타일 목록."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lats = range(math.floor(min_lat), math.floor(max_lat) + 1)
    lons = range(math.floor(min_lon), math.floor(max_lon) + 1)
    return [(la, lo) for la in lats for lo in lons]


@dataclass
class DEM:
    """북향 정렬(north-up) 격자 표고 모델.

    Attributes:
        grid: (rows, cols) float32 표고 배열. 결측은 NaN.
        lon0, lat0: 좌상단 픽셀 **모서리**의 경위도.
        dlon, dlat: 픽셀 간격. dlat 는 음수(북→남 진행).
        geoid_offset_m: 조회 결과에 더할 보정. 모듈 docstring 참조.
    """

    grid: np.ndarray
    lon0: float
    lat0: float
    dlon: float
    dlat: float
    geoid_offset_m: float = 0.0

    # -------------------------------------------------------------- 적재

    @classmethod
    def from_files(
        cls,
        paths: list[Path | str],
        bbox: tuple[float, float, float, float] | None = None,
        geoid_offset_m: float = 0.0,
    ) -> DEM:
        """GeoTIFF 여러 장을 모자이크해 적재한다. bbox 를 주면 그 범위로 자른다."""
        import rasterio
        from rasterio.merge import merge

        srcs = [rasterio.open(str(p)) for p in paths]
        try:
            bounds = None
            if bbox is not None:
                # merge 의 bounds 는 (left, bottom, right, top).
                bounds = (bbox[0], bbox[1], bbox[2], bbox[3])
            mosaic, transform = merge(srcs, bounds=bounds)
            nodata = srcs[0].nodata
        finally:
            for s in srcs:
                s.close()

        band = mosaic[0].astype(np.float32)
        if nodata is not None:
            band[band == nodata] = np.nan
        # Copernicus 는 해양을 0 이 아니라 nodata 로 두지만, 방어적으로
        # 극단값도 결측 처리한다.
        band[band < -1000.0] = np.nan

        if transform.b != 0.0 or transform.d != 0.0:
            raise ValueError("회전된 격자는 지원하지 않는다 (north-up 만).")

        return cls(
            grid=band,
            lon0=transform.c,
            lat0=transform.f,
            dlon=transform.a,
            dlat=transform.e,
            geoid_offset_m=geoid_offset_m,
        )

    # -------------------------------------------------------------- 조회

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)."""
        rows, cols = self.grid.shape
        lon_a, lon_b = self.lon0, self.lon0 + cols * self.dlon
        lat_a, lat_b = self.lat0, self.lat0 + rows * self.dlat
        return (min(lon_a, lon_b), min(lat_a, lat_b), max(lon_a, lon_b), max(lat_a, lat_b))

    def elevation(self, lon, lat) -> np.ndarray:
        """경위도에서의 표고(m). 이중선형 보간, 벡터화.

        격자 밖이거나 결측이면 NaN. 형상은 입력을 따른다.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        rows, cols = self.grid.shape

        # 픽셀 모서리 기준 좌표 → 픽셀 **중심** 기준 좌표로 0.5 이동.
        col_f = (lon - self.lon0) / self.dlon - 0.5
        row_f = (lat - self.lat0) / self.dlat - 0.5

        c0 = np.floor(col_f).astype(np.int64)
        r0 = np.floor(row_f).astype(np.int64)
        tc = col_f - c0
        tr = row_f - r0

        inside = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < cols) & (r0 + 1 < rows)
        # 인덱싱 안전을 위해 클립한 뒤, 마지막에 inside 마스크로 덮어쓴다.
        c0c = np.clip(c0, 0, cols - 2)
        r0c = np.clip(r0, 0, rows - 2)

        g = self.grid
        v00 = g[r0c, c0c]
        v01 = g[r0c, c0c + 1]
        v10 = g[r0c + 1, c0c]
        v11 = g[r0c + 1, c0c + 1]

        top = v00 * (1.0 - tc) + v01 * tc
        bot = v10 * (1.0 - tc) + v11 * tc
        out = top * (1.0 - tr) + bot * tr

        out = np.where(inside, out, np.nan)
        return out + self.geoid_offset_m

    def sample_stats(self) -> dict:
        """적재 확인용 요약."""
        valid = np.isfinite(self.grid)
        return {
            "shape": tuple(self.grid.shape),
            "bounds": self.bounds,
            "valid_frac": float(valid.mean()),
            "min_m": float(np.nanmin(self.grid)) if valid.any() else float("nan"),
            "max_m": float(np.nanmax(self.grid)) if valid.any() else float("nan"),
            "geoid_offset_m": self.geoid_offset_m,
        }


def synthetic_dem(
    bbox: tuple[float, float, float, float],
    resolution_deg: float = 1.0 / 3600.0,
    seed: int = 0,
    relief_m: float = 400.0,
    base_m: float = 150.0,
) -> DEM:
    """테스트용 합성 산악 지형.

    실제 DEM 없이도 광선-지형 교차와 폐루프 검증이 돌아가야 한다
    (CI, 그리고 데이터 내려받기 전 개발 단계). 여러 주파수의 사인파를
    겹쳐 능선·계곡이 있는 지형을 만든다.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    cols = max(2, int(round((max_lon - min_lon) / resolution_deg)))
    rows = max(2, int(round((max_lat - min_lat) / resolution_deg)))

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=6)

    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    xn = xx / max(cols - 1, 1)
    yn = yy / max(rows - 1, 1)

    h = (
        0.50 * np.sin(2 * np.pi * (1.5 * xn) + phase[0]) * np.cos(2 * np.pi * (1.2 * yn) + phase[1])
        + 0.30 * np.sin(2 * np.pi * (3.1 * xn) + phase[2]) * np.cos(2 * np.pi * (2.7 * yn) + phase[3])
        + 0.20 * np.sin(2 * np.pi * (6.3 * xn) + phase[4]) * np.cos(2 * np.pi * (5.9 * yn) + phase[5])
    )
    grid = (base_m + relief_m * (h - h.min()) / max(h.max() - h.min(), 1e-9)).astype(np.float32)

    return DEM(
        grid=grid,
        lon0=min_lon,
        lat0=max_lat,          # 좌상단 = 최대 위도
        dlon=(max_lon - min_lon) / cols,
        dlat=-(max_lat - min_lat) / rows,
    )
