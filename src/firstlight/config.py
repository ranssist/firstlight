"""설정 파일 적재 — 현장(site)과 카메라 제원.

DEM 취득/캐시도 여기서 다룬다. 사이트 설정이 bbox 와 캐시 경로를 모두
알고 있으므로, "이 사이트의 DEM 을 준비해라"가 한 줄이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from firstlight.geo.camera import CameraIntrinsics
from firstlight.geo.dem import DEM, tile_url, tiles_for_bbox

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class SiteConfig:
    name: str
    label: str
    lat: float
    lon: float
    bbox: tuple[float, float, float, float]
    dem_cache_dir: Path
    geoid_offset_m: float = 0.0
    patrol_altitude_agl_m: float = 300.0
    response_units: list = field(default_factory=list)
    # 국토지리정보원 5m DEM 등, 수동으로 받아 둔 고해상도 DEM 디렉터리.
    # 설정되어 있으면 Copernicus 30m 보다 우선한다.
    local_dem_dir: Path | None = None

    @classmethod
    def load(cls, name_or_path: str | Path) -> SiteConfig:
        path = _resolve(name_or_path, CONFIG_ROOT / "sites")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        dem = raw.get("dem", {})
        return cls(
            name=raw["name"],
            label=raw.get("label", raw["name"]),
            lat=float(raw["lat"]),
            lon=float(raw["lon"]),
            bbox=tuple(float(v) for v in raw["bbox"]),
            dem_cache_dir=_project_path(dem.get("cache_dir", f"data/dem/{raw['name']}")),
            geoid_offset_m=float(dem.get("geoid_offset_m", 0.0)),
            patrol_altitude_agl_m=float(raw.get("patrol_altitude_agl_m", 300.0)),
            response_units=raw.get("response_units") or [],
            local_dem_dir=(
                _project_path(dem["local_dir"]) if dem.get("local_dir") else None
            ),
        )

    # ---------------------------------------------------------------- DEM

    def dem_tile_paths(self) -> list[Path]:
        """이 사이트를 덮는 타일들의 로컬 경로 (존재 여부와 무관)."""
        from firstlight.geo.dem import tile_name

        return [
            self.dem_cache_dir / f"{tile_name(la, lo)}.tif"
            for la, lo in tiles_for_bbox(self.bbox)
        ]

    def missing_dem_tiles(self) -> list[tuple[int, int]]:
        from firstlight.geo.dem import tile_name

        return [
            (la, lo)
            for la, lo in tiles_for_bbox(self.bbox)
            if not (self.dem_cache_dir / f"{tile_name(la, lo)}.tif").exists()
        ]

    def load_dem(self) -> DEM:
        """DEM 을 적재한다.

        `local_dem_dir` 이 설정돼 있으면 그쪽을 **먼저** 쓴다. 작품설명서가
        지정한 DEM 은 국토지리정보원 5m 급인데, 이는 회원가입 후 수동
        내려받기라 자동화할 수 없다. 받아 둔 GeoTIFF 를 그 디렉터리에 넣으면
        Copernicus 30m 대신 그것을 쓴다.

        해상도가 좌표 오차에 얼마나 기여하는지는 `firstlight geo-selftest` 를
        두 DEM 으로 각각 돌려 비교하면 된다 — 작품설명서가 "핵심 실험 과제"로
        꼽은 항목이다.
        """
        if self.local_dem_dir is not None:
            tiffs = sorted(
                p for p in self.local_dem_dir.glob("*")
                if p.suffix.lower() in {".tif", ".tiff", ".img"}
            )
            if tiffs:
                return DEM.from_files(
                    tiffs, bbox=self.bbox, geoid_offset_m=self.geoid_offset_m
                )
            raise FileNotFoundError(
                f"local_dem_dir 로 지정된 {self.local_dem_dir} 에 GeoTIFF 가 없다.\n"
                f"  국토지리정보원(map.ngii.go.kr)에서 5m DEM 을 받아 넣거나,\n"
                f"  설정에서 local_dem_dir 을 지워 Copernicus 30m 를 쓴다."
            )

        missing = self.missing_dem_tiles()
        if missing:
            urls = "\n  ".join(tile_url(la, lo) for la, lo in missing)
            raise FileNotFoundError(
                f"'{self.name}' 사이트의 DEM 타일 {len(missing)}개가 없다.\n"
                f"  firstlight fetch-dem --site {self.name}\n"
                f"필요한 타일:\n  {urls}"
            )
        return DEM.from_files(
            self.dem_tile_paths(), bbox=self.bbox, geoid_offset_m=self.geoid_offset_m
        )


@dataclass
class CameraConfig:
    name: str
    intrinsics: CameraIntrinsics

    @classmethod
    def load(cls, name_or_path: str | Path) -> CameraConfig:
        path = _resolve(name_or_path, CONFIG_ROOT / "cameras")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        dist = tuple(float(v) for v in raw.get("dist", (0, 0, 0, 0, 0)))

        if "hfov_deg" in raw:
            intr = CameraIntrinsics.from_fov(
                width=int(raw["width"]),
                height=int(raw["height"]),
                hfov_deg=float(raw["hfov_deg"]),
                vfov_deg=float(raw["vfov_deg"]) if "vfov_deg" in raw else None,
                dist=dist,
                name=raw["name"],
            )
        elif "focal_mm" in raw:
            intr = CameraIntrinsics.from_sensor(
                width=int(raw["width"]),
                height=int(raw["height"]),
                focal_mm=float(raw["focal_mm"]),
                sensor_width_mm=float(raw["sensor_width_mm"]),
                sensor_height_mm=(
                    float(raw["sensor_height_mm"]) if "sensor_height_mm" in raw else None
                ),
                dist=dist,
                name=raw["name"],
            )
        else:
            raise ValueError(f"{path}: hfov_deg 또는 focal_mm 중 하나는 있어야 한다")

        return cls(name=raw["name"], intrinsics=intr)


# --------------------------------------------------------------------------


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else CONFIG_ROOT.parent / path


def _resolve(name_or_path: str | Path, default_dir: Path) -> Path:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return path
    candidate = default_dir / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    available = sorted(p.stem for p in default_dir.glob("*.yaml"))
    raise FileNotFoundError(
        f"설정을 찾을 수 없다: {name_or_path}\n"
        f"  찾은 곳: {default_dir}\n"
        f"  사용 가능: {', '.join(available) if available else '(없음)'}"
    )
