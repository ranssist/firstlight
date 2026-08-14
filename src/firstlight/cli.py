"""firstlight 명령줄 도구."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from firstlight.config import CameraConfig, SiteConfig

app = typer.Typer(
    add_completion=False,
    help="드론·AI 기반 산불 조기탐지 파이프라인",
    no_args_is_help=True,
)
console = Console()


# --------------------------------------------------------------- fetch-dem


@app.command("fetch-dem")
def fetch_dem(
    site: str = typer.Option(..., "--site", "-s", help="configs/sites/ 의 사이트 이름"),
    force: bool = typer.Option(False, "--force", help="이미 있어도 다시 받는다"),
) -> None:
    """Copernicus GLO-30 DEM 타일을 내려받는다 (AWS 공개 데이터, 인증 불필요)."""
    import requests

    from firstlight.geo.dem import tile_name, tile_url, tiles_for_bbox

    cfg = SiteConfig.load(site)
    cfg.dem_cache_dir.mkdir(parents=True, exist_ok=True)
    tiles = tiles_for_bbox(cfg.bbox)

    console.print(f"[bold]{cfg.label}[/bold] — 타일 {len(tiles)}개, 저장 위치 {cfg.dem_cache_dir}")

    downloaded = skipped = failed = 0
    for lat, lon in tiles:
        dest = cfg.dem_cache_dir / f"{tile_name(lat, lon)}.tif"
        if dest.exists() and not force:
            console.print(f"  [dim]건너뜀[/dim] {dest.name}")
            skipped += 1
            continue

        url = tile_url(lat, lon)
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                if resp.status_code == 404:
                    # GLO-30 은 해양 전용 타일을 배포하지 않는다. 정상 상황이다.
                    console.print(f"  [yellow]없음[/yellow]  {tile_name(lat, lon)} (해양 추정)")
                    skipped += 1
                    continue
                resp.raise_for_status()
                tmp = dest.with_suffix(".part")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
            size_mb = dest.stat().st_size / 1e6
            console.print(f"  [green]받음[/green]  {dest.name} ({size_mb:.1f} MB)")
            downloaded += 1
        except Exception as exc:  # noqa: BLE001 — 어떤 실패든 사용자에게 보여준다
            console.print(f"  [red]실패[/red]  {tile_name(lat, lon)}: {exc}")
            failed += 1

    console.print(f"\n받음 {downloaded} / 건너뜀 {skipped} / 실패 {failed}")
    if failed:
        raise typer.Exit(1)

    try:
        dem = cfg.load_dem()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    stats = dem.sample_stats()
    console.print(
        f"\n[bold]적재 확인[/bold]  격자 {stats['shape'][0]}x{stats['shape'][1]}, "
        f"유효 {stats['valid_frac'] * 100:.1f}%, "
        f"표고 {stats['min_m']:.0f}~{stats['max_m']:.0f} m"
    )
    console.print(
        f"사이트 지점 표고: {float(dem.elevation(cfg.lon, cfg.lat)):.1f} m"
    )


# ------------------------------------------------------------ geo-selftest


@app.command("geo-selftest")
def geo_selftest(
    site: str = typer.Option("uiseong", "--site", "-s"),
    camera: str = typer.Option("generic_wide", "--camera", "-c"),
    trials: int = typer.Option(200, "--trials", "-n", help="몬테카를로 시행 수"),
    altitude: float = typer.Option(None, "--altitude", "-a", help="지표 기준 고도 m"),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="실제 DEM 대신 합성 지형 사용 (다운로드 불필요)"
    ),
    out: Path = typer.Option(None, "--out", "-o", help="결과 JSON 저장 경로"),
) -> None:
    """폐루프 왕복 검증 + 부각별 CEP 스윕.

    소개서가 주장하는 "오차 50m 이내"의 근거를 만든다.
    """
    from firstlight.evaluation.geo_accuracy import run_report
    from firstlight.geo.dem import synthetic_dem

    site_cfg = SiteConfig.load(site)
    cam_cfg = CameraConfig.load(camera)
    alt_agl = altitude if altitude is not None else site_cfg.patrol_altitude_agl_m

    if synthetic:
        console.print("[yellow]합성 지형 사용[/yellow] — 실제 지형 통계가 아니다")
        dem = synthetic_dem(site_cfg.bbox, resolution_deg=1 / 1200, seed=7)
    else:
        try:
            dem = site_cfg.load_dem()
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            console.print("\n[dim]다운로드 없이 돌려보려면 --synthetic 을 쓴다.[/dim]")
            raise typer.Exit(1) from exc

    console.print(
        f"\n[bold]{site_cfg.label}[/bold] · 카메라 {cam_cfg.name} "
        f"({cam_cfg.intrinsics.width}x{cam_cfg.intrinsics.height}, "
        f"화각 {cam_cfg.intrinsics.hfov_deg:.1f}도) · 고도 {alt_agl:.0f} m AGL "
        f"· 시행 {trials}회"
    )

    report = run_report(
        dem,
        cam_cfg.intrinsics,
        site_cfg.lat,
        site_cfg.lon,
        altitude_agl_m=alt_agl,
        trials=trials,
    )

    _print_roundtrip(report.roundtrip)
    for label, rows in report.sweeps.items():
        _print_sweep(label, rows, report.max_cep90_m)
    _print_verdict(report)

    if out is not None:
        _write_json(report, out, site_cfg, cam_cfg, alt_agl)
        console.print(f"\n[dim]JSON 저장: {out}[/dim]")


def _print_roundtrip(rt) -> None:
    console.print("\n[bold]1. 폐루프 왕복 정확도[/bold] [dim](노이즈 없음 — 순수 구현 오차)[/dim]")
    if rt.n_samples == 0:
        console.print("  [red]표본 없음[/red]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("표본")
    table.add_column("중앙값", justify="right")
    table.add_column("95퍼센타일", justify="right")
    table.add_column("최대", justify="right")
    table.add_column("픽셀 잔차", justify="right")
    table.add_row(
        str(rt.n_samples),
        f"{rt.median_m:.3f} m",
        f"{rt.p95_m:.3f} m",
        f"{rt.max_m:.3f} m",
        f"{rt.pixel_residual_px:.4f} px",
    )
    console.print(table)


def _print_sweep(label: str, rows, max_cep90_m: float) -> None:
    console.print(f"\n[bold]2. 부각별 CEP — {label}[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("부각", justify="right")
    table.add_column("사거리", justify="right")
    table.add_column("CEP50", justify="right")
    table.add_column("CEP90", justify="right")
    table.add_column("발행률", justify="right")
    table.add_column("판정")

    for row in rows:
        if row.n_solved == 0:
            table.add_row(f"{row.depression_deg:.0f}°", "—", "—", "—", "0%",
                          "[dim]교차 실패[/dim]")
            continue
        ok = row.cep90_m <= max_cep90_m
        verdict = "[green]발행[/green]" if ok else "[red]거절[/red]"
        table.add_row(
            f"{row.depression_deg:.0f}°",
            f"{row.median_range_m:,.0f} m",
            f"{row.cep50_m:.0f} m",
            f"{row.cep90_m:.0f} m",
            f"{row.publish_rate * 100:.0f}%",
            verdict,
        )
    console.print(table)


def _print_verdict(report) -> None:
    """소개서 §5 의 '50m 이내' 주장이 어느 조건에서 성립하는지."""
    console.print("\n[bold]3. 판정[/bold]")

    rt = report.roundtrip
    if rt.n_samples and rt.max_m < 1.0:
        console.print(f"  [green]OK[/green]  왕복 최대오차 {rt.max_m:.3f} m < 1 m")
    else:
        console.print(f"  [red]NG[/red]  왕복 최대오차 {rt.max_m:.3f} m — 구현 오차가 있다")

    for label, rows in report.sweeps.items():
        passing = [r for r in rows if r.n_solved and r.cep50_m <= 50.0]
        if passing:
            best = min(r.depression_deg for r in passing)
            console.print(
                f"  [green]OK[/green]  {label}: 부각 {best:.0f}° 이상에서 CEP50 ≤ 50 m"
            )
        else:
            console.print(f"  [yellow]--[/yellow]  {label}: 어느 부각에서도 CEP50 ≤ 50 m 미달")


def _write_json(report, out: Path, site_cfg, cam_cfg, alt_agl: float) -> None:
    import json
    from dataclasses import asdict

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "site": site_cfg.name,
        "camera": cam_cfg.name,
        "altitude_agl_m": alt_agl,
        "max_cep90_m": report.max_cep90_m,
        "roundtrip": asdict(report.roundtrip),
        "sweeps": {k: [asdict(r) for r in v] for k, v in report.sweeps.items()},
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------- bench-detect


@app.command("bench-detect")
def bench_detect(
    devices: str = typer.Option("cpu,gpu,npu", "--devices", help="쉼표 구분"),
    images: int = typer.Option(12, "--images", "-n"),
    imgsz: int = typer.Option(1024, "--imgsz"),
    out: Path = typer.Option(None, "--out", "-o"),
) -> None:
    """장치별 추론 속도를 잰다 (CPU / Arc iGPU / NPU).

    로컬에 NVIDIA GPU 가 없어도 실시간이 나오는지가 이 프로젝트의
    현실성 판단 기준이다.
    """
    import time

    import cv2
    import numpy as np

    from firstlight.detect import OnnxDetector

    paths = sorted(Path("data/pyro_sdis/val/images").glob("*.jpg"))[:images]
    if not paths:
        console.print("[red]검증 이미지가 없다.[/red]")
        console.print("  uv run python scripts/fetch_pyro_sdis.py --split val --count 200")
        raise typer.Exit(1)

    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    console.print(f"이미지 {len(frames)}장 · {frames[0].shape[1]}x{frames[0].shape[0]} · imgsz={imgsz}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("장치")
    table.add_column("중앙값", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("최소~최대", justify="right")
    table.add_column("가속비", justify="right")

    baseline_ms = None
    results = []
    for device in [d.strip() for d in devices.split(",") if d.strip()]:
        try:
            det = OnnxDetector(device=device, imgsz=imgsz)
            for frame in frames[:3]:
                det.detect(frame)              # 워밍업 (컴파일·캐시)
            samples = []
            for frame in frames:
                start = time.perf_counter()
                det.detect(frame)
                samples.append((time.perf_counter() - start) * 1000.0)
            arr = np.array(samples)
            median = float(np.median(arr))
            if baseline_ms is None:
                baseline_ms = median
            results.append(
                {
                    "device": device,
                    "median_ms": median,
                    "fps": 1000.0 / median,
                    "min_ms": float(arr.min()),
                    "max_ms": float(arr.max()),
                    "speedup": baseline_ms / median,
                }
            )
            table.add_row(
                device.upper(),
                f"{median:.1f} ms",
                f"{1000 / median:.1f}",
                f"{arr.min():.0f}~{arr.max():.0f} ms",
                f"{baseline_ms / median:.1f}x",
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row(device.upper(), "[red]실패[/red]", "—", str(exc)[:40], "—")

    console.print(table)

    if out is not None:
        import json

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"imgsz": imgsz, "n_images": len(frames), "devices": results},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        console.print(f"\n[dim]JSON 저장: {out}[/dim]")


# --------------------------------------------------------- detect-baseline


@app.command("detect-baseline")
def detect_baseline(
    device: str = typer.Option("gpu", "--device", "-d"),
    limit: int = typer.Option(None, "--limit", "-n", help="이미지 수 제한"),
    tiled: bool = typer.Option(False, "--tiled", help="슬라이스 추론 사용"),
    conf: float = typer.Option(0.05, "--conf", help="PR 곡선용 낮은 임계값"),
    out: Path = typer.Option(None, "--out", "-o"),
) -> None:
    """pyro-sdis 검증셋 PR 곡선 / AP50 / 프레임 오탐률.

    여기서 나온 오탐률이 M3 시퀀스 검증기가 줄여야 할 출발점이다.
    """
    from firstlight.detect import OnnxDetector, TiledDetector
    from firstlight.evaluation.detect_baseline import run_baseline, save_json

    manifest = Path("data/pyro_sdis/val_manifest.json")
    if not manifest.exists():
        console.print(f"[red]매니페스트가 없다: {manifest}[/red]")
        console.print("  uv run python scripts/fetch_pyro_sdis.py --split val --count 200")
        raise typer.Exit(1)

    detector = OnnxDetector(device=device, conf_threshold=conf)
    if tiled:
        detector = TiledDetector(detector)
    console.print(f"{detector}\n")

    with console.status("추론 중...") as status:
        def progress(done: int, total: int) -> None:
            status.update(f"추론 중... {done}/{total}")

        result = run_baseline(detector, manifest, limit=limit, device=device,
                              progress=progress)

    console.print(
        f"[bold]이미지[/bold] {result.n_images}장 "
        f"(연기 있음 {result.n_images_with_smoke} / 없음 {result.n_images_without_smoke}) · "
        f"정답 박스 {result.n_ground_truth}개 · 예측 {result.n_predictions}개"
    )
    console.print(
        f"[bold]AP50[/bold] {result.ap50:.3f} · "
        f"중앙 지연 {result.median_latency_ms:.0f} ms ({device.upper()})"
    )

    table = Table(show_header=True, header_style="bold", title="신뢰도 임계값별")
    table.add_column("임계값", justify="right")
    table.add_column("정밀도", justify="right")
    table.add_column("재현율", justify="right")
    table.add_column("F1", justify="right")
    for point in result.curve:
        mark = "  [bold]<- 최적[/bold]" if point is result.best else ""
        table.add_row(
            f"{point.confidence:.2f}",
            f"{point.precision:.3f}",
            f"{point.recall:.3f}",
            f"{point.f1:.3f}{mark}",
        )
    console.print(table)

    console.print("\n[bold]프레임 단위 오탐[/bold] [dim](M3 시퀀스 검증기의 공격 대상)[/dim]")
    console.print(
        f"  연기 없는 이미지 1장당 오탐 {result.false_positives_per_negative_image:.2f}건"
    )
    console.print(
        f"  연기 없는 이미지 중 경보가 뜬 비율 "
        f"{result.negative_image_alarm_rate * 100:.1f}%"
    )

    if out is not None:
        save_json(result, out)
        console.print(f"\n[dim]JSON 저장: {out}[/dim]")


# -------------------------------------------------------------- falsealarm


@app.command("falsealarm")
def falsealarm(
    device: str = typer.Option("gpu", "--device", "-d"),
    conf: float = typer.Option(0.20, "--conf", help="운용 신뢰도 임계값"),
    limit: int = typer.Option(None, "--limit", "-n", help="시퀀스 수 제한"),
    scorer_path: Path = typer.Option(None, "--scorer", help="학습된 스코어러 JSON"),
    train: bool = typer.Option(
        False, "--train", help="측정 후 라벨로 스코어러를 학습하고 재측정한다"
    ),
    save_scorer: Path = typer.Option(
        Path("models/scorer_sparse.json"), "--save-scorer"
    ),
    out: Path = typer.Option(None, "--out", "-o"),
) -> None:
    """FIgLib 발화 전 구간으로 오경보율을 잰다 (소개서 §4① 근거).

    세 정책을 비교한다: raw(프레임 경보) / dedup(트랙 경보) / verified(시퀀스 검증).
    """
    from firstlight.detect import OnnxDetector
    from firstlight.evaluation.falsealarm import (  # noqa: F401 — run_false_alarm 은 홀드아웃에 쓴다
        build_training_set,
        run_false_alarm,
        save_json,
    )
    from firstlight.verify.features import VerifierMode
    from firstlight.verify.scorer import SequenceScorer

    manifest = Path("data/figlib/manifest.json")
    if not manifest.exists():
        console.print(f"[red]FIgLib 매니페스트가 없다: {manifest}[/red]")
        console.print("  uv run python scripts/fetch_figlib.py --sequences 16")
        raise typer.Exit(1)

    detector = OnnxDetector(device=device, conf_threshold=conf)
    scorer = (
        SequenceScorer.load(scorer_path)
        if scorer_path
        else SequenceScorer(mode=VerifierMode.SPARSE)
    )
    console.print(f"{detector}\n{scorer}\n")

    def run(active_scorer):
        with console.status("시퀀스 처리 중...") as status:
            def progress(done, total, seq):
                status.update(f"시퀀스 {done}/{total} — {seq}")

            return run_false_alarm(
                detector, manifest, scorer=active_scorer, limit=limit, progress=progress
            )

    result, labelled = run(scorer)
    _print_falsealarm(
        result,
        "사전값 스코어러 (전 시퀀스)" if not scorer.is_fitted else "학습된 스코어러",
        note="학습을 하지 않았으므로 데이터 누수가 없다. 가장 방어하기 쉬운 수치다.",
    )

    if train:
        # 학습과 평가에 같은 시퀀스를 쓰면 누수다. 시퀀스 단위로 절반씩 나눈다.
        # 프레임 단위로 나누면 같은 안개가 양쪽에 들어가므로 의미가 없다.
        all_seqs = sorted({seq for seq, _, _ in labelled})
        train_seqs = set(all_seqs[::2])
        test_seqs = set(all_seqs[1::2])

        features, labels = build_training_set(
            labelled, mode=VerifierMode.SPARSE, sequences=train_seqs
        )
        n_pos = sum(labels)
        console.print(
            f"\n[bold]스코어러 학습[/bold] — 학습 시퀀스 {len(train_seqs)}개 / "
            f"평가 시퀀스 {len(test_seqs)}개 (겹치지 않음)"
        )
        console.print(
            f"  학습 표본 {len(labels)}개 (양성 {n_pos} / 음성 {len(labels) - n_pos})"
        )
        # 표본이 적으면 학습을 **거부한다**. 음성 1개로 계수 9개를 맞추면
        # 로지스틱 회귀는 조용히 퇴화한 해를 낸다 (계수가 -0.00 으로 죽는다).
        # 그렇게 나온 가중치로 측정한 오경보율은 아무 의미가 없다.
        n_neg = len(labels) - n_pos
        min_per_class = 8
        if n_pos < min_per_class or n_neg < min_per_class:
            console.print(
                f"[yellow]학습 생략[/yellow] — 클래스당 최소 {min_per_class}개가 필요하다 "
                f"(양성 {n_pos} / 음성 {n_neg}).\n"
                f"  [dim]60초 간격에서는 오탐 대부분이 다중 프레임 트랙을 이루지 못해\n"
                f"  음성 표본이 잘 모이지 않는다. 시퀀스를 더 받아야 한다:\n"
                f"    uv run python scripts/fetch_figlib.py --sequences 64 --skip 16[/dim]"
            )
        else:
            fitted = SequenceScorer(mode=VerifierMode.SPARSE).fit(features, labels)
            fitted.save(save_scorer)
            console.print(f"  저장: {save_scorer}")

            table = Table(show_header=True, header_style="bold", title="학습된 계수")
            table.add_column("특징")
            table.add_column("사전값", justify="right")
            table.add_column("학습값", justify="right")
            for name in fitted.names:
                table.add_row(
                    name,
                    f"{scorer.weights.get(name, 0.0):+.2f}",
                    f"{fitted.weights[name]:+.2f}",
                )
            console.print(table)

            # 홀드아웃 시퀀스에서만 평가한다.
            with console.status("홀드아웃 평가 중...") as status:
                def progress(done, total, seq):
                    status.update(f"홀드아웃 {done}/{total} — {seq}")

                holdout, _ = run_false_alarm(
                    detector, manifest, scorer=fitted, limit=limit,
                    only=test_seqs, progress=progress,
                )
            _print_falsealarm(
                holdout,
                "학습된 스코어러 (홀드아웃 시퀀스만)",
                note="학습에 쓰지 않은 시퀀스에서만 측정했다.",
            )

            # 같은 홀드아웃 구간에서 사전값과 직접 비교해야 공정하다.
            with console.status("홀드아웃 기준선 측정 중..."):
                baseline, _ = run_false_alarm(
                    detector, manifest,
                    scorer=SequenceScorer(mode=VerifierMode.SPARSE),
                    limit=limit, only=test_seqs,
                )
            console.print(
                f"\n[bold]같은 홀드아웃 구간 비교[/bold] — "
                f"사전값 {baseline.verified_per_hour:.2f} → "
                f"학습 후 {holdout.verified_per_hour:.2f} 건/시간"
            )
            result = holdout

    if out is not None:
        save_json(result, out)
        console.print(f"\n[dim]JSON 저장: {out}[/dim]")


def _print_falsealarm(result, title: str, note: str | None = None) -> None:
    console.print(
        f"\n[bold]{title}[/bold] — 시퀀스 {result.n_sequences}개 · "
        f"발화 전 관측 [bold]{result.negative_hours:.2f}시간[/bold]"
    )
    if note:
        console.print(f"  [dim]{note}[/dim]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("경보 정책")
    table.add_column("설명")
    table.add_column("건/시간", justify="right")
    table.add_column("목표 대비")

    target = 0.5      # 소개서 §4① 목표: 0.5건/비행시간 이하
    for label, desc, value in (
        ("raw", "탐지 1건 = 경보 1건", result.raw_per_hour),
        ("dedup", "트랙 1개 = 경보 1건", result.dedup_per_hour),
        ("verified", "시퀀스 검증 통과만", result.verified_per_hour),
    ):
        verdict = (
            "[green]달성[/green]" if value <= target else f"[red]{value / target:.1f}배 초과[/red]"
        )
        table.add_row(label, desc, f"{value:.2f}", verdict)
    console.print(table)

    console.print(
        f"  검증에 의한 감소: raw 대비 [bold]{result.reduction_vs_raw * 100:.1f}%[/bold], "
        f"dedup 대비 [bold]{result.reduction_vs_dedup * 100:.1f}%[/bold]"
    )

    # T2 — 작품설명서 표 Ⅲ-2 는 오탐률을 %로 적었다 (목표 3% 이하).
    fpr = Table(show_header=True, header_style="bold",
                title="T2 프레임 단위 오탐률 (작품설명서 표 Ⅲ-2)")
    fpr.add_column("판정 방식")
    fpr.add_column("오탐률", justify="right")
    fpr.add_column("예상치")
    fpr.add_column("판정")
    fpr.add_row(
        "단일 프레임", f"{result.fpr_single_frame * 100:.1f}%", "12~18%",
        "[dim]기준선[/dim]",
    )
    seq_ok = result.fpr_sequence <= 0.03
    fpr.add_row(
        "시퀀스 검증", f"{result.fpr_sequence * 100:.1f}%", "3% 이하",
        "[green]달성[/green]" if seq_ok else "[red]미달[/red]",
    )
    console.print(fpr)
    console.print(
        f"  오탐 저감률 [bold]{result.fpr_reduction * 100:.1f}%[/bold] "
        f"(연기 없는 프레임 {result.n_negative_frames:,}장 기준)"
    )

    # T4 — 목표 30초 이내.
    if result.latency_median_ms == result.latency_median_ms:      # not NaN
        within = result.latency_p95_ms <= 30_000
        console.print(
            f"\n[bold]T4 End-to-End 지연[/bold] (추론→검증→경보 확정) — "
            f"중앙값 [bold]{result.latency_median_ms:.0f}ms[/bold] · "
            f"p95 {result.latency_p95_ms:.0f}ms · 최대 {result.latency_max_ms:.0f}ms"
        )
        console.print(
            f"  목표 30초 이내: "
            + ("[green]달성[/green]" if within else "[red]미달[/red]")
            + "  [dim](영상 디코딩·네트워크 전송은 제외한 파이프라인 구간)[/dim]"
        )

    console.print(
        f"\n  발화 후 탐지율: {result.detection_rate * 100:.0f}% "
        f"({result.n_sequences}개 시퀀스 중)"
    )
    if result.median_time_to_flare_s is not None:
        console.print(
            f"  발화→FLARE 중앙값: {result.median_time_to_flare_s / 60:.0f}분 "
            f"[dim](60초 간격 데이터라 이보다 세밀하게는 측정 불가)[/dim]"
        )


# ---------------------------------------------------------------------- run


@app.command("run")
def run_pipeline(
    source: str = typer.Option(..., "--source", help="영상 파일 또는 FIgLib 시퀀스 디렉터리"),
    site: str = typer.Option("uiseong", "--site", "-s"),
    camera: str = typer.Option("generic_wide", "--camera", "-c"),
    device: str = typer.Option("gpu", "--device", "-d"),
    conf: float = typer.Option(0.20, "--conf"),
    telemetry: str = typer.Option(
        "synthetic", "--telemetry", help="synthetic | <CSV 경로>"
    ),
    altitude: float = typer.Option(None, "--altitude", help="지표 기준 고도 m"),
    pitch: float = typer.Option(-35.0, "--pitch", help="짐벌 부각 (음수가 하방)"),
    yaw: float = typer.Option(20.0, "--yaw"),
    interval: float = typer.Option(None, "--interval", help="프레임 간격 초 (미지정시 자동)"),
    scorer_path: Path = typer.Option(None, "--scorer"),
    db: Path = typer.Option(Path("data/events.db"), "--db"),
    reset: bool = typer.Option(False, "--reset", help="시작 전 DB 를 비운다"),
    max_frames: int = typer.Option(None, "--max-frames", "-n"),
) -> None:
    """영상을 흘려 탐지→검증→좌표→경보 전 과정을 돌린다.

    텔레메트리가 없는 영상(공개 데이터셋 전부)에는 --telemetry synthetic 으로
    고정 자세를 가정한다. 이때 나오는 좌표는 **기하 계산이 동작함을 보이는
    것**이지 실제 발화 위치가 아니다.
    """
    from firstlight.events.store import EventStore
    from firstlight.geo.pose import CameraPose
    from firstlight.pipeline import build_pipeline

    try:
        reader = _open_source(source, max_frames)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    # 이미지 시퀀스는 파일명의 시각을, 동영상은 fps 를 쓴다.
    step = interval if interval is not None else reader.interval_s
    site_cfg = SiteConfig.load(site)

    if reset and db.exists():
        EventStore(db).clear()

    pipeline, intrinsics = build_pipeline(
        site_name=site,
        camera_name=camera,
        device=device,
        conf=conf,
        frame_interval_s=step,
        db_path=str(db),
        scorer_path=str(scorer_path) if scorer_path else None,
        use_egomotion=(telemetry != "synthetic"),
        synthetic_dem_fallback=True,
    )

    alt_agl = altitude if altitude is not None else site_cfg.patrol_altitude_agl_m
    terrain = float(pipeline.geo_solver.dem.elevation(site_cfg.lon, site_cfg.lat))
    pose = CameraPose(
        lat=site_cfg.lat, lon=site_cfg.lon, alt_msl=terrain + alt_agl,
        yaw_deg=yaw, pitch_deg=pitch,
    )

    console.print(
        f"[bold]{site_cfg.label}[/bold] · {reader.describe()} · 간격 {step:g}초 "
        f"({pipeline.mode.value} 모드) · {pipeline.scorer}"
    )
    console.print(
        f"  가정 자세: 고도 {alt_agl:.0f}m AGL, 방위 {yaw:.0f}°, 부각 {-pitch:.0f}°"
    )

    counts = {"FLARE": 0, "GLOW": 0, "SPARK": 0}
    notified = 0
    processed = 0
    with console.status("처리 중...") as status:
        for i, (image, timestamp) in enumerate(reader):
            result = pipeline.process_frame(image, timestamp, i, pose=pose)
            processed += 1
            for verdict in result.verdicts:
                counts[verdict.tier.value] += 1
                if verdict.notification is not None:
                    notified += 1
                    console.print(
                        pipeline.router.format_alert(verdict.notification)
                    )
            status.update(f"프레임 {i + 1}/{reader.total or '?'} · 통지 {notified}건")

    pipeline.close()
    console.print(
        f"\n[bold]완료[/bold] — 판정 {sum(counts.values())}회 "
        f"(FLARE {counts['FLARE']} / GLOW {counts['GLOW']} / SPARK {counts['SPARK']}), "
        f"실제 통지 [bold]{notified}[/bold]건"
    )
    console.print(f"  [dim]억제 효과: 판정 {sum(counts.values())}회 → 통지 {notified}건[/dim]")
    console.print(f"\n대시보드에서 확인:\n  uv run firstlight serve --db {db}")


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".ts"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class _FrameReader:
    """이미지 시퀀스와 동영상을 같은 인터페이스로 흘려보낸다.

    (프레임 BGR, 타임스탬프 초) 를 순서대로 내놓는다. 타임스탬프는
    시간축 특징의 단위가 되므로 실제 초여야 한다 — 프레임 번호가 아니다.
    """

    def __init__(self, interval_s: float, total: int | None, label: str) -> None:
        self.interval_s = interval_s
        self.total = total
        self.label = label

    def describe(self) -> str:
        return f"{self.label} {self.total or '?'}프레임"


class _ImageSequenceReader(_FrameReader):
    def __init__(self, frames: list[tuple[Path, float]], interval_s: float) -> None:
        super().__init__(interval_s, len(frames), "이미지 시퀀스")
        self.frames = frames

    def __iter__(self):
        import cv2

        for path, timestamp in self.frames:
            image = cv2.imread(str(path))
            if image is not None:
                yield image, timestamp


class _VideoReader(_FrameReader):
    def __init__(self, path: Path, max_frames: int | None) -> None:
        import cv2

        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"동영상을 열 수 없다: {path}")
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_frames:
            count = min(count, max_frames) if count else max_frames
        super().__init__(1.0 / fps if fps > 0 else 1.0, count or None,
                         f"동영상({fps:.0f}fps)")
        self.max_frames = max_frames

    def __iter__(self):
        index = 0
        while True:
            if self.max_frames and index >= self.max_frames:
                break
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame, index * self.interval_s
            index += 1
        self.capture.release()


def _open_source(source: str, max_frames: int | None = None) -> _FrameReader:
    """영상 파일 또는 이미지 디렉터리를 프레임 리더로 연다."""
    import re

    path = Path(source)

    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return _VideoReader(path, max_frames)

    if path.is_dir():
        # FIgLib 처럼 파일명에 발화 기준 오프셋(초)이 박혀 있으면 그걸 쓴다.
        offset = re.compile(r"_([+-]?\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
        found: list[tuple[Path, float]] = []
        for image in path.iterdir():
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            match = offset.search(image.name)
            found.append((image, float(match.group(1)) if match else 0.0))
        if not found:
            raise FileNotFoundError(f"이미지가 없다: {path}")

        if all(t == 0.0 for _, t in found):
            # 시각 정보가 없으면 1초 간격으로 가정한다.
            found = [(p, float(i)) for i, (p, _) in enumerate(sorted(found))]
        found.sort(key=lambda t: t[1])
        if max_frames:
            found = found[:max_frames]

        return _ImageSequenceReader(found, _estimate_cadence([t for _, t in found]))

    raise FileNotFoundError(f"영상 파일도 이미지 디렉터리도 아니다: {source}")


def _estimate_cadence(timestamps: list[float]) -> float:
    """프레임 간격을 추정한다.

    단순 중앙값을 쓰면 안 된다. FIgLib 시퀀스는 발화 시점을 전후로 큰 공백이
    있고, 표본이 적으면 그 공백이 중앙값을 통째로 끌어올린다 (실제로
    3프레임 테스트에서 60초가 2280초로 나왔다).

    프레임은 촬영 간격보다 촘촘할 수 없으므로 **가장 작은 간격 주변의
    군집**이 곧 촬영 간격이다. 최소값의 2배 이내만 남기고 중앙을 취하면
    공백에도, 동영상의 타임스탬프 지터에도 견딘다.
    """
    import numpy as np

    diffs = np.diff(np.asarray(timestamps, dtype=float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    base = float(diffs.min())
    cluster = diffs[diffs <= 2.0 * base]
    return float(np.median(cluster)) if cluster.size else base


# -------------------------------------------------------------------- serve


@app.command("serve")
def serve(
    db: Path = typer.Option(Path("data/events.db"), "--db"),
    scorer_path: Path = typer.Option(Path("models/scorer_sparse.json"), "--scorer"),
    site: str = typer.Option("uiseong", "--site", "-s", help="지도 중심·진화대 위치"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """관제 대시보드를 띄운다."""
    import uvicorn

    from firstlight.api.main import create_app

    console.print(f"[bold]FIRSTLIGHT 관제[/bold] → http://{host}:{port}")
    console.print(f"  DB: {db}  ·  스코어러: {scorer_path}  ·  현장: {site}")
    uvicorn.run(
        create_app(str(db), scorer_path, site_name=site),
        host=host, port=port, log_level="warning",
    )


# ------------------------------------------------------------------- report


@app.command("report")
def report(
    out_dir: Path = typer.Option(Path("out"), "--out-dir"),
    dest: Path = typer.Option(Path("out/RESULTS.md"), "--dest"),
) -> None:
    """측정 결과를 소개서 §5 형식의 문서로 묶는다."""
    from firstlight.evaluation.report import build_claims, write_report

    claims = build_claims(out_dir)
    if not claims:
        console.print(f"[red]{out_dir} 에 측정 결과 JSON 이 없다.[/red]")
        console.print("  먼저 geo-selftest / detect-baseline / falsealarm 을 --out 과 함께 실행한다.")
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold")
    table.add_column("항목")
    table.add_column("목표")
    table.add_column("실측")
    table.add_column("판정")
    colours = {"달성": "green", "조건부": "yellow", "미측정": "red", "참고": "dim"}
    for claim in claims:
        table.add_row(
            claim.item,
            claim.proposal_target,
            claim.measured,
            f"[{colours.get(claim.verdict, 'white')}]{claim.verdict}[/]",
        )
    console.print(table)

    path = write_report(out_dir, dest)
    console.print(f"\n[dim]문서 저장: {path}[/dim]")


# ------------------------------------------------------------------ 기타


@app.command("sites")
def list_sites() -> None:
    """등록된 사이트와 DEM 준비 상태."""
    from firstlight.config import CONFIG_ROOT

    table = Table(show_header=True, header_style="bold")
    table.add_column("이름")
    table.add_column("설명")
    table.add_column("중심")
    table.add_column("DEM")

    for path in sorted((CONFIG_ROOT / "sites").glob("*.yaml")):
        cfg = SiteConfig.load(path)
        missing = cfg.missing_dem_tiles()
        status = "[green]준비됨[/green]" if not missing else f"[yellow]{len(missing)}개 없음[/yellow]"
        table.add_row(cfg.name, cfg.label, f"{cfg.lat:.4f}, {cfg.lon:.4f}", status)

    console.print(table)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
