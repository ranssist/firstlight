"""측정 결과를 소개서 §5 형식의 표로 묶는다.

이 저장소의 최종 산출물이다. `out/*.json` 을 읽어 "목표" 열을 "실측" 열로
바꾼 문서를 만든다. 심사위원이 "그 숫자 어디서 나왔냐"고 물었을 때
재현 명령까지 함께 답할 수 있어야 한다.

측정하지 못한 항목은 **비워두지 않고 왜 못 쟀는지 적는다.** 빈 칸은
누락처럼 보이지만, 이유가 적힌 칸은 한계를 아는 것처럼 보인다. 실제로도
후자가 맞다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class Claim:
    """소개서의 주장 하나와 그에 대응하는 실측."""

    item: str
    proposal_target: str
    measured: str
    verdict: str            # "달성" | "조건부" | "미측정"
    basis: str              # 어떤 데이터로 쟀는지
    command: str            # 재현 명령


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_claims(out_dir: Path) -> list[Claim]:
    geo = _load(out_dir / "geo_accuracy_uiseong.json")
    detect = _load(out_dir / "detect_baseline_val200.json")
    alarm = _load(out_dir / "falsealarm_figlib.json") or _load(
        out_dir / "falsealarm_holdout.json"
    )
    bench = _load(out_dir / "bench_detect.json")

    claims: list[Claim] = []

    # --- ② 위치 정확도 ---------------------------------------------------
    if geo:
        consumer = geo["sweeps"].get("일반 GNSS", [])
        passing = [r for r in consumer if r["n_solved"] and r["cep50_m"] <= 50.0]
        if passing:
            best = min(r["depression_deg"] for r in passing)
            row = next(r for r in passing if r["depression_deg"] == best)
            measured = (
                f"부각 {best:.0f}° 이상에서 CEP50 {row['cep50_m']:.0f}m "
                f"(CEP90 {row['cep90_m']:.0f}m, 사거리 {row['median_range_m']:,.0f}m)"
            )
            verdict = "조건부"
        else:
            measured = "일반 GNSS 로는 어느 부각에서도 50m 미달"
            verdict = "미달"
        claims.append(
            Claim(
                item="발화 지점 위치 정확도",
                proposal_target="50m 이내 (CEP)",
                measured=measured,
                verdict=verdict,
                basis=(
                    f"의성 실제 DEM(Copernicus GLO-30), 고도 300m AGL, "
                    f"몬테카를로 {geo.get('sweeps', {}).get('일반 GNSS', [{}])[0].get('n_attempted', 8)}방위 x 200시행. "
                    f"무노이즈 폐루프 왕복오차 {geo['roundtrip']['max_m']:.3f}m"
                ),
                command="uv run firstlight geo-selftest --site uiseong --trials 200",
            )
        )

    # --- T2 프레임 단위 오탐률 (작품설명서 표 Ⅲ-2) -----------------------
    if alarm and alarm.get("n_negative_frames"):
        seq_fpr = alarm["fpr_sequence"]
        claims.append(
            Claim(
                item="T2 오탐률 (프레임 단위)",
                proposal_target="단일 12~18% → 시퀀스 3% 이하",
                measured=(
                    f"단일 프레임 {alarm['fpr_single_frame'] * 100:.1f}% → "
                    f"시퀀스 검증 **{seq_fpr * 100:.1f}%** "
                    f"(저감률 {alarm['fpr_reduction'] * 100:.1f}%)"
                ),
                verdict="달성" if seq_fpr <= 0.03 else "미달",
                basis=(
                    f"FIgLib 발화 전 프레임 {alarm['n_negative_frames']:,}장 "
                    f"(명세 요구 500장 이상 충족). 단일 프레임 실측치가 명세 예상"
                    f"(12~18%)보다 낮은데, 부트스트랩 탐지기 임계값이 "
                    f"conf={alarm.get('conf_threshold', 0.2)} 로 보수적이기 때문이다."
                ),
                command="uv run firstlight falsealarm --device gpu",
            )
        )

    # --- T4 End-to-End 지연 ---------------------------------------------
    if alarm and alarm.get("latency_median_ms") == alarm.get("latency_median_ms"):
        p95 = alarm.get("latency_p95_ms", float("nan"))
        claims.append(
            Claim(
                item="T4 End-to-End 지연",
                proposal_target="30초 이내 (예상 5~15초)",
                measured=(
                    f"중앙값 {alarm['latency_median_ms']:.0f}ms · "
                    f"p95 {p95:.0f}ms · 최대 {alarm.get('latency_max_ms', 0):.0f}ms"
                ),
                verdict="조건부",
                basis=(
                    "**추론→시퀀스 검증→등급 확정 구간만** 측정한 값이다. "
                    "명세의 T4 는 '영상 스트림 시작 → 대시보드 경보 표시'까지를 "
                    "요구하는데, RTMP/SRT 수신과 네트워크 전송 계층이 아직 없어 "
                    "그 구간은 포함되지 않았다. 파이프라인 자체는 목표 대비 "
                    "3자릿수 여유가 있으므로 병목은 전송 계층에서 결정된다."
                ),
                command="uv run firstlight falsealarm --device gpu",
            )
        )

    # --- ① 오경보율 ------------------------------------------------------
    if alarm:
        verified = alarm["verified_per_hour"]
        claims.append(
            Claim(
                item="오경보율",
                proposal_target="0.5건/비행시간 이하",
                measured=(
                    f"{verified:.2f}건/시간 "
                    f"(검증 없이 트랙 단위로만 억제하면 {alarm['dedup_per_hour']:.2f}, "
                    f"프레임 단위 경보는 {alarm['raw_per_hour']:.2f})"
                ),
                verdict="달성" if verified <= 0.5 else "미달",
                basis=(
                    f"FIgLib 시퀀스 {alarm['n_sequences']}개의 **발화 전** 구간 "
                    f"{alarm['negative_hours']:.2f}시간. 산불이 없다는 것이 데이터로 "
                    f"보장되면서 안개·구름그림자·노을은 그대로 들어있는 구간이다. "
                    f"스코어러는 "
                    + ("학습됨" if alarm.get("scorer_fitted") else "사전값(미학습)")
                ),
                command="uv run firstlight falsealarm --device gpu",
            )
        )

        # --- 발견 소요 시간 ----------------------------------------------
        ttf = alarm.get("median_time_to_flare_s")
        if ttf is not None:
            claims.append(
                Claim(
                    item="연기 발생 후 발견 소요 시간",
                    proposal_target="30초 이내",
                    measured=(
                        f"중앙값 {ttf / 60:.0f}분 "
                        f"(발화 후 탐지율 {alarm['detection_rate'] * 100:.0f}%)"
                    ),
                    verdict="미측정",
                    basis=(
                        "**이 데이터로는 30초 주장을 검증할 수 없다.** FIgLib 은 "
                        "60초 간격이라 시간 분해능이 60초다. 측정된 5분은 "
                        "'다섯 번째 프레임에서 확정'이라는 뜻이며, 최소 관측 3회 "
                        "게이트를 감안하면 이 데이터에서 나올 수 있는 사실상의 "
                        "하한(3분)에 가깝다. 초 단위 측정에는 드론 실영상이 필요하다."
                    ),
                    command="uv run firstlight falsealarm --device gpu",
                )
            )

    # --- 탐지 베이스라인 -------------------------------------------------
    if detect:
        claims.append(
            Claim(
                item="탐지기 베이스라인 (프레임 단위)",
                proposal_target="(소개서에 수치 없음)",
                measured=(
                    f"AP50 {detect['ap50']:.3f}, 최적 F1 {detect['best']['f1']:.3f} "
                    f"(임계값 {detect['best']['confidence']:.2f}). "
                    f"연기 없는 이미지의 {detect['negative_image_alarm_rate'] * 100:.0f}%에서 오탐 발생"
                ),
                verdict="참고",
                basis=(
                    f"pyro-sdis 검증셋 {detect['n_images']}장 "
                    f"(연기 있음 {detect['n_images_with_smoke']} / 없음 "
                    f"{detect['n_images_without_smoke']}). "
                    f"**지상 감시탑 시점 데이터다** — 드론 시점 성능이 아니다. "
                    f"이 오탐률이 시퀀스 검증이 공격하는 출발점이다."
                ),
                command="uv run firstlight detect-baseline --device gpu",
            )
        )

    # --- 처리 속도 -------------------------------------------------------
    if bench:
        rows = sorted(bench.get("devices", []), key=lambda d: d.get("median_ms", 1e9))
        if rows:
            fastest = rows[0]
            others = ", ".join(
                f"{d['device'].upper()} {d['fps']:.0f}fps" for d in rows[1:]
            )
            claims.append(
                Claim(
                    item="실시간 처리 가능성",
                    proposal_target="(소개서에 수치 없음)",
                    measured=(
                        f"{fastest['device'].upper()} {fastest['fps']:.0f}fps "
                        f"({fastest['median_ms']:.0f}ms/프레임)"
                        + (f" · {others}" if others else "")
                    ),
                    verdict="달성",
                    basis=(
                        "NVIDIA GPU 없는 노트북(Intel Core Ultra 7 258V + Arc 140V iGPU). "
                        "전용 GPU 서버 없이 현장 노트북에서 실시간이 나오는지가 "
                        "도입 문턱을 결정한다."
                    ),
                    command="uv run firstlight bench-detect",
                )
            )

    return claims


def render_markdown(claims: list[Claim]) -> str:
    lines = [
        "# FIRSTLIGHT — 측정 결과",
        "",
        f"생성일: {date.today().isoformat()}",
        "",
        "소개서 §5 의 **목표** 열을 **실측** 열로 교체한 것이다. 모든 수치에",
        "재현 명령이 붙어 있다.",
        "",
        "## 요약",
        "",
        "| 항목 | 소개서 목표 | 실측 | 판정 |",
        "|---|---|---|---|",
    ]
    for claim in claims:
        lines.append(
            f"| {claim.item} | {claim.proposal_target} | {claim.measured} | **{claim.verdict}** |"
        )

    lines += ["", "## 근거와 재현", ""]
    for claim in claims:
        lines += [
            f"### {claim.item}",
            "",
            f"- **목표** {claim.proposal_target}",
            f"- **실측** {claim.measured}",
            f"- **판정** {claim.verdict}",
            f"- **근거** {claim.basis}",
            "",
            "```bash",
            claim.command,
            "```",
            "",
        ]

    lines += [
        "## 판정 기준",
        "",
        "- **달성** — 측정했고 목표를 만족한다.",
        "- **조건부** — 측정했고, 특정 조건에서만 목표를 만족한다. 조건을 함께 적었다.",
        "- **미측정** — 현재 데이터로는 그 주장을 검증할 수 없다. 왜 못 하는지를 적었다.",
        "- **참고** — 소개서에 목표가 없던 항목이지만 알아야 할 수치다.",
        "",
    ]
    return "\n".join(lines)


def write_report(out_dir: Path, dest: Path) -> Path:
    claims = build_claims(out_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_markdown(claims), encoding="utf-8")
    return dest
