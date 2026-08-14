# FIRSTLIGHT

드론·AI 기반 산불 조기탐지 파이프라인.
**탐지 → 시퀀스 검증 → 좌표 산출 → 등급 라우팅**을 사람 개입 없이 통과시킨다.

[![CI](https://github.com/ranssist/firstlight/actions/workflows/ci.yml/badge.svg)](https://github.com/ranssist/firstlight/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

> **산불에서 빛은 나쁜 소식이다.** 능선 너머로 첫 빛이 보였다는 건 이미 불이 났다는 뜻이고,
> 그 빛을 **누가 먼저 보느냐**가 100㎡와 100헥타르를 가른다.

산불 대응에서 가장 큰 손실은 진화 능력 부족이 아니라 **발견 지연**에서 나온다.
FIRSTLIGHT 는 진화가 아니라 그 앞단인 **감시 계층**만 노린다 — 대형 진화드론과
경쟁하는 것이 아니라, 그 드론이 떠야 할 이유를 30분 먼저 알려주는 레이어다.

**이 저장소의 목적은 제품이 아니라 측정이다.** 작품설명서가 내건 주장 하나하나에
측정 하네스를 붙이고, 재지 못한 것은 왜 못 쟀는지를 적었다.

---

## 30초 만에 확인하기

외부 데이터 없이 도는 검증부터. 합성 지형을 쓰므로 내려받기가 없다.

```bash
uv sync --group dev --extra cv --extra serve
```

```bash
uv run firstlight geo-selftest --synthetic --trials 200
```

부각별 CEP 표가 나온다. 여기서 **무노이즈 왕복오차 0.000 m** 를 확인할 수 있고,
그건 좌표 오차가 전부 측위·자세·DEM 에서 오는 것이지 구현에서 새는 것이 아니라는 뜻이다.

---

## 이 저장소가 증명하려는 것

작품설명서는 세 가지를 주장한다. 각 주장에는 대응하는 **측정 하네스**가 있고, 모든 수치에 재현 명령이 붙는다.

| 주장 | 구현 | 측정 | 재현 |
|---|---|---|---|
| ① 시간축 검증으로 오탐 억제 | `verify/` | T2 오탐률 (%·건/시간) | `firstlight falsealarm` |
| ② 자동 지오레퍼런싱 (50m 이내) | `geo/` | T3 폐루프 CEP 스윕 | `firstlight geo-selftest` |
| ③ 3단계 등급 라우팅 | `verify/criteria.py` | 억제율 · GLOW 트리아지 | `firstlight run` → `firstlight serve` |

관제 화면은 따로 볼 수 있다 — **[대시보드 데모](https://claude.ai/code/artifact/0339c414-9756-41fb-a891-007c23b39705)** (실제 FIgLib 시퀀스 처리 결과, 백엔드 없이 도는 정적 스냅샷).

### 등급은 점수가 아니라 조건 수로 정해진다

작품설명서 Ⅱ-1 이 판정 규칙을 명시한다 — **두 조건을 동시에 만족하면 FLARE, 하나만 만족하면 GLOW, 둘 다 미달이면 SPARK.**

| 조건 | 연기 | 안개 | 구름 그림자 | 노을·반사광 |
|---|---|---|---|---|
| **광류** — 상승·확산 운동 | ✓ | 정지 | 평행이동 | — |
| **마스크 면적** — 단조 증가 | ✓ | 불변 | 불변 | 불변/소멸 |

`verify/criteria.py` 가 이 규칙 자체다. 로지스틱 스코어러는 등급을 정하지 않고, 같은 등급 안의 우선순위와 재학습 환류를 담당하는 보조 신호로만 쓴다. 관제 요원에게 "왜 GLOW냐"고 물었을 때 **"둘 중 면적 조건만 통과했습니다"**라고 답할 수 있어야 하기 때문이다 — 대시보드가 그 두 줄을 그대로 보여준다.

FLARE 에는 명세에서 유도된 강등 조건이 두 개 더 붙는다: **관측 12프레임 미만**(명세의 "후속 12프레임 시퀀스 검증")과 **좌표 미발행**(표 3-1 의 FLARE 동작이 "좌표 확정"이다). 강등은 언제나 GLOW 까지이고 후보를 버리지 않는다.

**숫자가 없는 주장은 주장이 아니다.** `firstlight report` 가 이 결과들을 모아 소개서 §5의 "목표" 열을 "실측" 열로 교체한 문서를 만든다.

---

## 측정된 것 (요약)

전체 결과와 조건은 `out/RESULTS.md` 를 보라. 여기 있는 것은 요점이다.

**② 위치 정확도 — 조건부 달성.** 의성 실제 지형(Copernicus GLO-30) 위에서 고도 300m AGL 기준:

| 부각 | 사거리 | CEP50 (일반 GNSS) | CEP50 (RTK) |
|---:|---:|---:|---:|
| 5° | 3,108 m | 169 m | 51 m |
| 8° | 2,145 m | 92 m | 25 m |
| 15° | 1,213 m | 47 m | 15 m |
| 20° | 865 m | 34 m | 12 m |
| 45° | 399 m | 9 m | 4 m |
| 90° | 300 m | 3 m | 1 m |

읽는 법: **거리가 아니라 부각이 지배한다.** 소개서의 "50m 이내"는 일반 GNSS 기준 **부각 15° 이상**에서 성립하며 무조건 성립하는 값이 아니다. 그래서 엔진은 부각이 낮거나 CEP가 상한을 넘으면 **좌표를 발행하지 않고** GLOW로 강등한다.

무노이즈 폐루프 왕복오차는 **1.4e-9 m** — 오차 전부가 측위·자세·DEM에서 오는 것이지 구현에서 새는 것이 아니라는 뜻이다.

**① 오경보율 — 달성.** FIgLib 시퀀스 46개의 **발화 전** 구간, 즉 산불이 없다는 것이 데이터로 보장되면서 안개·구름그림자·노을은 그대로 들어있는 **28.83시간**:

| 경보 정책 | 건/시간 | 목표(0.5) 대비 |
|---|---:|---|
| raw — 탐지 1건 = 경보 1건 | 4.40 | 8.8배 초과 |
| dedup — 트랙 1개 = 경보 1건 | 1.53 | 3.1배 초과 |
| **verified — 2조건 검증 통과만** | **0.03** | **달성** |

작품설명서 표 Ⅲ-2 형식(프레임 단위 %)으로도 같은 구간을 잰다:

| 판정 방식 | 오탐률 | 명세 예상 |
|---|---:|---|
| 단일 프레임 | 7.3% | 12~18% |
| **시퀀스 검증** | **0.3%** | 3% 이하 |

저감률 95.3% (발화 전 프레임 1,730장 — 명세 요구 500장 이상 충족).

**트레이드오프를 함께 봐야 한다.** 명세의 2조건 게이트는 이전의 점수 임계값 방식보다 엄격해서, 오탐이 0.14→0.03건/시간으로 더 줄어든 대신 **발화 후 탐지율이 96%→80%로 내려갔다.** 명세가 "탐지율보다 오경보율을 먼저 방어한다"고 명시했으므로 이 방향이 설계 의도와 맞지만, 문턱값(`verify/criteria.py`의 `CriteriaConfig`)은 현장 데이터로 재조정할 대상이다.

**이 수치는 학습되지 않은 사전값 기준이라 데이터 누수가 없다.**

**③ 억제 — 동작.** FIgLib 시퀀스 1개(81프레임) 처리 시 판정 57회 → 실제 통지 **5건**.

**속도 — NVIDIA GPU 없이 실시간.** Intel Arc 140V iGPU에서 YOLO11s(1024px) 추론이 **31 fps**(CPU 대비 7.5배). NPU 23 fps. 전용 GPU 서버 없이 현장 노트북에서 돌아간다는 뜻이고, 이게 도입 문턱을 결정한다.

---

## 알려진 한계 (감추지 않는다)

1. **시점 도메인 격차** — 부트스트랩 탐지기는 지상 감시탑 데이터(pyro-sdis)로 학습됐다. 드론 시점 성능 저하는 확실하며 실비행 데이터 없이는 해소되지 않는다. **현재 검증된 것은 파이프라인과 좌표 정확도이지 드론 시점 탐지 성능이 아니다.**
2. **"30초 이내 발견"은 검증 불가** — FIgLib은 60초 간격이라 시간 분해능이 60초다. 초 단위 측정에는 드론 실영상(FLAME 등 29fps)이 필요하다.
3. **스코어러 학습은 미검증** — 60초 간격에서는 오탐 대부분이 다중 프레임 트랙을 이루지 못해 음성 표본이 모이지 않는다. 시퀀스를 16개→46개로 3배 늘려도 음성은 6개뿐이었다(양성 29). **표본 부족이 아니라 구조적 한계다.** 표본이 부족하면 학습을 **거부**하도록 해 두었다(클래스당 최소 8개) — 음성 1개로 계수 9개를 맞추면 로지스틱 회귀가 조용히 퇴화한 해를 내놓기 때문이다. 실제로 그렇게 나온 가중치는 계수 두 개가 정확히 `-0.00`으로 죽어 있었다. 위 수치가 **사전값 가중치** 기준인 이유다. 환류 루프 자체는 대시보드에서 동작하며, 드론 실영상(초 단위 간격)에서는 음성 표본이 훨씬 많이 나온다.
4. **DEM 30m 격자** — 급경사에서 오차가 커진다. 파일럿 시 국토지리정보원 5m DEM으로 교체.
5. **풍향 보정은 가장 약한 고리** — 단일 시점 영상에서는 연기 표류 시간이 관측되지 않는다. 기본값은 보정을 끄고(`drift_seconds=0`) 박스 하단 중앙을 연기 기저로 쓰는 데까지만 한다. 보정을 켜도 보정 전/후 좌표를 **둘 다** 반환한다.
6. **대기굴절 미모델링** — 부각 8° 이상만 좌표를 발행하는 현재 정책에서는 CEP 대비 작지만, 장거리 운용으로 확장하면 넣어야 한다.

---

## 설치

Python 3.13+ 는 rasterio/onnxruntime 휠 가용성이 불안정해 **3.12로 고정**한다. `uv` 가 인터프리터까지 받아온다.

```bash
uv sync --group dev --extra cv --extra serve
```

대시보드까지 쓰려면 Node 20+ 가 필요하다 (`web/` 이 React + shadcn/ui 앱이다).

```bash
cd web && npm install && npm run build
```

## 빠른 확인

외부 데이터 없이 몇 초 만에 도는 것부터. 합성 지형이라 다운로드가 필요 없다.

```bash
uv run firstlight geo-selftest --synthetic --trials 200
```

실제 지형으로 하려면 DEM을 먼저 받는다 (AWS 공개 데이터, 인증 불필요, 약 70MB).

```bash
uv run firstlight fetch-dem --site uiseong
```

테스트:

```bash
uv run pytest -q
```

## 전체 파이프라인

```bash
uv run python scripts/fetch_model.py
```

```bash
uv run python scripts/fetch_figlib.py --sequences 16
```

```bash
uv run firstlight run --source data/figlib/20160604_FIRE_rm-n-mobo-c --site uiseong --reset
```

관제 대시보드는 `web/` 의 React + [shadcn/ui](https://ui.shadcn.com) 앱이다. 처음 한 번은 빌드해야 한다:

```bash
cd web && npm install && npm run build
```

```bash
uv run firstlight serve
```

대시보드에서 GLOW 큐를 판정하면 그 라벨이 재학습 입력이 된다 (§4③ 환류 루프).

**프런트엔드를 고칠 때**는 두 프로세스를 나란히 띄운다. Vite 가 `/api` 를 :8000 으로 프록시하므로 HMR 을 쓰면서도 실제 이벤트 DB 를 그대로 본다.

```bash
uv run firstlight serve
```

```bash
cd web && npm run dev
```

## 측정 재현

```bash
uv run firstlight geo-selftest --site uiseong --trials 200 --out out/geo_accuracy_uiseong.json
```

```bash
uv run firstlight detect-baseline --device gpu --out out/detect_baseline_val200.json
```

```bash
uv run firstlight falsealarm --device gpu --out out/falsealarm_figlib.json
```

```bash
uv run firstlight report
```

---

## 라이선스

**이 저장소의 코드는 [Apache-2.0](LICENSE)** 이다. 번들·취득하는 제3자 자산은 각각 조건이 다르며 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) 에 정리했다. 요약:

| 대상 | 라이선스 | 저장소 포함 |
|---|---|---|
| FIRSTLIGHT 코드 | Apache-2.0 | ✅ |
| Paperlogy 폰트 | SIL OFL 1.1 | ✅ (`web/src/assets/fonts/`) |
| 탐지 모델 가중치 | ⚠️ **미해결** (아래) | ❌ 스크립트로 취득 |
| Copernicus DEM | 무료, 출처 표시 필요 | ❌ 스크립트로 취득 |

### ⚠️ 모델 가중치는 미해결 상태다

부트스트랩 탐지기는 [Pyronear yolo11s_sensitive-detector](https://huggingface.co/pyronear/yolo11s_sensitive-detector_v1.0.0) 가중치를 쓴다. 여기에 **정리되지 않은 충돌**이 있고, 공공 납품 전에 법률 검토가 필요하다.

| 출처 | 선언된 라이선스 |
|---|---|
| Pyronear HF 저장소 메타데이터 | `apache-2.0` |
| 우리가 받은 `model.onnx` 내부 메타데이터 | `AGPL-3.0 License (https://ultralytics.com/license)` |

ONNX 안의 AGPL 문자열은 Ultralytics 익스포터가 모든 산출물에 자동으로 박는 것이다. 쟁점은 **AGPL 도구로 학습된 가중치가 그 자체로 파생저작물인가**이며, Ultralytics는 그렇다는 입장(Enterprise 라이선스 판매의 근거)이고 Pyronear는 Apache-2.0으로 배포한다.

이 저장소가 한 것과 하지 않은 것:

- **한 것** — 추론을 `onnxruntime` / `openvino` 로만 구현해 `ultralytics` **패키지에 링크하지 않는다.** AGPL 코드와의 결합이라는 축 하나는 제거된다.
- **하지 않은 것** — 가중치의 라이선스 지위는 **해결되지 않았다.** ONNX 경로로 바꾼다고 가중치에 걸린 주장이 사라지지 않는다.

실제 납품 경로는 둘 중 하나여야 한다: Ultralytics Enterprise 라이선스를 구매하거나, 허용적 라이선스 프레임워크로 **자체 학습한 가중치로 교체**한다(D-FINE, RT-DETR 등 Apache-2.0 계열). `detect/detector.py` 의 `Detector` 프로토콜 뒤에 두었으므로 교체해도 파이프라인은 손대지 않는다.

---

## 데이터 출처

| 출처 | 용도 | 접근 |
|---|---|---|
| [Copernicus GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/) | 지형면 | AWS 공개, 인증 불필요 |
| [pyronear/pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis) | 탐지 베이스라인 | HF datasets-server |
| [HPWREN FIgLib](http://hpwren.ucsd.edu/HPWREN-FIgLib/) | 오경보율 (발화 전 40분) | 공개 HTTP |
| [기상청 단기예보](https://www.data.go.kr) | 풍향·풍속 | **서비스키 필요** (없으면 스텁) |

## 구조

```
src/firstlight/
├─ geo/         좌표계·광선-지형 교차·오차전파   ← 데이터 없이 검증 가능
├─ detect/      ONNX/OpenVINO 추론, 슬라이스 추론
├─ verify/      트래커·자기운동 보정·특징·스코어러
├─ events/      저장(sqlite)·등급 라우팅·억제
├─ evaluation/  측정 하네스 (이 저장소의 목적)
├─ api/         관제 백엔드 (web/dist 서빙)
└─ pipeline.py  프레임 하나가 경보가 되기까지

web/            React + TypeScript + Tailwind v4 + shadcn/ui + Leaflet
├─ src/components/ui/   shadcn 생성 컴포넌트 (건드리지 않는다)
├─ src/components/      대시보드 컴포넌트
├─ src/assets/fonts/    Paperlogy woff2 (번들)
├─ src/demo/            정적 데모용 고정 데이터 (build_demo_data.py 가 생성)
└─ src/lib/api.ts       백엔드 응답 타입 — models.py 와 맞춰야 한다
```

### 관제 화면이 보여주는 것

작품설명서 「관제 대시보드 화면 설계」의 항목들이다.

| 명세 항목 | 구현 |
|---|---|
| 지도 패널 | Leaflet · 탐지 이벤트 + **CEP50/CEP90 원** + 최근접 진화대 + 관측 지점·시선 |
| 이벤트 카드 | 등급 · 좌표 · 신뢰도 · **탐지 스냅샷** · **시퀀스 GIF** · 1클릭 판정 |
| 이력 타임라인 | **대응 상태(접수-출동-진화)** + 각 단계 시각 |
| 판단 근거 | 2조건 통과/미달을 수치와 함께 표시 |
| 실시간 수신 | WebSocket 푸시 (끊기면 폴링으로 물러나고 상태를 표시) |
| 순찰 경로 | ❌ 고정 자세 가정이라 경로가 존재하지 않는다 — 실비행 텔레메트리 필요 |
| 확산 예측 폴리곤 | ❌ 임상도(산림청 가입 필요) 미확보 |

**탐지 스냅샷**(`events/snapshot.py`)은 박스만이 아니라 주변 맥락을 함께 자른다 — 박스만 보면 안개인지 연기인지 알 수 없고, 요원의 1클릭 판정이 성립하지 않기 때문이다. 이벤트마다 별도 파일로 남긴다: 트랙 단위로 덮어쓰면 과거 이벤트가 나중 프레임의 그림을 가리키게 된다.

**시퀀스 GIF**(`scripts/build_sequence_gifs.py`)는 트랙 전체를 **고정 크롭**으로 묶는다. 프레임마다 박스를 따라가며 자르면 연기가 화면 가운데 머물러 "커진다"가 보이지 않는다. 고정해 두면 그 안에서 연기가 자라 올라가는 것이 그대로 읽힌다 — 시퀀스 검증이 판정 근거로 삼는 바로 그 움직임이라, 요원이 시스템의 판단을 눈으로 확인할 수 있다.

**대응 이력**(`ResponseStatus`)은 판정과 다른 축이다. 판정은 "연기가 맞았나"를 묻고 재학습 라벨이 되며, 대응 상태는 "그래서 어떻게 됐나"를 기록한다. 같은 필드에 섞으면 오탐으로 판정한 건에 '출동'을 표시할 수 없다. 각 단계에 시각을 남기고 되돌리기를 막는데, **대응 지연이 이 시스템이 줄이려는 값**이라 측정 가능해야 하기 때문이다.

**지도 배경**은 실시간 모드에서 OSM 타일을, 정적 데모에서는 **DEM 음영기복도**를 쓴다. 데모는 외부 요청이 불가능한 환경(아티팩트 CSP)에 배포되기 때문이고, 산불에서 읽어야 할 배경이 도로망보다 능선·계곡이라 오히려 맞는 선택이다. 화질의 상한은 픽셀 수가 아니라 **DEM 해상도**가 정한다 — 30m 격자로는 9km 범위에 표본이 300개뿐이라, 5m DEM을 `dem.local_dir`에 연결하면 6배 세밀해진다.

### UI 폰트

[Paperlogy](https://noonnu.cc/font_page/1456) (한국제지, SIL Open Font License) — 한글은 G마켓산스, 라틴은 Montserrat 계열이라 한글과 숫자가 같은 리듬으로 붙는다. **숫자 폭이 기본에서 이미 균일해** 좌표·CEP·지표가 세로로 정렬된다.

CDN 이 아니라 번들한다 — 관제실이 폐쇄망일 수 있고, 외부 CDN 이 죽으면 글꼴이 통째로 대체 폰트로 떨어지기 때문이다. 대시보드가 실제로 쓰는 4개 웨이트(400/500/700/800)만 싣는다. 웨이트당 약 160KB 라 안 쓰는 것까지 넣으면 용량만 는다.
