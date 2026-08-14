# 제3자 자산 및 라이선스

이 저장소가 **번들하거나 내려받는** 것들의 출처와 조건이다.
자체 코드의 라이선스는 [LICENSE](LICENSE) 를 따른다.

---

## 1. 번들된 자산 (저장소에 포함)

### Paperlogy (UI 폰트)

| | |
|---|---|
| 위치 | `web/src/assets/fonts/Paperlogy-*.woff2` (4개 웨이트) |
| 라이선스 | **SIL Open Font License 1.1** |
| 저작권 | 기획 김도균(Paperlogy) · 제작 이주임(PTKKUN) |
| 배포처 | https://freesentation.blog/paperlogyfont |
| 웹폰트 미러 | https://github.com/fonts-archive/Paperlogy |

배포처가 명시한 조건:

> 글꼴 단독 판매 또는 글꼴 라이선스 변경을 제외한 모든 상업적 행위 및 수정,
> 재배포가 가능합니다.

OFL 은 폰트 파일의 재배포를 허용하되 **저작권 표시와 라이선스 고지를 함께
배포할 것**을 요구한다. 이 문서가 그 고지다. 폰트를 단독 판매하거나 라이선스를
바꿔서는 안 된다.

### 데모 데이터

| | |
|---|---|
| 위치 | `web/src/demo/*.json` |
| 내용 | 이벤트 스냅샷, 현장 설정, DEM 음영기복도 |

지어낸 값이 아니라 **HPWREN FIgLib 시퀀스를 실제 파이프라인에 통과시킨 결과**와
Copernicus DEM 에서 계산한 음영기복도다. 원본 이미지는 포함하지 않는다.
재생성:

```bash
uv run python scripts/build_demo_data.py
```

---

## 2. 내려받는 자산 (저장소에 **미포함**, `.gitignore` 처리)

### 탐지 모델 가중치 — ⚠️ 라이선스 미해결

| | |
|---|---|
| 위치 | `models/pyronear/model.onnx` (`scripts/fetch_model.py` 로 취득) |
| 출처 | https://huggingface.co/pyronear/yolo11s_sensitive-detector_v1.0.0 |

**상충하는 두 표시가 있고, 이 저장소는 그것을 해결하지 못했다:**

| 출처 | 선언된 라이선스 |
|---|---|
| Pyronear HF 저장소 메타데이터 | `apache-2.0` |
| 내려받은 `model.onnx` 내부 메타데이터 | `AGPL-3.0 License (ultralytics.com/license)` |

ONNX 안의 AGPL 문자열은 Ultralytics 익스포터가 모든 산출물에 자동으로 넣는
것이다. 쟁점은 **AGPL 도구로 학습된 가중치가 그 자체로 파생저작물인가**이며,
Ultralytics 는 그렇다는 입장(Enterprise 라이선스 판매의 근거), Pyronear 는
Apache-2.0 으로 배포한다.

이 저장소가 한 것과 하지 않은 것:

- **한 것** — 추론을 `onnxruntime` / `openvino` 로만 구현해 `ultralytics`
  **패키지에 링크하지 않는다.** AGPL 코드와의 결합이라는 축은 제거된다.
- **하지 않은 것** — **가중치의 라이선스 지위는 그대로 남는다.** 추론 경로를
  바꾼다고 가중치에 걸린 주장이 사라지지 않는다.

실제 배포·납품 전에는 둘 중 하나를 택해야 한다:
Ultralytics Enterprise 라이선스를 구매하거나, 허용적 라이선스 프레임워크로
**자체 학습한 가중치로 교체**한다(D-FINE, RT-DETR 등). 교체 지점은
`src/firstlight/detect/detector.py` 의 `Detector` 프로토콜 하나다.

### 수치표고모델 (DEM)

| | |
|---|---|
| 출처 | Copernicus DEM GLO-30 — https://registry.opendata.aws/copernicus-dem/ |
| 조건 | ESA / Copernicus 무료 이용, **출처 표시 필요** |

> © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided
> under COPERNICUS by the European Union and ESA; all rights reserved.

작품설명서가 지정한 국토지리정보원 5m DEM 은 가입·수동 내려받기가 필요해
기본 경로에 넣지 않았다. `configs/sites/*.yaml` 의 `dem.local_dir` 로 연결한다.

### 산불 데이터셋

| 데이터셋 | 출처 | 용도 | 조건 |
|---|---|---|---|
| pyro-sdis | https://huggingface.co/datasets/pyronear/pyro-sdis | 탐지 베이스라인 | 저장소 표기 확인 |
| HPWREN FIgLib | http://hpwren.ucsd.edu/HPWREN-FIgLib/ | 오경보율 측정 | 연구·교육 목적, HPWREN/UCSD 출처 표시 |

두 데이터셋 모두 **지상 감시탑 시점**이다. 드론 시점 성능의 근거가 아니라는
점은 [README](README.md) 의 한계 항목에 적어 두었다.

### 기상 자료

| | |
|---|---|
| 출처 | 기상청 단기예보 조회서비스 (공공데이터포털) |
| 조건 | **서비스키 발급 필요** — `KMA_SERVICE_KEY` 또는 `configs/secrets.yaml` |

키가 없으면 설정된 고정 풍향으로 물러나며, 파이프라인은 계속 동작한다.

---

## 3. 소프트웨어 의존성

Python·npm 의존성 목록과 각각의 라이선스는 `pyproject.toml` / `uv.lock` 과
`web/package.json` / `web/package-lock.json` 에서 확인할 수 있다.

특기할 것:

| 패키지 | 라이선스 | 비고 |
|---|---|---|
| `ultralytics` | AGPL-3.0 | **의존성에 넣지 않았다.** 추론은 onnxruntime/openvino 로만 한다 |
| `leaflet`, `react-leaflet` | BSD-2-Clause / Hippocratic 아님 | 지도 |
| `shadcn/ui` 생성 컴포넌트 | MIT | `web/src/components/ui/` — 복사본이라 우리 코드로 관리된다 |
| OpenStreetMap 타일 | ODbL (데이터) | 실시간 모드에서만 사용. 데모는 DEM 음영기복도를 쓴다 |
