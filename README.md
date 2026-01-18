# LiteRT On-Device Image Classification Project

본 프로젝트는 PyTorch MobileNetV2 모델을 구글의 **LiteRT (구 TensorFlow Lite)** 런타임에서 실행하기 위한 온디바이스 AI 파이프라인 구축 예제입니다.

FP32 모델 변환부터 INT8 양자화(Dynamic Quantization), 추론, 그리고 성능 벤치마크까지의 [End-to-End] 과정을 다룹니다.

## 📁 프로젝트 구조

```bash
.
├── notebooks/          # Jupyter Notebook 리포트
│   └── report.ipynb    # [Main] 실행 및 결과 시각화
├── src/                # 파이썬 패키지 소스
│   ├── convert.py      # PyTorch -> LiteRT 변환
│   ├── inference.py    # LiteRT 추론 엔진
│   ├── benchmark.py    # 성능 측정 및 비교
│   └── data_loader.py  # Calibration 데이터 로더
├── models/             # 변환된 .tflite 모델 저장소
└── README.md           # 프로젝트 문서
```

## 🚀 시작하기 (Getting Started)

### 1. 환경 설정
본 프로젝트는 **`uv`** 패키지 매니저를 사용합니다.

```bash
# 의존성 설치
uv sync
```

### 2. Jupyter Notebook 실행 (추천)
모든 과정을 시각적으로 확인하고 리포트를 보려면 노트북을 실행하세요.

```bash
# VS Code에서 notebooks/report.ipynb 열기
# 또는 터미널에서:
uv run jupyter notebook notebooks/report.ipynb
```

## 📊 벤치마크 결과 요약 (Benchmark Results)

본 프로젝트에서는 세 가지 변환 전략을 시도하였으며, 최종적으로 **ONNX를 경유한 Static INT8 양자화**가 가장 유의미한 모델 크기 감소를 달성했습니다.

| 모델 (Model) | 파일 크기 (Size) | 평균 지연 시간 (Latency on CPU) | 비고 (Note) |
| :--- | :--- | :--- | :--- |
| **FP32 (Base)** | 13.43 MB | **12.44 ms** | 기준 모델 |
| **INT8 (Dynamic)** | 13.43 MB | 12.95 ms | `ai-edge-torch` 양자화 미적용됨 (FP32와 동일) |
| **INT8 (Legacy)** | **3.82 MB** | 46.58 ms | **ONNX Route 성공 (72% 압축)** |
| **INT8 (ai-edge)** | **3.83 MB** | 49.86 ms | **SavedModel Intercept 성공** |

### 💡 결과 분석 (Key Findings)


### 3. 방법론 비교: ONNX Route vs ai-edge Hybrid
두 가지 성공적인 변환 방식의 기술적 차이점을 비교합니다.

| 비교 항목 | **Legacy Route (ONNX)** | **Modern Hybrid Route (ai-edge)** |
| :--- | :--- | :--- |
| **변환 파이프라인** | PyTorch → **ONNX** → TF SavedModel → TFLite | PyTorch → **StableHLO** → TF SavedModel → TFLite |
| **핵심 기술** | **ONNX** (오픈 표준) | **Google StableHLO** (공식 권장) |
| **특징** | • 호환성 높지만 변환 단계가 복잡함<br>• 외부 라이브러리(`onnx`, `onnx2tf`) 필요 | • **Google 공식 파이프라인** (미래 지향적)<br>• 불필요한 제3 포맷 변환 없음<br>• 파이프라인이 더 간결함 |
| **결론** | **Result 동일** | **유지보수 및 미래 호환성 우수** |

### 4. 결과 분석 (Key Findings)
1.  **성공적인 경량화 (Compression Success):**
    *   **ONNX Route:** PyTorch -> ONNX -> TF -> TFLite 방식을 통해 **3.82MB** 달성.
    *   **ai-edge Hybrid Route:** PyTorch -> ai-edge (SavedModel) -> TFLite 방식을 통해 **3.83MB** 달성.
    *   두 방식 모두 FP32 대비 **약 72%의 용량 절감**효과를 보였습니다.
    *   이는 모바일 앱 배포 시 용량 절감에 큰 이점을 제공합니다.

2.  **CPU에서의 속도 저하 (Latency Trade-off on x86 CPU):**
    *   현재 테스트 환경(x86 CPU)에서는 INT8 모델이 FP32 모델보다 약 3.7배 느린 것으로 측정되었습니다.
    *   **원인:** 일반적인 PC/서버 CPU는 부동소수점(Float32) 연산에 최적화되어 있으며(AVX 등), INT8 연산 시 데이터 변환 오버헤드가 발생할 수 있습니다.
    *   **모바일 배포 시 예상:** Android/iOS 디바이스의 NPU(Neural Processing Unit)나 DSP는 INT8 연산에 특화되어 있어, 실제 모바일 환경에서는 INT8 모델이 훨씬 빠르고 전력 효율적일 것으로 예상됩니다.

## 🚀 결론 및 추천 (Conclusion)
*   **추천 방식:** **`ai-edge` Hybrid 방식** (SavedModel Intercept)이 ONNX 방식보다 관리 포인트가 적고 Google의 최신 기술 스택을 따르므로 장기적으로 더 유리합니다.
*   **모바일 배포:** **INT8 모델**을 사용하여 NPU 가속을 활용하세요.
*   **일반 서버 배포:** **FP32 모델**이 현재 CPU 환경에서는 더 빠를 수 있습니다.

## 📊 주요 기능 및 결과

### 1. 모델 변환 (Model Conversion)
`ai-edge-torch` 라이브러리를 사용하여 PyTorch 모델을 tflite 포맷으로 변환합니다.
- **FP32**: 기본 변환
- **INT8 (Dynamic)**: 동적 양자화 시도 (현재 환경 호환성 이슈로 FP32와 동일 크기 유지됨)

### 2. 성능 비교 (Benchmark Results)
`notebooks/report.ipynb`에서 최신 결과를 확인할 수 있습니다.

| Model | Size (MB) | Avg Latency (ms) | 비고 |
| :--- | :--- | :--- | :--- |
| **MobileNetV2 (FP32)** | 13.43 | ~13.37 | Base |
| **MobileNetV2 (INT8)** | 13.43 | ~12.69 | Dynamic Quantization 시도했으나 효과 미미 |

## 🛠️ 트러블슈팅
- **`ModuleNotFoundError: 'ai_edge_torch'`**: 추론 시에는 `ai-edge-torch`가 필요 없으므로 `inference.py` 등을 실행할 때 해당 모듈이 없어도 되도록 Lazy Import 처리가 되어 있습니다.
- **`numpy` 버전**: `ai-edge-torch`는 numpy 2.x를, `tflite-runtime`은 numpy 1.x를 선호합니다. `uv` 환경에서는 이를 유연하게 관리합니다.

