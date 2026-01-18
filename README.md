# LiteRT (TFLite) 온디바이스 AI 최적화 프로젝트

본 프로젝트는 PyTorch로 학습된 **MobileNetV2** 이미지 분류 모델을 **모바일 기기(Android)**에서 빠르고 가볍게 실행하기 위해 **LiteRT (구 TensorFlow Lite)**로 변환 및 최적화하는 과정을 다룹니다.

## 핵심 성과 (Executive Summary)

| 최적화 지표 | 결과 | 비고 |
| :--- | :--- | :--- |
| **모델 크기 감축** | 13.4MB → **3.8MB** | **약 72% 용량 절감** (Full Integer Quantization) |
| **모바일 추론 속도** | 68ms → **39ms** | **약 1.7배 가속** (Android ARM64 CPU 기준) |
| **변환 전략** | **ai-edge Hybrid** | 구글 최신 툴체인(`ai-edge-torch`)과 안정성을 모두 확보한 하이브리드 전략 |

---

## 변환 파이프라인 (Methodology)

우리는 최적의 성능을 찾기 위해 총 3가지 경로를 탐색했으며, 최종적으로 **Strategy C (Hybrid)**를 채택했습니다.

```mermaid
graph TD
    A[PyTorch Model<br>(MobileNetV2)] -->|ai-edge-torch| B(FP32 TFLite)
    A -->|SavedModel Intercept| C{TensorFlow<br>SavedModel}
    C -->|TFLiteConverter<br>Standard Quantizer| D[INT8 TFLite<br>(Hybrid Route)]
    A -->|ONNX Export| E(ONNX Model)
    E -->|onnx2tf| D2[INT8 TFLite<br>(Legacy Route)]
    
    style B fill:#f9f,stroke:#333
    style D fill:#9f9,stroke:#333,stroke-width:2px
    style D2 fill:#eee,stroke:#333
```

### 1. FP32 (Baseline)
- 가장 기본적인 변환. 정확도는 유지되지만 용량이 크고 모바일 가속 효율이 낮음.

### 2. INT8 (ai-edge Hybrid) - **(Recommended)**
- **\"SavedModel Intercept\" 전략**: `ai-edge-torch`의 우수한 그래프 변환(Lowering) 기능만 사용해 TF SavedModel을 추출하고, 양자화는 검증된 표준 TFLiteConverter를 사용하는 방식입니다.
- **장점:** 복잡한 ONNX 변환 없이도 안정적이고 강력한 성능의 INT8 모델을 생성합니다.

### 3. INT8 (Legacy ONNX)
- 전통적인 방식(PyTorch->ONNX->TF). 결과물 성능은 Hybrid와 동일하지만, 외부 라이브러리 의존성이 높고 과정이 복잡합니다.

---

## 📊 벤치마크 결과 (Benchmark Results)

### 1. 안드로이드 기기 (ADB Real Device)
실제 **Android ARM64** 기기에서 측정한 결과입니다.

| 모델 (Model) | CPU Latency | GPU Latency | NPU Latency |
| :--- | :--- | :--- | :--- |
| **FP32** | 68.52 ms | **15.88 ms** | 68.86 ms |
| **INT8 (ai-edge Hybrid)** | **39.95 ms** | 17.57 ms | **39.86 ms** |

> **Insight:** 모바일 CPU에서는 INT8 모델이 **약 1.7배** 더 빠릅니다. NPU 지원 기기라면 더 큰 격차를 기대할 수 있습니다.

### 2. 로컬 PC (x86 CPU)
| 모델 | 크기 | Latency |
| :--- | :--- | :--- |
| FP32 | 13.43 MB | ~5.5 ms |
| INT8 | **3.83 MB** | ~6.9 ms |

> **Insight:** PC CPU(AVX)는 FP32 연산에 극도로 최적화되어 있어, INT8 변환 시 속도 이득보다는 **크기 이득(72% 감소)**에 주목해야 합니다.

---

## 실행 방법 (Getting Started)

### 1. 환경 설정 (Setup)
`uv` 패키지 매니저를 사용하여 의존성을 설치하고 환경을 구성합니다.

```bash
uv sync
```

### 2. 주요 스크립트 설명 (Scripts Details)

이 프로젝트는 `src` 디렉토리 내의 모듈화된 파이썬 스크립트들로 구성되어 있습니다.

| 스크립트 파일 | 역할 및 동작 원리 | 실행 예시 |
| :--- | :--- | :--- |
| **`src/convert_aiedge.py`** | **(핵심)** PyTorch 모델을 다운로드하고, `ai-edge-torch`로 SavedModel을 추출한 뒤, TFLiteConverter로 INT8 양자화를 수행합니다. | `uv run python src/convert_aiedge.py` |
| **`src/convert_onnx.py`** | (레거시) PyTorch 모델을 ONNX 포맷을 거쳐 TensorFlow 포맷으로 변환하고 양자화합니다. 비교 검증용입니다. | `uv run python src/convert_onnx.py` |
| **`src/benchmark.py`** | **PC(x86)** 환경에서 3가지 모델(FP32, Hybrid INT8, Legacy INT8)의 크기와 Latency를 자동으로 측정하고 비교합니다. | `uv run python src/benchmark.py` |
| **`src/benchmark_adb.py`** | PC와 연결된 **안드로이드 기기**에 모델과 벤치마크 툴을 전송하고, ADB Shell로 실행하여 **모바일 CPU/GPU/NPU** 성능을 측정합니다. | `uv run python src/benchmark_adb.py` |

### 3. 변환 실행 (Conversion)
새로운 Hybrid 방식으로 모델을 변환합니다.

```bash
# PyTorch -> SavedModel -> INT8 TFLite 변환
uv run python src/convert_aiedge.py
```

### 4. 벤치마크 (Benchmark)

**로컬 PC 측정:**
```bash
uv run python src/benchmark.py
```

**안드로이드 기기 측정 (ADB 필요):**
안드로이드 기기를 USB로 연결하고 USB 디버깅을 켠 상태여야 합니다.
```bash
uv run python src/benchmark_adb.py
```

### 4. 리포트 보기
상세한 과정과 분석은 Jupyter Notebook에 담겨 있습니다.

```bash
uv run jupyter notebook notebooks/report.ipynb
```
