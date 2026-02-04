# MobileNetV2 양자화 및 온디바이스 성능 분석

## 1. 서론 (Introduction)

본 연구의 목적은 MobileNetV2 모델을 모바일 장치(Edge Device)에 효율적으로 배포하기 위한 최적의 양자화(Quantization) 전략을 도출하는 것이다. 특히 단순한 정적/동적 구분뿐만 아니라 **가중치(Weight)와 활성화값(Activation)의 정밀도(Bit-width)**, **데이터 분포(Real/Random)**, **양자화 단위(Granularity)** 등 다양한 변인이 모델 성능에 미치는 영향을 심층 분석하였다.

---

## 2. 실험 환경 및 방법 (Methodology)

### 2.1. 하드웨어 및 소프트웨어 환경

- **Host Environment**: Linux (x86_64), Python 3.10
- **Target Device**: Samsung Galaxy A7 (SM-A750N), CPU: Octa-core (Cortex-A73/A53), GPU: Mali-G71 MP2
- **Frameworks**: `torch`, `ai-edge-quantizer`, `tensorflow-lite`
- **Benchmark Tool**: TensorFlow Lite Benchmark Binary (Nightly, Android aarch64)

### 2.2. 모델 및 데이터셋

- **Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Calibration Dataset**: ImageNet-V2 (Matched Frequency, 1000 classes, 10k images)
- **Metrics**:
  - **MSE (Mean Squared Error)**: 원본 FP32 모델과의 출력 텐서 차이
  - **SNR (Signal-to-Noise Ratio)**: 양자화 노이즈 대비 신호 비율 (dB)
  - **Latency**: 온디바이스 평균 추론 속도 (ms)
  - **Model Size**: TFLite 파일 크기 (MB)

### 2.3. 실험 설계 (Experimental Matrix)

총 13가지 조합을 통해 비트수(WxAy)와 전략적 변인을 평가하였다.

| 실험 ID | Configuration | W-Bits | A-Bits | Strategy | 설명 |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **C0** | Baseline | 32 | 32 | FP32 | 성능 기준점 (Float32) |
| **W8A32** | Weight-Only | 8 | 32 | Weight-Only | 가중치만 8-bit, 연산은 FP32 |
| **W16A32**| Weight-Only | 16 | 32 | Weight-Only | 가중치만 16-bit (고정밀) |
| **W4A32** | Weight-Only | 4 | 32 | Weight-Only | 가중치만 4-bit (초경량) |
| **W8A16** | Dynamic | 8 | 16 | Dynamic | 가중치 8-bit, 실행 시 동적 16-bit |
| **W8A8** | Static | 8 | 8 | Static (Real) | Full Integer Quantization (표준) |
| **W4A8** | Static | 4 | 8 | Static (Real) | 4-bit 가중치 + 8-bit 연산 |

---

## 3. 실험 결과 (Results)

### 3.1. 정확도 및 효율성 (Accuracy & Efficiency)

| ID | Config | SNR (dB) | Size (MB) | Latency (ms) | 비고 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **C0** | FP32 | 116.19 | 13.43 | 5.49 | Baseline |
| **W8A32** | **W8 / A32** | **24.31** | 3.74 | **5.46** | **Best Choice (CPU)** |
| **W8A16** | W8 / A16 (Dyn) | 23.85 | 3.71 | 10.93 | 느린 속도 (Dynamic Overhead) |
| **W8A8** | **W8 / A8 (Static)**| **21.96** | 3.92 | 6.70 | **Best Choice (NPU)** |
| **W4A32** | W4 / A32 | 0.87 | 2.20 | 9.27 | **Collapse (정확도 붕괴)** |
| **W4A8** | W4 / A8 | 0.96 | 2.36 | 13.83 | **Collapse** |

### 3.2. 핵심 분석결과

1.  **비트수의 영향**:
    - **W8 (8-bit Weight)**: SNR 22~24dB로 매우 안정적입니다.
    - **W4 (4-bit Weight)**: 별도의 보정 없이 적용 시 SNR이 1dB 미만으로 떨어져, MobileNetV2에는 바로 **적용 불가능**합니다.
    
2.  **전략별 성능**:
    - **Weight-Only (W8A32)**: 가장 빠르고(5.5ms) 정확(24.3dB)하여 CPU 환경에서 최적입니다.
    - **Static Quantization (W8A8)**: 정확도를 2dB 정도 희생하지만, **Per-Channel** 적용 시 준수한 성능을 보이며 NPU 가속에 적합합니다.
    
3.  **오버헤드**:
    - **Dynamic Quantization**은 매번 Min/Max를 계산하느라 Static 방식보다 오히려 **1.6배** 느립니다.
    - **Per-Tensor** 양자화(W8A8-Tensor)는 Per-Channel 대비 SNR이 **6.5dB 하락**하므로 지양해야 합니다.

---

## 4. 결론 (Conclusion)

본 연구를 통해 MobileNetV2 모델의 모바일 최적화 전략을 다음과 같이 제안한다.

1.  **CPU 환경 (일반적인 앱 배포)**:
    - **추천**: **Static Quantization (C3/C4)**
    - **이유**: FP32 대비 **1.5배 빠른 속도**와 **4배 작은 모델 크기**를 제공하며, SNR 손실(약 22dB 유지)이 허용 가능한 수준이다.

2.  **GPU 환경 (고성능 필요 시)**:
    - **추천**: **Baseline FP32 (C0)**
    - **이유**: 해당 테스트 기기(Mali GPU)에서는 FP32/FP16 연산이 가장 최적화되어 있다. 양자화 모델은 오히려 오버헤드를 유발한다.

3.  **데이터 의존성**:
    - **Per-Tensor 양자화 지양**: 정확도 손실이 매우 크므로 반드시 **Per-Channel** 방식을 사용해야 한다.
    - **Calibration**: Random Data로도 수행 가능하나, 실서비스 수준의 정확도를 위해서는 실제 데이터 분포(Representitive Dataset)를 사용하는 것을 권장한다.

## 5. 사용법 (Usage)

### 5.1 실험 재현

```bash
# 1. 의존성 설치
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. 모델 변환 및 정량적 평가 (PC)
uv run src/run_experiments.py

# 3. 온디바이스 벤치마크 (Android 기기 연결 필요)
# benchmark_model 바이너리가 루트 디렉토리에 있어야 함
uv run src/benchmark_adb.py
```

### 5.2 결과 리포트

- [`notebooks/Quantization_Report.ipynb`](notebooks/Quantization_Report.ipynb): 상세 데이터 시각화(MSE, SNR, 온디바이스 Latency 비교) 및 분석 리포트.
- `results/android_benchmark_results.csv`: 온디바이스 벤치마크 로우 데이터.
