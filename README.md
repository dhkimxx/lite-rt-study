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

