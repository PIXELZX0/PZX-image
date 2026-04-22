# PZX-image

Good Data Good Model

PZX-image is a project for building an image generation model that supports both T2I and I2I workflows.

## 학습 이미지 캡셔닝

이 프로젝트는 학습 데이터셋 구축을 위해 이미지 캡셔닝 파이프라인을 포함합니다.

### 모델 선택

기본 캡셔닝 모델로 `google/gemma-4-e2b-it`을 사용합니다.  
요청하신 `HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive`는 현재 GGUF 포맷만 제공하므로 `transformers` 기반 파이프라인에서는 직접 로드할 수 없습니다. 향후 safetensors 버전이 공개되거나 직접 변환하면 `config.yaml`의 `model_id`만 교체하여 사용할 수 있습니다.

### 설치

```bash
pip install -r requirements.txt
```

### 사용법

단일 이미지 캡셔닝:
```bash
python cli.py --image path/to/image.jpg
```

디렉토리 일괄 캡셔닝:
```bash
python cli.py --dir path/to/images --output-dir path/to/captions
```

설정은 `config.yaml`에서 수정할 수 있습니다.
