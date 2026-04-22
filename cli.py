import argparse
import sys
from pathlib import Path
from caption import CaptionConfig, ImageCaptioner


def main():
    parser = argparse.ArgumentParser(description="학습 이미지 캡셔닝 도구")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--image", type=str, help="단일 이미지 파일 경로")
    parser.add_argument("--dir", type=str, help="이미지가 들어 있는 디렉토리 경로")
    parser.add_argument(
        "--output-dir", type=str, help="캡션 저장 디렉토리 (미지정 시 이미지와 같은 폭더)"
    )

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    config = CaptionConfig(args.config)
    print(f"🤖 모델 로딩 중... {config.model_id}")
    captioner = ImageCaptioner(config)

    if args.image:
        caption = captioner.caption(args.image)
        print(f"\n📝 캡션:\n{caption}\n")
    elif args.dir:
        captioner.caption_batch(args.dir, args.output_dir)


if __name__ == "__main__":
    main()
