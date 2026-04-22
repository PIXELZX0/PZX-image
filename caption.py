import os
from pathlib import Path
from typing import Union, List, Dict, Optional
from PIL import Image
import yaml
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM


class CaptionConfig:
    """설정 파일(config.yaml)을 파싱하는 간단한 래퍼."""

    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        m = cfg["model"]
        c = cfg["caption"]

        self.model_id: str = m["model_id"]
        self.device_map: str = m.get("device_map", "auto")
        self.dtype_str: str = m.get("dtype", "auto")
        self.max_new_tokens: int = m.get("max_new_tokens", 512)
        self.do_sample: bool = m.get("do_sample", True)
        self.temperature: float = m.get("temperature", 1.0)
        self.top_p: float = m.get("top_p", 0.95)
        self.top_k: int = m.get("top_k", 64)

        self.prompt: str = c.get("prompt", "Describe this image in detail.")
        self.output_extension: str = c.get("output_extension", ".txt")

    @property
    def torch_dtype(self):
        if self.dtype_str == "auto":
            return "auto"
        return getattr(torch, self.dtype_str)


class ImageCaptioner:
    """Gemma 4 기반 멀티모달 캡셔너."""

    def __init__(self, config: CaptionConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
        )

    def caption(self, image_path: Union[str, Path]) -> str:
        """단일 이미지에 대해 캡션을 생성합니다."""
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path.absolute())},
                    {"type": "text", "text": self.config.prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        result = self.processor.parse_response(response)
        return result

    def caption_batch(
        self,
        image_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Optional[str]]:
        """디렉토리 내 이미지들을 배치 처리하고 캡션 파일을 저장합니다."""
        image_dir = Path(image_dir)
        out_dir = Path(output_dir) if output_dir else image_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in extensions])

        results: Dict[str, Optional[str]] = {}
        for img_path in image_files:
            try:
                caption = self.caption(img_path)
                results[img_path.name] = caption

                out_path = out_dir / (img_path.stem + self.config.output_extension)
                out_path.write_text(caption, encoding="utf-8")
                print(f"✅ {img_path.name} → {out_path.name}")
            except Exception as exc:
                print(f"❌ {img_path.name}: {exc}")
                results[img_path.name] = None

        return results
