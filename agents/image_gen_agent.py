from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional

from .base_agent import BaseAgent


class ImageGenAgent(BaseAgent):
    """Generates images using only local machine backends.

    The agent never calls a hosted image API. It tries local Diffusers
    weights first, then an optional loopback Automatic1111 server, then
    writes a deterministic local SVG preview so image mode always returns
    a usable local file without an API key.
    """

    LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[1]

    @classmethod
    def _output_dir(cls) -> Path:
        output_dir = cls._repo_root() / "frontend" / "static" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _wrap_text(text: str, width: int = 34, max_lines: int = 5) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if len(candidate) > width and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(" ".join(current))
        return lines or ["Generated image"]

    @staticmethod
    def _is_loopback_url(url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and (parsed.hostname or "") in ImageGenAgent.LOOPBACK_HOSTS

    def _save_pil_image(self, image: Any, prefix: str = "image") -> Optional[str]:
        filename = f"{prefix}_{int(time.time() * 1000)}.png"
        output_path = self._output_dir() / filename
        try:
            image.save(output_path)
        except Exception as exc:
            self._log("image_save_failed", {"error": str(exc)})
            return None
        return f"/static/generated/{filename}"

    def _save_png_bytes(self, image_bytes: bytes, prefix: str = "image") -> Optional[str]:
        filename = f"{prefix}_{int(time.time() * 1000)}.png"
        output_path = self._output_dir() / filename
        try:
            output_path.write_bytes(image_bytes)
        except OSError as exc:
            self._log("image_save_failed", {"error": str(exc)})
            return None
        return f"/static/generated/{filename}"

    def _local_diffusers_model_path(self) -> Optional[Path]:
        configured = os.getenv("DIZI_LOCAL_IMAGE_MODEL", "").strip()
        candidates = [Path(configured).expanduser()] if configured else []
        candidates.extend(
            [
                self._repo_root() / "models" / "image",
                self._repo_root() / "models" / "stable-diffusion",
                self._repo_root() / "models" / "sdxl",
                self._repo_root() / "models" / "flux",
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _call_local_diffusers(self, prompt: str) -> Optional[str]:
        model_path = self._local_diffusers_model_path()
        if model_path is None:
            self._log("local_diffusers_skipped", {"reason": "no local image model path found"})
            return None

        try:
            import torch
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(str(model_path), local_files_only=True)
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            steps = int(os.getenv("DIZI_LOCAL_IMAGE_STEPS", "20"))
            guidance = float(os.getenv("DIZI_LOCAL_IMAGE_GUIDANCE", "7.0"))
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
            return self._save_pil_image(image, prefix="local_image")
        except Exception as exc:
            self._log("local_diffusers_failed", {"error": str(exc), "model_path": str(model_path)})
            return None

    def _call_local_automatic1111(self, prompt: str) -> Optional[str]:
        base_url = os.getenv("DIZI_AUTOMATIC1111_URL", "").strip().rstrip("/")
        if not base_url:
            return None
        if not self._is_loopback_url(base_url):
            self._log("local_a1111_rejected", {"reason": "url is not loopback", "url": base_url})
            return None

        endpoint = f"{base_url}/sdapi/v1/txt2img"
        payload = json.dumps(
            {
                "prompt": prompt,
                "steps": int(os.getenv("DIZI_A1111_STEPS", "20")),
                "width": int(os.getenv("DIZI_A1111_WIDTH", "768")),
                "height": int(os.getenv("DIZI_A1111_HEIGHT", "768")),
                "batch_size": 1,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=float(os.getenv("DIZI_A1111_TIMEOUT", "120"))) as response:
                data = json.loads(response.read().decode("utf-8"))
            encoded_image = (data.get("images") or [""])[0]
            encoded_image = encoded_image.split(",", 1)[-1]
            image_bytes = base64.b64decode(encoded_image)
            return self._save_png_bytes(image_bytes, prefix="a1111_image")
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError, IndexError) as exc:
            self._log("local_a1111_failed", {"error": str(exc)})
            return None

    def _create_local_preview(self, prompt: str) -> Optional[str]:
        output_dir = self._output_dir()
        filename = f"preview_{int(time.time() * 1000)}.svg"
        output_path = output_dir / filename
        lines = self._wrap_text(prompt)
        palette_index = sum(ord(char) for char in prompt) % 4
        palettes = [
            ("#0b1020", "#2563eb", "#38bdf8"),
            ("#111827", "#7c3aed", "#f472b6"),
            ("#102018", "#059669", "#a3e635"),
            ("#1f1307", "#ea580c", "#facc15"),
        ]
        bg, primary, accent = palettes[palette_index]
        text_lines = "\n".join(
            f'<text x="48" y="{230 + i * 34}" fill="#f8fafc" font-size="24" font-family="Segoe UI, Arial, sans-serif">{escape(line)}</text>'
            for i, line in enumerate(lines)
        )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="768" viewBox="0 0 1024 768">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="{bg}"/>
      <stop offset="100%" stop-color="#020617"/>
    </linearGradient>
    <radialGradient id="glow" cx="70%" cy="35%" r="55%">
      <stop offset="0%" stop-color="{accent}" stop-opacity="0.75"/>
      <stop offset="100%" stop-color="{accent}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1024" height="768" fill="url(#bg)"/>
  <rect width="1024" height="768" fill="url(#glow)"/>
  <circle cx="790" cy="210" r="118" fill="{primary}" opacity="0.7"/>
  <circle cx="690" cy="310" r="76" fill="{accent}" opacity="0.75"/>
  <path d="M0 590 C190 500 310 690 500 600 C690 510 790 560 1024 470 L1024 768 L0 768 Z" fill="{primary}" opacity="0.34"/>
  <text x="48" y="120" fill="{accent}" font-size="34" font-weight="700" font-family="Segoe UI, Arial, sans-serif">DIZI-AI Local Image</text>
  <text x="48" y="170" fill="#cbd5e1" font-size="22" font-family="Segoe UI, Arial, sans-serif">Generated on this machine with local fallback</text>
  {text_lines}
</svg>
'''
        try:
            output_path.write_text(svg, encoding="utf-8")
        except OSError as exc:
            self._log("preview_save_failed", {"error": str(exc)})
            return None
        return f"/static/generated/{filename}"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(context.get("prompt") or context.get("description") or "").strip()
        self._log("start", {"prompt_length": len(prompt), "backend": "local_only"})

        if not prompt:
            output = "Describe the image you want to generate."
            self._log("fallback", {"output_length": len(output)})
            return {"output": output, "image_url": None, "image_at": datetime.utcnow().isoformat()}

        image_url = (
            self._call_local_diffusers(prompt)
            or self._call_local_automatic1111(prompt)
            or self._create_local_preview(prompt)
        )
        if image_url:
            output = f"Generated local image for: {prompt}"
        else:
            output = "I could not create a local image file for this request."

        self._log("success", {"has_image_url": bool(image_url), "local_only": True})
        return {"output": output, "image_url": image_url, "image_at": datetime.utcnow().isoformat()}
