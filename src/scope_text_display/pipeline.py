from typing import TYPE_CHECKING
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scope.core.pipelines.interface import Pipeline

from .schema import TextDisplayConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class TextDisplayPipeline(Pipeline):
    """CPU-only text display pipeline. Uses zero VRAM - safe to run alongside large models."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return TextDisplayConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        # Always CPU - text rendering never needs GPU, keep VRAM free for real models
        self.device = torch.device("cpu")

        # Cache: font_name -> font_path
        self._font_path_cache = {}

        # Cache: (font_name, prompt, width, height) -> (font_size, lines)
        self._layout_cache = {}

        # Cache: full rendered frame
        # Key: (font_name, prompt, width, height, text_r, text_g, text_b, bg_opacity)
        self._frame_cache = {}
        self._frame_cache_key = None

        self._current_font = None

    def _get_font_path(self, requested_font: str) -> tuple[str | None, str]:
        """Get font path with fallback chain. Returns (font_path, font_name)."""
        if requested_font in self._font_path_cache:
            return self._font_path_cache[requested_font]

        import platform
        import os

        system = platform.system()

        font_map = {
            "Helvetica": {
                "Darwin": ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Helvetica.ttc"],
                "Linux": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
                "Windows": ["C:\\Windows\\Fonts\\arial.ttf"],
            },
            "Arial": {
                "Darwin": ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"],
                "Linux": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
                "Windows": ["C:\\Windows\\Fonts\\arial.ttf"],
            },
            "Times New Roman": {
                "Darwin": ["/Library/Fonts/Times New Roman.ttf"],
                "Linux": ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
                "Windows": ["C:\\Windows\\Fonts\\times.ttf"],
            },
            "Courier": {
                "Darwin": ["/System/Library/Fonts/Courier.dfont"],
                "Linux": ["/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
                "Windows": ["C:\\Windows\\Fonts\\cour.ttf"],
            },
            "Monaco": {
                "Darwin": ["/System/Library/Fonts/Monaco.dfont"],
                "Linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
                "Windows": ["C:\\Windows\\Fonts\\consola.ttf"],
            },
            "SF Pro Display": {
                "Darwin": ["/System/Library/Fonts/SFNS.ttf"],
                "Linux": [],
                "Windows": [],
            },
            "SF Mono": {
                "Darwin": ["/System/Library/Fonts/SFNSMono.ttf"],
                "Linux": [],
                "Windows": [],
            },
        }

        generic_fallbacks = {
            "Darwin": ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"],
            "Linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "Windows": ["C:\\Windows\\Fonts\\arial.ttf"],
        }

        if requested_font in font_map and system in font_map[requested_font]:
            for font_path in font_map[requested_font][system]:
                if os.path.exists(font_path):
                    result = (font_path, requested_font)
                    self._font_path_cache[requested_font] = result
                    return result

        if system in generic_fallbacks:
            for font_path in generic_fallbacks[system]:
                if os.path.exists(font_path):
                    result = (font_path, "System Default")
                    self._font_path_cache[requested_font] = result
                    return result

        result = (None, "PIL Default")
        self._font_path_cache[requested_font] = result
        return result

    def _get_font(self, font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
                    current_line = ""

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]

    def __call__(self, **kwargs) -> dict:
        """Render text prompt as centered, auto-scaled text. Zero VRAM usage."""
        # Extract prompt
        prompts = kwargs.get("prompts", [])
        prompt = ""
        if prompts and isinstance(prompts, list) and len(prompts) > 0:
            first = prompts[0]
            if isinstance(first, dict) and "text" in first:
                prompt = first["text"]
        if not prompt:
            prompt = "Enter text in the prompt box"

        # Read parameters
        requested_font = kwargs.get("font_name", "Helvetica")
        text_r = float(kwargs.get("text_r", 1.0))
        text_g = float(kwargs.get("text_g", 1.0))
        text_b = float(kwargs.get("text_b", 1.0))
        bg_opacity = float(kwargs.get("bg_opacity", 0.0))
        height = int(kwargs.get("height", 512))
        width = int(kwargs.get("width", 512))

        # Clear layout cache if font changed
        if self._current_font != requested_font:
            self._layout_cache.clear()
            self._frame_cache.clear()
            self._current_font = requested_font

        # Check full frame cache - return immediately if nothing changed
        frame_key = (requested_font, prompt, width, height, text_r, text_g, text_b, bg_opacity)
        if frame_key == self._frame_cache_key and self._frame_cache:
            return self._frame_cache

        font_path, font_display_name = self._get_font_path(requested_font)

        # Get or compute layout (font size + line breaks)
        layout_key = (requested_font, prompt, width, height)
        if layout_key in self._layout_cache:
            best_size, lines = self._layout_cache[layout_key]
        else:
            max_width = int(width * 0.9)
            max_height = int(height * 0.9)

            # Temporary draw surface for text measurement only
            tmp_img = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(tmp_img)

            min_size, max_size_b, best_size = 10, 500, 10
            while min_size <= max_size_b:
                mid = (min_size + max_size_b) // 2
                font = self._get_font(font_path, mid)
                lines = self._wrap_text(prompt, font, max_width, draw)

                total_h = 0
                max_w = 0
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    total_h += bbox[3] - bbox[1]
                    max_w = max(max_w, bbox[2] - bbox[0])
                if len(lines) > 1:
                    total_h += (len(lines) - 1) * (mid // 4)

                if total_h <= max_height and max_w <= max_width:
                    best_size = mid
                    min_size = mid + 1
                else:
                    max_size_b = mid - 1

            font = self._get_font(font_path, best_size)
            lines = self._wrap_text(prompt, font, max_width, draw)
            self._layout_cache[layout_key] = (best_size, lines)

            sys.stderr.write(f"[TEXT DISPLAY] New layout: '{prompt[:30]}' | {best_size}px | {width}x{height} | font={font_display_name}\n")
            sys.stderr.flush()

        # Render to numpy array (CPU only, no VRAM)
        text_color = (int(text_r * 255), int(text_g * 255), int(text_b * 255), 255)
        font = self._get_font(font_path, best_size)

        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Calculate centering
        line_spacing = best_size // 4
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])
        total_height = sum(line_heights) + line_spacing * (len(lines) - 1)

        y = (height - total_height) // 2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (width - (bbox[2] - bbox[0])) // 2
            draw.text((x, y), line, fill=text_color, font=font)
            y += line_heights[i] + line_spacing

        # Fast PIL → numpy → torch (no list conversion, no VRAM)
        img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 4)
        rgb = img_np[..., :3]
        alpha = img_np[..., 3:4]

        if bg_opacity > 0:
            result = np.ones((height, width, 3), dtype=np.float32) * bg_opacity + rgb * (1 - bg_opacity) * (1 - alpha)
            result = result * (1 - alpha) + rgb * alpha
        else:
            result = rgb

        # Final tensor on CPU - no VRAM used
        tensor = torch.from_numpy(result.clip(0, 1)).unsqueeze(0)  # (1, H, W, 3)

        output = {"video": tensor}
        self._frame_cache = output
        self._frame_cache_key = frame_key

        return output
