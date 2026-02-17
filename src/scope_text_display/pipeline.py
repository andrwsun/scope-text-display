from typing import TYPE_CHECKING
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

from scope.core.pipelines.interface import Pipeline

from .schema import TextDisplayConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class TextDisplayPipeline(Pipeline):
    """GPU-accelerated text display pipeline that renders prompts as centered text."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return TextDisplayConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Get requested font name from load-time parameters
        requested_font = kwargs.get("font_name", "Helvetica")

        # Try to find the requested font with fallback chain
        self.font_path = None
        self.font_name = None
        self._load_font(requested_font)

        # Cache for font size calculations
        self._font_cache = {}  # {(prompt_text, width, height): (font_size, lines)}

    def _load_font(self, requested_font: str):
        """Load font with fallback chain for cross-platform compatibility."""
        import platform
        import os

        system = platform.system()

        # Map font names to potential file paths on different systems
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

        # Generic fallbacks for fonts not in the map
        generic_fallbacks = {
            "Darwin": [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNS.ttf",
            ],
            "Linux": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ],
            "Windows": [
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\segoeui.ttf",
            ],
        }

        # Try requested font paths
        if requested_font in font_map and system in font_map[requested_font]:
            for font_path in font_map[requested_font][system]:
                if os.path.exists(font_path):
                    self.font_path = font_path
                    self.font_name = requested_font
                    sys.stderr.write(f"[TEXT DISPLAY] Loaded font: {requested_font} from {font_path}\n")
                    sys.stderr.flush()
                    return

        # Try system-specific fallbacks
        sys.stderr.write(f"[TEXT DISPLAY] Warning: '{requested_font}' not found, trying fallbacks...\n")
        sys.stderr.flush()

        if system in generic_fallbacks:
            for font_path in generic_fallbacks[system]:
                if os.path.exists(font_path):
                    self.font_path = font_path
                    self.font_name = "System Default"
                    sys.stderr.write(f"[TEXT DISPLAY] Using fallback font from {font_path}\n")
                    sys.stderr.flush()
                    return

        # Final fallback: use PIL's built-in default (very small)
        sys.stderr.write(f"[TEXT DISPLAY] Warning: No TrueType fonts found, using PIL default (small)\n")
        sys.stderr.flush()
        self.font_name = "PIL Default"

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Get a font at the specified size."""
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size)
            except Exception:
                pass
        # Fallback to default
        return ImageFont.load_default()

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, add it anyway
                    lines.append(word)
                    current_line = ""

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]

    def __call__(self, **kwargs) -> dict:
        """Render text prompt as centered, auto-scaled text."""
        # Get the prompt text from the prompts array
        prompts = kwargs.get("prompts", [])
        prompt = ""

        if prompts and len(prompts) > 0 and isinstance(prompts, list):
            # Extract the text from the first prompt object
            first_prompt = prompts[0]
            if isinstance(first_prompt, dict) and "text" in first_prompt:
                prompt = first_prompt["text"]

        # Fallback if no prompt provided
        if not prompt:
            prompt = "Enter text in the prompt box"

        # Read runtime parameters
        text_r = kwargs.get("text_r", 1.0)
        text_g = kwargs.get("text_g", 1.0)
        text_b = kwargs.get("text_b", 1.0)
        bg_opacity = kwargs.get("bg_opacity", 0.0)

        # Get output resolution from kwargs (Scope provides this)
        height = kwargs.get("height", 512)
        width = kwargs.get("width", 512)

        # Convert text color from [0,1] to [0,255]
        text_color = (
            int(text_r * 255),
            int(text_g * 255),
            int(text_b * 255),
            255  # Full alpha for text
        )

        # Create image with RGBA
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Auto-scale text to fill the screen
        # Use cache to avoid recalculating font size for the same text
        max_width = int(width * 0.9)  # Use 90% of width for padding
        max_height = int(height * 0.9)  # Use 90% of height for padding

        cache_key = (prompt, width, height)

        if cache_key in self._font_cache:
            # Use cached values (silent - no logging spam)
            best_size, lines = self._font_cache[cache_key]
            font = self._get_font(best_size)
        else:
            # Binary search for optimal font size
            min_size = 10
            max_size = 500
            best_size = min_size

            while min_size <= max_size:
                mid_size = (min_size + max_size) // 2
                font = self._get_font(mid_size)

                # Wrap text at this font size
                lines = self._wrap_text(prompt, font, max_width, draw)

                # Calculate total height
                total_height = 0
                max_line_width = 0
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_height = bbox[3] - bbox[1]
                    line_width = bbox[2] - bbox[0]
                    total_height += line_height
                    max_line_width = max(max_line_width, line_width)

                # Add line spacing
                if len(lines) > 1:
                    total_height += (len(lines) - 1) * (mid_size // 4)

                # Check if it fits
                if total_height <= max_height and max_line_width <= max_width:
                    best_size = mid_size
                    min_size = mid_size + 1  # Try larger
                else:
                    max_size = mid_size - 1  # Try smaller

            # Use the best size found
            font = self._get_font(best_size)
            lines = self._wrap_text(prompt, font, max_width, draw)

            # Cache the result
            self._font_cache[cache_key] = (best_size, lines)

            # Debug: log font information to stderr
            sys.stderr.write(f"\n[TEXT DISPLAY] Rendering Info:\n")
            sys.stderr.write(f"  Font: {self.font_name}\n")
            sys.stderr.write(f"  Font path: {self.font_path}\n")
            sys.stderr.write(f"  Font size: {best_size}px\n")
            sys.stderr.write(f"  Text content: '{prompt}'\n")
            sys.stderr.write(f"  Number of lines: {len(lines)}\n")
            sys.stderr.write(f"  Lines: {lines}\n")
            sys.stderr.write(f"  Resolution: {width}x{height}\n")
            sys.stderr.flush()

        # Calculate total text block height for centering
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])

        line_spacing = best_size // 4
        total_height = sum(line_heights) + line_spacing * (len(lines) - 1)

        # Start y position to center vertically
        y = (height - total_height) // 2

        # Draw each line centered
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = (width - line_width) // 2

            # Draw text
            draw.text((x, y), line, fill=text_color, font=font)
            y += line_heights[i] + line_spacing

        # Convert to tensor
        img_array = torch.tensor(list(img.getdata()), dtype=torch.float32, device=self.device)
        img_array = img_array.reshape(height, width, 4) / 255.0  # Normalize to [0, 1]

        # Separate RGB and alpha
        rgb = img_array[..., :3]
        alpha = img_array[..., 3:4]

        # Create background based on opacity
        # bg_opacity: 0 = transparent (use alpha), 1 = white background
        if bg_opacity > 0:
            # Blend with white background
            white_bg = torch.ones_like(rgb, device=self.device)
            background = white_bg * bg_opacity + rgb * (1 - bg_opacity) * (1 - alpha)
            # Composite text over background
            result = background * (1 - alpha) + rgb * alpha
        else:
            # Just the text with its alpha (transparent background)
            result = rgb

        # Add batch dimension (T=1 for single frame)
        result = result.unsqueeze(0)  # (1, H, W, 3)

        return {"video": result.clamp(0, 1)}
