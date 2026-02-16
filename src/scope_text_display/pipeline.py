from typing import TYPE_CHECKING

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
        # Use PIL's default font for simplicity
        self.font = ImageFont.load_default()

    def __call__(self, **kwargs) -> dict:
        """Render text prompt as centered, auto-scaled text."""
        # Get the prompt text
        prompt = kwargs.get("prompt", "")
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

        # Auto-scale text to fit - start with a large font size and shrink if needed
        # For default font, we'll use a simple size estimation
        # PIL's default font is fixed size, so we'll need to estimate based on text length

        # For default font, approximate character width
        char_width = 6  # Approximate width of default font characters
        char_height = 13  # Approximate height of default font

        # Calculate how many characters fit
        max_chars_width = width // char_width
        max_lines = height // (char_height + 5)  # Add padding between lines

        # Simple word wrapping
        words = prompt.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_chars_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Limit to max lines
        lines = lines[:max_lines]

        # Calculate total text block height
        total_height = len(lines) * (char_height + 5)

        # Start y position to center vertically
        y = (height - total_height) // 2

        # Draw each line centered
        for line in lines:
            # Get line width for centering
            line_width = len(line) * char_width
            x = (width - line_width) // 2

            # Draw text
            draw.text((x, y), line, fill=text_color, font=self.font)
            y += char_height + 5

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
