from typing import Literal, get_args
from pydantic import Field
import platform
import os

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


def _get_available_fonts() -> tuple[str, ...]:
    """Scan system for available fonts and return tuple of font names."""
    system = platform.system()

    # Map font names to potential file paths
    font_checks = {
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
        "Verdana": {
            "Darwin": ["/Library/Fonts/Verdana.ttf"],
            "Linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "Windows": ["C:\\Windows\\Fonts\\verdana.ttf"],
        },
        "Georgia": {
            "Darwin": ["/Library/Fonts/Georgia.ttf"],
            "Linux": ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
            "Windows": ["C:\\Windows\\Fonts\\georgia.ttf"],
        },
        "Comic Sans MS": {
            "Darwin": ["/Library/Fonts/Comic Sans MS.ttf"],
            "Linux": [],
            "Windows": ["C:\\Windows\\Fonts\\comic.ttf"],
        },
        "Impact": {
            "Darwin": ["/Library/Fonts/Impact.ttf"],
            "Linux": [],
            "Windows": ["C:\\Windows\\Fonts\\impact.ttf"],
        },
        "Trebuchet MS": {
            "Darwin": ["/Library/Fonts/Trebuchet MS.ttf"],
            "Linux": [],
            "Windows": ["C:\\Windows\\Fonts\\trebuc.ttf"],
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

    available = []

    for font_name, paths_dict in font_checks.items():
        if system in paths_dict:
            for font_path in paths_dict[system]:
                if os.path.exists(font_path):
                    available.append(font_name)
                    break  # Found this font, move to next

    # Always include at least one fallback
    if not available:
        available.append("System Default")

    return tuple(available)


# Dynamically generate available fonts
_AVAILABLE_FONTS = _get_available_fonts()


class TextDisplayConfig(BasePipelineConfig):
    """Configuration for the Text Display pipeline."""

    pipeline_id = "text-display"
    pipeline_name = "Text Display"
    pipeline_description = "Renders user prompts as centered, auto-scaled text on screen"

    supports_prompts = True  # AI generation prompt box — separate from the display text field below

    modes = {"text": ModeDefaults(default=True)}

    # --- Text Content ---

    text: str = Field(
        default="Hello Scope!",
        description="Text to display on screen. Controlled independently from the video model prompt.",
        json_schema_extra=ui_field_config(order=0, label="Text"),
    )

    # --- Font Selection (Runtime) ---

    # Dynamically constrain to available fonts only
    font_name: str = Field(
        default=_AVAILABLE_FONTS[0] if _AVAILABLE_FONTS else "System Default",
        description="Font to use for text rendering. Only shows fonts available on your system.",
        json_schema_extra={
            **ui_field_config(order=0, label="Font"),
            "enum": list(_AVAILABLE_FONTS),  # This creates the dropdown with only available fonts
        },
    )

    # --- Text Color ---

    text_r: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Text color - Red channel",
        json_schema_extra=ui_field_config(order=1, label="Text Red"),
    )

    text_g: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Text color - Green channel",
        json_schema_extra=ui_field_config(order=2, label="Text Green"),
    )

    text_b: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Text color - Blue channel",
        json_schema_extra=ui_field_config(order=3, label="Text Blue"),
    )

    # --- Background ---

    bg_opacity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Background opacity (0 = transparent/text only, 1 = solid white)",
        json_schema_extra=ui_field_config(order=10, label="Background Opacity"),
    )
