from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


class TextDisplayConfig(BasePipelineConfig):
    """Configuration for the Text Display pipeline."""

    pipeline_id = "text-display"
    pipeline_name = "Text Display"
    pipeline_description = "Renders user prompts as centered, auto-scaled text on screen"

    supports_prompts = True  # Enable the prompt box!

    modes = {"text": ModeDefaults(default=True)}

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
