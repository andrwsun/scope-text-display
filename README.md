# scope-text-display

GPU-accelerated text display plugin for Daydream Scope. Renders user prompts as centered, auto-scaled text.

## What it does

Takes text from Scope's prompt box and displays it as centered text on screen. Automatically wraps and scales text to fit the frame. Supports custom text colors and background opacity control.

## Installation

### From Git (for sharing)

```bash
# In Scope Settings > Plugins, install from:
git+https://github.com/andrwsun/scope-text-display.git
```

### Local Development

```bash
# In Scope Settings > Plugins, browse to:
/Users/andrew/Desktop/scope local/scope-text-display
```

Click **Install** and Scope will restart with the plugin loaded.

## Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| Text Red | Float | 0.0 - 1.0 | 1.0 | Red component of text color |
| Text Green | Float | 0.0 - 1.0 | 1.0 | Green component of text color |
| Text Blue | Float | 0.0 - 1.0 | 1.0 | Blue component of text color |
| Background Opacity | Float | 0.0 - 1.0 | 0.0 | Background opacity (0=transparent, 1=white) |

All parameters are **runtime** - adjust them live during streaming!

## Usage

1. Select **Text Display** from the pipeline dropdown
2. **Type your text** in the prompt box at the bottom of Scope
3. Press Enter or click outside the box to update
4. Adjust **text color** using RGB sliders:
   - R=1, G=1, B=1 = White text
   - R=1, G=0, B=0 = Red text
   - R=0, G=0, B=0 = Black text
5. Adjust **Background Opacity**:
   - 0.0 = Transparent (text only)
   - 0.5 = Semi-transparent white background
   - 1.0 = Solid white background

## Features

- ✅ **Prompt-driven**: Text from Scope's prompt box
- ✅ **Auto-scaling**: Text automatically wraps and fits the frame
- ✅ **Centered**: Horizontally and vertically centered
- ✅ **GPU-accelerated**: Uses PyTorch for fast rendering
- ✅ **Live control**: Adjust colors in real-time
- ✅ **Transparent support**: Can render text-only or with background

## How it works

1. Takes text from the `prompt` parameter (Scope's prompt box)
2. Renders text using PIL (Pillow) with default font
3. Auto-wraps text to fit frame width
4. Centers text block vertically and horizontally
5. Converts to PyTorch tensor for GPU processing
6. Applies background opacity blending
7. Returns as video frames for display

## Development

After editing the code:
1. Go to Settings > Plugins
2. Click **Reload** next to "scope-text-display"
3. Changes take effect immediately (no reinstall needed)

## Example Use Cases

- Live text overlays
- Lyric displays
- Prompt visualization
- Title cards
- Message boards
- Simple text-based content
