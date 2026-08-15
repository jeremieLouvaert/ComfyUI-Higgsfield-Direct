# ComfyUI-Higgsfield-Direct

![The Higgs field, the one this API is named after](assets/higgs-field-banner.jpg)

**Direct Higgsfield API integration for ComfyUI**, unified access to 100+ generative models through a single node pack.

No credits system, no middleman. Uses your own Higgsfield API key with transparent pricing.

## Nodes

| Node | Description |
|------|-------------|
| **Higgsfield Text-to-Image** | Generate images from text prompts (Soul, Seedream 4, Reve) |
| **Higgsfield Image Edit** | Edit existing images with text instructions (Seedream 4 Edit) |
| **Higgsfield Image-to-Video** | Convert images to video (Kling 2.1, Seedance, DOP), returns a native ComfyUI `VIDEO` |
| **Higgsfield Model Info** | List all available models and capabilities |

## Features

- **Multi-model access**: switch between Soul, Seedream 4, Reve, Seedance, Kling and DOP
- **Reference image support**: upload reference images for style/character consistency (Soul ID)
- **Resolution control**: 1K, 2K, 4K output options
- **Aspect ratio presets**: 1:1, 2:3, 3:2, 4:3, 9:16, 16:9, 21:9
- **Video generation**: 5/10/15 second clips from a source image, wired straight into any `VIDEO` node
- **Auto-save**: generated images and videos saved to the ComfyUI output folder
- **Hash Vault compatible**: outputs `cache_key` for use with ComfyUI-API-Optimizer

## Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jeremieLouvaert/ComfyUI-Higgsfield-Direct.git
```

### 2. Install dependencies

```bash
pip install -r ComfyUI-Higgsfield-Direct/requirements.txt
```

### 3. Set up API key

Get your API key at [cloud.higgsfield.ai](https://cloud.higgsfield.ai)

**Option A** (recommended): a file.
Create `higgsfield_api_key.txt` in your ComfyUI root directory with content:
```
your_api_key:your_api_secret
```

**Option B**: environment variable.
```bash
export HF_KEY="your_api_key:your_api_secret"
```

**Option C**: direct input on each node via the `api_key` field.

### 4. Restart ComfyUI

Nodes appear under the **AKURATE/Higgsfield** category.

## Models

### Text-to-Image
| Model | ID |
|-------|-----|
| Soul 2.0 (Photorealistic) | `higgsfield-ai/soul/standard` |
| Reve | `reve/text-to-image` |
| Seedream 4 (ByteDance) | `bytedance/seedream/v4/text-to-image` |

### Image Edit
| Model | ID |
|-------|-----|
| Seedream 4 Edit | `bytedance/seedream/v4/edit` |

### Image-to-Video
| Model | ID |
|-------|-----|
| DOP Preview | `higgsfield-ai/dop/preview` |
| Seedance Pro | `bytedance/seedance/v1/pro/image-to-video` |
| Kling 2.1 Pro | `kling-video/v2.1/pro/image-to-video` |

## Pricing

Higgsfield charges per generation. Failed/NSFW requests are not charged.
Check [higgsfield.ai/pricing](https://higgsfield.ai/pricing) for current rates.

## A note on file retention

Higgsfield keeps generated files for a minimum of 7 days, after which the returned URL
stops resolving. Every node here downloads the result and writes it into your ComfyUI
output folder, so the artifact outlives the link. The `video_url` output is still there
if you want it, but treat it as short-lived. Set `auto_save` to false on the video node
and the file goes to the temp folder instead, cleared with the session.

## License

MIT

## Author

**Jeremie Louvaert**, [github.com/jeremieLouvaert](https://github.com/jeremieLouvaert)
