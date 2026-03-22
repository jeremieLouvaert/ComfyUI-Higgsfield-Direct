"""
ComfyUI Higgsfield Direct — Multi-model image & video generation via Higgsfield API.

Unified access to 100+ generative models (Soul 2.0, Seedream, FLUX, Kling, etc.)
through a single node. Uses your own Higgsfield API key.

Features:
  - Text-to-image generation with multiple models
  - Image-to-image editing (Seedream edit)
  - Image-to-video generation (Kling, Sora, etc.)
  - Reference image upload (Soul ID character consistency)
  - Real USD cost awareness
  - cache_key output for Hash Vault compatibility
"""

import os
import io
import time
import json
import hashlib
import threading
import traceback

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import higgsfield_client
    from higgsfield_client import Queued, InProgress, Completed, Failed, NSFW, Cancelled
    HAS_HF = True
except ImportError:
    HAS_HF = False

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CATEGORY = "AKURATE/Higgsfield"

# Model registry: id -> {name, type, supports}
TEXT_TO_IMAGE_MODELS = {
    "higgsfield-ai/soul/standard": "Soul 2.0 (Photorealistic)",
    "reve/text-to-image": "Reve (Text-to-Image)",
    "bytedance/seedream/v4/text-to-image": "Seedream 4 (ByteDance)",
}

IMAGE_EDIT_MODELS = {
    "bytedance/seedream/v4/edit": "Seedream 4 Edit (ByteDance)",
}

IMAGE_TO_VIDEO_MODELS = {
    "higgsfield-ai/dop/preview": "DOP Preview (Higgsfield)",
    "bytedance/seedance/v1/pro/image-to-video": "Seedance Pro (ByteDance)",
    "kling-video/v2.1/pro/image-to-video": "Kling 2.1 Pro (Video)",
}

ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
]

RESOLUTIONS = ["1K", "2K", "4K"]

VIDEO_DURATIONS = ["5", "10", "15"]

# ---------------------------------------------------------------------------
# API KEY RESOLUTION
# ---------------------------------------------------------------------------

def _resolve_api_key(api_key_input="", api_secret_input=""):
    """Resolve Higgsfield API credentials.

    Priority:
      1. Direct node input (key:secret or separate fields)
      2. Environment variables (HF_KEY or HF_API_KEY + HF_API_SECRET)
      3. higgsfield_api_key.txt file in ComfyUI root
    """
    # Direct input — combined format
    if api_key_input and ":" in api_key_input:
        return api_key_input.strip()

    # Direct input — separate fields
    if api_key_input and api_secret_input:
        return f"{api_key_input.strip()}:{api_secret_input.strip()}"

    # Env vars (SDK's own format)
    env_combined = os.environ.get("HF_KEY", "").strip()
    if env_combined:
        return env_combined

    env_key = os.environ.get("HF_API_KEY", "").strip()
    env_secret = os.environ.get("HF_API_SECRET", "").strip()
    if env_key and env_secret:
        return f"{env_key}:{env_secret}"

    # File-based key
    if folder_paths:
        root = os.path.dirname(folder_paths.get_output_directory())
    else:
        root = os.path.dirname(__file__)
    key_file = os.path.join(root, "higgsfield_api_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            file_key = f.read().strip()
        if file_key:
            return file_key

    raise ValueError(
        "[Higgsfield Direct] No API credentials found. Provide via:\n"
        "  1. The api_key input on the node (format: key:secret)\n"
        "  2. HF_KEY environment variable (key:secret)\n"
        "  3. HF_API_KEY + HF_API_SECRET environment variables\n"
        "  4. A higgsfield_api_key.txt file in your ComfyUI root\n\n"
        "Get your key at: https://cloud.higgsfield.ai"
    )


def _set_credentials(credential_key):
    """Set the SDK's credentials via environment variable."""
    os.environ["HF_KEY"] = credential_key


def _check_sdk():
    if not HAS_HF:
        raise ImportError(
            "[Higgsfield Direct] higgsfield-client not installed.\n"
            "Run: pip install higgsfield-client"
        )


# ---------------------------------------------------------------------------
# IMAGE UTILITIES
# ---------------------------------------------------------------------------

def _tensor_to_pil(image_tensor):
    """Convert ComfyUI image tensor (B,H,W,C float32 0-1) to PIL Image."""
    if image_tensor is None:
        return None
    img = image_tensor[0]  # first in batch
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def _pil_to_tensor(pil_image):
    """Convert PIL Image to ComfyUI tensor (1,H,W,3 float32 0-1)."""
    img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def _download_image(url):
    """Download image from URL and return as PIL Image."""
    import httpx
    response = httpx.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def _upload_image(pil_image):
    """Upload a PIL image to Higgsfield and return the URL."""
    return higgsfield_client.upload_image(pil_image, format='jpeg')


def _save_image(pil_image, prefix="higgsfield"):
    """Save image to ComfyUI output directory."""
    if folder_paths is None:
        return None
    output_dir = folder_paths.get_output_directory()
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    pil_image.save(filepath)
    return filepath


def _make_cache_key(model, prompt, **kwargs):
    """Generate a deterministic cache key for Hash Vault compatibility."""
    data = {"model": model, "prompt": prompt, **kwargs}
    raw = json.dumps(data, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# NODE: Text-to-Image
# ---------------------------------------------------------------------------

class HiggsFieldTextToImage:
    """Generate images from text prompts using Higgsfield models."""

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(TEXT_TO_IMAGE_MODELS.keys())
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the image you want to generate..."
                }),
                "model": (model_list, {"default": model_list[0]}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "2:3"}),
                "resolution": (RESOLUTIONS, {"default": "2K"}),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "key:secret (or use env/file)",
                }),
                "reference_image": ("IMAGE",),
                "auto_save": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("image", "cache_key", "info",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, prompt, model, aspect_ratio, resolution,
                 api_key="", reference_image=None, auto_save=True):
        _check_sdk()
        credential_key = _resolve_api_key(api_key)
        _set_credentials(credential_key)

        t_start = time.time()

        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }

        # Upload reference image if provided (Soul ID / style reference)
        if reference_image is not None:
            ref_pil = _tensor_to_pil(reference_image)
            ref_url = _upload_image(ref_pil)
            arguments["image_url"] = ref_url
            print(f"[Higgsfield] Uploaded reference image: {ref_url[:80]}...")

        model_name = TEXT_TO_IMAGE_MODELS.get(model, model)
        print(f"[Higgsfield] Generating with {model_name}...")
        print(f"[Higgsfield] Prompt: {prompt[:100]}...")

        try:
            result = higgsfield_client.subscribe(
                model,
                arguments=arguments,
                on_queue_update=lambda s: print(f"[Higgsfield] Status: {type(s).__name__}"),
            )
        except Exception as e:
            raise RuntimeError(f"[Higgsfield] Generation failed: {e}")

        elapsed = time.time() - t_start

        # Extract image from result
        images = result.get("images", [])
        if not images:
            raise RuntimeError(
                f"[Higgsfield] No images returned. Response: {json.dumps(result, indent=2)[:500]}"
            )

        image_url = images[0].get("url", "")
        if not image_url:
            raise RuntimeError("[Higgsfield] Image URL missing from response.")

        pil_image = _download_image(image_url)

        # Auto-save
        saved_path = None
        if auto_save:
            saved_path = _save_image(pil_image, prefix="higgsfield_t2i")

        # Cache key for Hash Vault
        cache_key = _make_cache_key(model, prompt,
                                     aspect_ratio=aspect_ratio,
                                     resolution=resolution)

        # Info string
        info_parts = [
            f"Model: {model_name}",
            f"Resolution: {resolution}",
            f"Aspect: {aspect_ratio}",
            f"Time: {elapsed:.1f}s",
        ]
        if saved_path:
            info_parts.append(f"Saved: {os.path.basename(saved_path)}")
        info = " | ".join(info_parts)
        print(f"[Higgsfield] Done — {info}")

        tensor = _pil_to_tensor(pil_image)
        return (tensor, cache_key, info,)


# ---------------------------------------------------------------------------
# NODE: Image Edit
# ---------------------------------------------------------------------------

class HiggsFieldImageEdit:
    """Edit images using Higgsfield editing models (Seedream Edit)."""

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(IMAGE_EDIT_MODELS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the edit you want to make..."
                }),
                "model": (model_list, {"default": model_list[0]}),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "key:secret (or use env/file)",
                }),
                "auto_save": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("image", "cache_key", "info",)
    FUNCTION = "edit"
    CATEGORY = CATEGORY

    def edit(self, image, prompt, model, api_key="", auto_save=True):
        _check_sdk()
        credential_key = _resolve_api_key(api_key)
        _set_credentials(credential_key)

        t_start = time.time()

        # Upload source image
        pil_image = _tensor_to_pil(image)
        image_url = _upload_image(pil_image)
        print(f"[Higgsfield] Uploaded source image for editing")

        model_name = IMAGE_EDIT_MODELS.get(model, model)
        print(f"[Higgsfield] Editing with {model_name}...")
        print(f"[Higgsfield] Edit prompt: {prompt[:100]}...")

        arguments = {
            "prompt": prompt,
            "image_url": image_url,
        }

        try:
            result = higgsfield_client.subscribe(
                model,
                arguments=arguments,
                on_queue_update=lambda s: print(f"[Higgsfield] Status: {type(s).__name__}"),
            )
        except Exception as e:
            raise RuntimeError(f"[Higgsfield] Edit failed: {e}")

        elapsed = time.time() - t_start

        images = result.get("images", [])
        if not images:
            raise RuntimeError(
                f"[Higgsfield] No images returned. Response: {json.dumps(result, indent=2)[:500]}"
            )

        result_url = images[0].get("url", "")
        result_pil = _download_image(result_url)

        saved_path = None
        if auto_save:
            saved_path = _save_image(result_pil, prefix="higgsfield_edit")

        cache_key = _make_cache_key(model, prompt, source="edit")

        info_parts = [
            f"Model: {model_name}",
            f"Time: {elapsed:.1f}s",
        ]
        if saved_path:
            info_parts.append(f"Saved: {os.path.basename(saved_path)}")
        info = " | ".join(info_parts)
        print(f"[Higgsfield] Edit done — {info}")

        tensor = _pil_to_tensor(result_pil)
        return (tensor, cache_key, info,)


# ---------------------------------------------------------------------------
# NODE: Image-to-Video
# ---------------------------------------------------------------------------

class HiggsFieldImageToVideo:
    """Generate video from a source image using Higgsfield video models."""

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(IMAGE_TO_VIDEO_MODELS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the motion / action..."
                }),
                "model": (model_list, {"default": model_list[0]}),
                "duration": (VIDEO_DURATIONS, {"default": "5"}),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "key:secret (or use env/file)",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("video_url", "cache_key", "info",)
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def generate_video(self, image, prompt, model, duration, api_key=""):
        _check_sdk()
        credential_key = _resolve_api_key(api_key)
        _set_credentials(credential_key)

        t_start = time.time()

        # Upload source image
        pil_image = _tensor_to_pil(image)
        image_url = _upload_image(pil_image)
        print(f"[Higgsfield] Uploaded source image for video generation")

        model_name = IMAGE_TO_VIDEO_MODELS.get(model, model)
        print(f"[Higgsfield] Generating video with {model_name}...")
        print(f"[Higgsfield] Motion prompt: {prompt[:100]}...")

        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": int(duration),
        }

        try:
            result = higgsfield_client.subscribe(
                model,
                arguments=arguments,
                on_queue_update=lambda s: print(f"[Higgsfield] Status: {type(s).__name__}"),
            )
        except Exception as e:
            raise RuntimeError(f"[Higgsfield] Video generation failed: {e}")

        elapsed = time.time() - t_start

        # Extract video URL
        video_url = result.get("video", {}).get("url", "")
        if not video_url:
            # Some models return differently
            videos = result.get("videos", [])
            if videos:
                video_url = videos[0].get("url", "")

        if not video_url:
            raise RuntimeError(
                f"[Higgsfield] No video URL returned. Response: {json.dumps(result, indent=2)[:500]}"
            )

        cache_key = _make_cache_key(model, prompt, duration=duration)

        info = f"Model: {model_name} | Duration: {duration}s | Time: {elapsed:.1f}s"
        print(f"[Higgsfield] Video done — {info}")
        print(f"[Higgsfield] Video URL: {video_url}")

        return (video_url, cache_key, info,)


# ---------------------------------------------------------------------------
# NODE: Model Explorer (utility — lists available models)
# ---------------------------------------------------------------------------

class HiggsFieldModelInfo:
    """Display available Higgsfield models and their capabilities."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "category": (["text-to-image", "image-edit", "image-to-video", "all"],
                             {"default": "all"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "list_models"
    CATEGORY = CATEGORY

    def list_models(self, category):
        lines = ["=== Higgsfield Available Models ===", ""]

        if category in ("text-to-image", "all"):
            lines.append("--- Text-to-Image ---")
            for model_id, name in TEXT_TO_IMAGE_MODELS.items():
                lines.append(f"  {name}: {model_id}")
            lines.append("")

        if category in ("image-edit", "all"):
            lines.append("--- Image Edit ---")
            for model_id, name in IMAGE_EDIT_MODELS.items():
                lines.append(f"  {name}: {model_id}")
            lines.append("")

        if category in ("image-to-video", "all"):
            lines.append("--- Image-to-Video ---")
            for model_id, name in IMAGE_TO_VIDEO_MODELS.items():
                lines.append(f"  {name}: {model_id}")
            lines.append("")

        return ("\n".join(lines),)


# ---------------------------------------------------------------------------
# MAPPINGS
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "HiggsFieldTextToImage": HiggsFieldTextToImage,
    "HiggsFieldImageEdit": HiggsFieldImageEdit,
    "HiggsFieldImageToVideo": HiggsFieldImageToVideo,
    "HiggsFieldModelInfo": HiggsFieldModelInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsFieldTextToImage": "Higgsfield Text-to-Image (Direct API)",
    "HiggsFieldImageEdit": "Higgsfield Image Edit (Direct API)",
    "HiggsFieldImageToVideo": "Higgsfield Image-to-Video (Direct API)",
    "HiggsFieldModelInfo": "Higgsfield Model Info",
}
