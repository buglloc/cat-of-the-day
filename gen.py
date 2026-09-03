import base64
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "https://llm.ive.buglloc.cc")
PROMPT_MODEL_ID = os.environ.get("PROMPT_MODEL_ID", "gpt-5-mini")
PROMPT_PROVIDER = os.environ.get("PROMPT_PROVIDER", "openai")
SELECTION_MODEL_ID = os.environ.get("SELECTION_MODEL_ID", "qwen/qwen3-vl-32b-instruct")
SELECTION_PROVIDER = os.environ.get("SELECTION_PROVIDER", "openrouter")
IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "gpt-image-1-mini")

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("S3_REGION")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")

DEFAULT_TIMEOUT_SEC = 600
TARGET_SIZE = (800, 480)
LLM_RETRIES = 3
NUM_CANDIDATES = 3

CONCEPT_SYSTEM_PROMPT = """You are a sharp editorial art director.
Turn one dry, original observation into a simple visual joke that is immediately
readable in a four-color minimalist cat poster. Follow every output constraint."""

CONCEPT_INSTRUCTION = """Create one original English quote and its poster composition.

Quote:
- One sentence and at most 55 characters; 60 is a hard limit.
- Use printable ASCII only. Write "I'm", never "I’m"; use a hyphen, never an
  en dash or em dash. Count every character, including spaces and punctuation.
- Dry, concise, slightly philosophical humor with an unexpected turn.
- It must sound natural, not like an inspirational slogan.
- Avoid cat puns, famous quotations, idioms, emojis, hashtags, quotation marks,
  and references to Mondays, nine lives, boxes, curiosity, or "if I fits".
- Return the quote without surrounding quotation marks.

Composition:
- One concrete sentence describing a single cat-centered visual scene.
- Make the quote's twist visible through the cat's pose, action, scale, and at
  most one simple prop. Prefer a bold silhouette over tiny details.
- The cat must dominate a landscape frame with a clear area along the bottom
  for the quote.
- Do not mention text, typography, colors, style, mood, or camera instructions.

Return only the requested JSON object."""

POSTER_PROMPT_TEMPLATE = """Minimalist brush-pen poster of a lone cat, composed for a 1152x704 landscape canvas.

Visual idea:
- {composition}

Graphic treatment:
- Bold, expressive black brush shapes with sparse white cutouts.
- EXACTLY four solid flat colors: pure yellow background, pure black and pure
  white cat, and pure red used only for one or two small accent marks.
- Flat silkscreen print: hard edges, high contrast, no gradients, shading,
  shadows, halftones, textures, transparency, or extra colors.
- Full bleed with no frame, border, panel, rectangle, watermark, signature, or
  decorative background elements.
- The cat is the unmistakable focal point and fills most of the frame.

Typography:
- Reserve a clean, unobstructed strip at the very bottom.
- In that strip render EXACTLY this text, once, on one line:
  "{quote}"
- Bold black monospaced uppercase lettering, centered and fully legible.
- Preserve every character and word exactly. No other letters or words.
- The lettering must not overlap the cat or touch the canvas edges.

Overall effect: immediate visual joke, sharp minimalism, confident handmade energy."""

CONCEPT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "poster_concept",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "quote": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 60,
                    "pattern": r"^[\x20-\x21\x23-\x7e]+$",
                },
                "composition": {"type": "string"},
            },
            "required": ["quote", "composition"],
            "additionalProperties": False,
        },
    },
}

SELECTION_SYSTEM_PROMPT = """You are a meticulous poster proofreader and art director.
Compare every candidate against the supplied quote and rubric. Inspect the
actual image rather than trusting its order."""

SELECTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
            "additionalProperties": False,
        },
    },
}


ASCII_PUNCTUATION_TRANSLATION = str.maketrans({
    "\N{LEFT SINGLE QUOTATION MARK}": "'",
    "\N{RIGHT SINGLE QUOTATION MARK}": "'",
    "\N{EN DASH}": "-",
    "\N{EM DASH}": "-",
    "\N{HORIZONTAL ELLIPSIS}": "...",
    "\N{NO-BREAK SPACE}": " ",
})

def _require_env(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _litellm_root_url() -> str:
    return LITELLM_BASE_URL.rstrip("/").removesuffix("/v1")


def _chat_url(provider: str) -> str:
    if provider not in {"openai", "openrouter"}:
        raise ValueError(f"Unsupported chat provider: {provider}")
    return f"{_litellm_root_url()}/{provider}/v1/chat/completions"


def _image_generation_url() -> str:
    return f"{_litellm_root_url()}/openai/v1/images/generations"


def _parse_json_object(text: str) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got: {text!r}")
    return parsed


def _chat(
    messages: list[dict[str, Any]],
    *,
    model: str,
    provider: str,
    response_format: dict[str, Any],
    temperature: float,
    max_completion_tokens: int,
    reasoning_effort: str | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    headers = {"Content-Type": "application/json"}
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"

    resp = requests.post(
        _chat_url(provider),
        headers=headers,
        json=payload,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LiteLLM error {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected LiteLLM response: {json.dumps(data)}") from e
    if not isinstance(content, str):
        raise ValueError(f"LiteLLM returned non-text content: {content!r}")
    return content


def _parse_concept(content: str) -> tuple[str, str]:
    parsed = _parse_json_object(content)
    quote = parsed["quote"]
    composition = parsed["composition"]
    if not isinstance(quote, str) or not isinstance(composition, str):
        raise ValueError("Quote and composition must be strings")
    quote = quote.translate(ASCII_PUNCTUATION_TRANSLATION).strip()
    composition = composition.strip()
    if not quote or not composition:
        raise ValueError("Quote and composition must not be empty")
    if not quote.isascii() or "\n" in quote or len(quote) > 60:
        raise ValueError(f"Quote violates the single-line ASCII limit: {quote!r}")
    if '"' in quote:
        raise ValueError(f"Quote must not contain quotation marks: {quote!r}")
    if "\n" in composition:
        raise ValueError(f"Composition must be one line: {composition!r}")
    return quote, composition


def _build_poster_prompt(quote: str, composition: str) -> str:
    return POSTER_PROMPT_TEMPLATE.format(quote=quote, composition=composition)


def _gen_prompt() -> tuple[str, str]:
    last_err: Exception | None = None
    for attempt in range(1, LLM_RETRIES + 1):
        logger.info("Generating quote and prompt (attempt %d/%d)...", attempt, LLM_RETRIES)
        try:
            content = _chat(
                [
                    {"role": "system", "content": CONCEPT_SYSTEM_PROMPT},
                    {"role": "user", "content": CONCEPT_INSTRUCTION},
                ],
                model=PROMPT_MODEL_ID,
                provider=PROMPT_PROVIDER,
                response_format=CONCEPT_RESPONSE_FORMAT,
                temperature=1.0,
                max_completion_tokens=500,
                reasoning_effort="minimal",
            )
            quote, composition = _parse_concept(content)
            logger.info("Generated quote: %s", quote)
            return quote, _build_poster_prompt(quote, composition)
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as e:
            last_err = e
            logger.warning("Attempt %d failed: %s", attempt, e)
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Failed to generate quote+prompt after {LLM_RETRIES} attempts") from last_err


def _image_data_url(encoded: str) -> str:
    if encoded.startswith("UklGR"):
        media_type = "image/webp"
    elif encoded.startswith("iVBOR"):
        media_type = "image/png"
    elif encoded.startswith("/9j/"):
        media_type = "image/jpeg"
    else:
        raise ValueError("Unknown image format returned by OpenAI")
    return f"data:{media_type};base64,{encoded}"


def _gen_image_sources(prompt: str, num_images: int = NUM_CANDIDATES) -> list[str]:
    payload = {
        "model": IMAGE_MODEL_ID,
        "prompt": prompt,
        "n": num_images,
        "size": "1536x1024",
        "quality": "medium",
        "background": "opaque",
        "output_format": "webp",
        "output_compression": 90,
    }
    logger.info(
        "Generating %d candidate images with %s...",
        num_images,
        IMAGE_MODEL_ID,
    )
    headers = {"Content-Type": "application/json"}
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"
    resp = requests.post(
        _image_generation_url(),
        headers=headers,
        json=payload,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI Images error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        items = data["data"]
        sources = [
            item["url"] if item.get("url") else _image_data_url(item["b64_json"])
            for item in items
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"Unexpected OpenAI Images response: {json.dumps(data)}") from e
    if len(sources) != num_images:
        raise RuntimeError(f"OpenAI returned {len(sources)} of {num_images} requested images")
    return sources


def _choose_image(quote: str, image_sources: list[str]) -> str:
    if len(image_sources) == 1:
        return image_sources[0]

    user_content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f'The intended quote is exactly: "{quote}"\n\n'
            "Choose the strongest finished poster. Rank these requirements in order:\n"
            "1. The intended quote appears exactly once, with identical spelling and "
            "punctuation, and no other text or text-like artifacts.\n"
            "2. The quote is fully legible on one unobstructed line at the bottom, "
            "not cropped and not overlapping the cat.\n"
            "3. The image uses only a yellow background, a black-and-white cat, and "
            "small red accents; it has no border, gradients, shading, or texture.\n"
            "4. The cat dominates the frame and the visual joke clearly supports the quote.\n"
            "Reject a pretty image when another candidate follows the typography more exactly. "
            f"Return the zero-based index from 0 through {len(image_sources) - 1}."
        ),
    }]
    for source in image_sources:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": source, "detail": "high"},
        })

    try:
        logger.info("Asking %s to pick the best image...", SELECTION_MODEL_ID)
        content = _chat(
            [
                {"role": "system", "content": SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            model=SELECTION_MODEL_ID,
            provider=SELECTION_PROVIDER,
            response_format=SELECTION_RESPONSE_FORMAT,
            temperature=0.0,
            max_completion_tokens=32,
        )
        idx = _parse_json_object(content)["index"]
        if type(idx) is not int or not 0 <= idx < len(image_sources):
            raise ValueError(f"Index {idx!r} out of range")
        logger.info("Model picked image #%d", idx)
        return image_sources[idx]
    except (requests.RequestException, RuntimeError, ValueError, KeyError) as e:
        logger.warning("Image selection failed (%s); falling back to first image", e)
        return image_sources[0]


def _download_file(source: str, dest: str | Path, chunk_size: int = 8192) -> None:
    if source.startswith("data:"):
        _, encoded = source.split(",", 1)
        Path(dest).write_bytes(base64.b64decode(encoded, validate=True))
        return

    with requests.get(source, stream=True, timeout=DEFAULT_TIMEOUT_SEC) as r:
        r.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    out.write(chunk)


def _rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc

    s = np.zeros_like(maxc)
    nz = maxc != 0
    s[nz] = delta[nz] / maxc[nz]

    h = np.zeros_like(maxc)
    mask = delta != 0
    idx = (maxc == r) & mask
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = (maxc == g) & mask
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
    idx = (maxc == b) & mask
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4
    return np.stack([h / 6.0, s, v], axis=-1)


# 0=black, 1=white, 2=red, 3=yellow
PALETTE = np.array([
    [0,   0,   0],
    [255, 255, 255],
    [255, 0,   0],
    [255, 255, 0],
], dtype=np.uint8)


def _quantize_to_palette(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    hsv = _rgb_to_hsv_np(arr)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    nearest = np.full((h, w), -1, dtype=np.int32)
    nearest[(S < 0.25) & (V > 0.75)] = 1  # white
    nearest[(nearest == -1) & (H >= 35/360.0) & (H <= 75/360.0) & (S > 0.3) & (V > 0.25)] = 3  # yellow

    unassigned = nearest == -1
    if unassigned.any():
        pixels = arr[unassigned]
        palette_f = PALETTE.astype(np.float32) / 255.0
        dist = np.sum((pixels[:, None, :] - palette_f[None, :, :]) ** 2, axis=-1)
        nearest[unassigned] = np.argmin(dist, axis=1)

    return Image.fromarray(PALETTE[nearest])


def _fit_to_canvas(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, (255, 255, 0))  # yellow background
    x = (target_size[0] - img.width) // 2
    y = (target_size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def process_image(input_path: str | Path, output_path: str | Path) -> None:
    img = Image.open(input_path).convert("RGB")
    canvas = _fit_to_canvas(img, TARGET_SIZE)
    canvas = _quantize_to_palette(canvas)
    canvas.save(output_path, format="BMP")


def _check_s3_env() -> None:
    if not S3_BUCKET:
        return
    _require_env(S3_REGION, "S3_REGION")
    _require_env(S3_ENDPOINT_URL, "S3_ENDPOINT_URL")
    _require_env(S3_ACCESS_KEY_ID, "S3_ACCESS_KEY_ID")
    _require_env(S3_SECRET_ACCESS_KEY, "S3_SECRET_ACCESS_KEY")


def upload_file_to_s3(local_path: str | Path, bucket: str, key: str, content_type: str) -> str:
    session = boto3.session.Session()
    client = session.client(
        service_name="s3",
        region_name=S3_REGION,
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    )
    client.upload_file(str(local_path), bucket, key, ExtraArgs={"ContentType": content_type})
    return f"s3://{bucket}/{key}"


def main() -> None:
    _check_s3_env()

    quote, image_prompt = _gen_prompt()
    image_sources = _gen_image_sources(image_prompt)
    image_source = _choose_image(quote, image_sources)

    with tempfile.TemporaryDirectory(prefix="cat-of-the-day-") as tmpdir:
        original_path = Path(tmpdir) / "original.webp"
        poster_path = Path(tmpdir) / "poster.bmp"

        logger.info("Saving chosen image...")
        _download_file(image_source, original_path)

        logger.info("Postprocessing into 4-color BMP...")
        process_image(original_path, poster_path)
        logger.info("Poster saved to %s", poster_path)

        if S3_BUCKET:
            url = upload_file_to_s3(poster_path, S3_BUCKET, "poster.bmp", content_type="image/bmp")
            logger.info("Uploaded poster: %s", url)
            orig_url = upload_file_to_s3(
                original_path,
                S3_BUCKET,
                "original_poster.webp",
                content_type="image/webp",
            )
            logger.info("Uploaded original: %s", orig_url)
        else:
            sys.stdout.buffer.write(poster_path.read_bytes())
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
