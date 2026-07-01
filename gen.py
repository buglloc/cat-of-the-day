import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import boto3
import numpy as np
import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

AI_PROXY_TOKEN = os.environ.get("AI_PROXY_TOKEN")
AI_PROXY_BASE_URL = os.environ.get("AI_PROXY_BASE_URL")
MODEL_ID = os.environ.get("MODEL_ID", "google/gemini-2.5-pro-preview-03-25")

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("S3_REGION")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")

DEFAULT_TIMEOUT_SEC = 600
TARGET_SIZE = (800, 480)
LLM_RETRIES = 3
NUM_CANDIDATES = 3

SYSTEM_PROMPT = (
    "You are an award-winning art director and poster designer. "
    "You create striking, original poster concepts that grab attention, "
    "communicate a clear message, and match the brief precisely."
)

QUOTE_AND_PROMPT_INSTRUCTION = """Generate one new English quote in the spirit of minimalist, ironic, slightly philosophical sayings, then adapt the poster prompt to match the quote's mood.

Quote rules:
- English only.
- 1 sentence, maximum 2.
- Must fit on a single line of monospaced typography at the bottom of an 800x480 poster — keep it under 60 characters total.
- Humor, irony, minimalism, with a small unexpected twist or reversal.
- Use straight ASCII quotes only inside the JSON value. No emojis, smilies, or extra symbols.

Poster prompt rules:
Use this base template. Substitute `<COMPOSITION>` with a single concrete sentence describing the visual scene that matches the quote, and `<QUOTE OF THE DAY>` with the generated quote.

```
Minimalist wireframe brush-pen style poster of a lone cat.
Clean, bold silkscreen aesthetic with EXACTLY four solid flat colors:
- pure yellow (background)
- pure red (small accents)
- pure black (silhouettes)
- pure white

Strict rules:
- Background must be entirely pure yellow.
- Absolutely NO frames, borders, rectangles, textures, or extra layers.
- Red is ONLY for small accent scratches, details, or marks.
- No gradients, shading, halftone patterns, outlines, or extra colors.
- Must look like a flat silkscreen print or bold brush-pen artwork.
- The cat is black and white only.

Composition:
- <COMPOSITION>

Scale and framing:
- The cat dominates the composition and fills most of the frame.
- Full bleed, edge-to-edge design, with minimal unused background.

Typography:
- At the very bottom of the composition, render EXACTLY this quote as bold monospaced lettering on a single line:
"<QUOTE OF THE DAY>"
- The text is part of the artwork (not metadata), spans the full width as a clean strip, horizontally aligned, clearly visible, and never overlaps the cat.

Atmosphere: sharp minimalism, bold irony, humor.
```

Output:
Return ONLY a raw JSON object, no markdown fences, no commentary:
{"quote": "<quote>", "prompt": "<full poster prompt with substitutions applied>"}
"""

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _require_env(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _proxy_base_url() -> str:
    return _require_env(AI_PROXY_BASE_URL, "AI_PROXY_BASE_URL").rstrip("/")


def _chat_url() -> str:
    return f"{_proxy_base_url()}/openrouter/v1/chat/completions"


def _ideogram_url() -> str:
    return f"{_proxy_base_url()}/ideogram/v1/ideogram-v3/generate"


def _parse_json_object(text: str) -> dict:
    match = JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in response: {text!r}")
    return json.loads(match.group(0))


def _chat(messages: list[dict], model: str | None = None) -> str:
    payload = {"model": model or MODEL_ID, "messages": messages}
    token = _require_env(AI_PROXY_TOKEN, "AI_PROXY_TOKEN")
    resp = requests.post(
        _chat_url(),
        headers={
            "Authorization": f"OAuth {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Chat API error {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        return data["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected chat response shape: {json.dumps(data)}") from e


def _gen_prompt() -> tuple[str, str]:
    last_err: Exception | None = None
    for attempt in range(1, LLM_RETRIES + 1):
        logger.info("Generating quote and prompt (attempt %d/%d)...", attempt, LLM_RETRIES)
        try:
            content = _chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": QUOTE_AND_PROMPT_INSTRUCTION},
            ])
            parsed = _parse_json_object(content)
            quote = parsed["quote"].strip()
            prompt = parsed["prompt"].strip()
            logger.info("Generated quote: %s", quote)
            return quote, prompt
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            logger.warning("Attempt %d failed: %s", attempt, e)
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Failed to generate quote+prompt after {LLM_RETRIES} attempts") from last_err


def _gen_image_urls(prompt: str, num_images: int = NUM_CANDIDATES) -> list[str]:
    form = {
        "prompt": (None, prompt),
        "resolution": (None, "1152x704"),
        "rendering_speed": (None, "DEFAULT"),
        "num_images": (None, str(num_images)),
    }
    logger.info("Generating %d candidate images via Ideogram...", num_images)
    token = _require_env(AI_PROXY_TOKEN, "AI_PROXY_TOKEN")
    resp = requests.post(
        _ideogram_url(),
        headers={"Authorization": f"OAuth {token}"},
        files=form,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return [item["url"] for item in data["response"]["data"]]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Unexpected Ideogram response: {json.dumps(data)}") from e


def _choose_image(quote: str, image_urls: list[str]) -> str:
    if len(image_urls) == 1:
        return image_urls[0]

    valid_indices = ", ".join(str(i) for i in range(len(image_urls)))
    user_content: list[dict] = [{
        "type": "text",
        "text": (
            f'Select the single best image for this quote: "{quote}".\n'
            "Prefer images where the quote text is rendered clearly and accurately, "
            "with no spelling errors or text artifacts. Otherwise pick the image that "
            "most clearly represents the quote visually.\n"
            f"Respond with ONLY a single digit from this set: {valid_indices} "
            "(zero-indexed, in the order the images were given). No words, no punctuation."
        ),
    }]
    for url in image_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    try:
        logger.info("Asking model to pick the best image...")
        content = _chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]).strip()
        match = re.search(r"\d+", content)
        if not match:
            raise ValueError(f"No digit in response: {content!r}")
        idx = int(match.group(0))
        if not 0 <= idx < len(image_urls):
            raise ValueError(f"Index {idx} out of range")
        logger.info("Model picked image #%d", idx)
        return image_urls[idx]
    except Exception as e:
        logger.warning("Image selection failed (%s); falling back to first image", e)
        return image_urls[0]


def _download_file(url: str, dest: str | Path, chunk_size: int = 8192) -> None:
    with requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT_SEC) as r:
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

    quote, ideogram_prompt = _gen_prompt()
    image_urls = _gen_image_urls(ideogram_prompt)
    image_url = _choose_image(quote, image_urls)

    with tempfile.TemporaryDirectory(prefix="cat-of-the-day-") as tmpdir:
        original_path = Path(tmpdir) / "original.png"
        poster_path = Path(tmpdir) / "poster.bmp"

        logger.info("Downloading chosen image: %s", image_url)
        _download_file(image_url, original_path)

        logger.info("Postprocessing into 4-color BMP...")
        process_image(original_path, poster_path)
        logger.info("Poster saved to %s", poster_path)

        if S3_BUCKET:
            url = upload_file_to_s3(poster_path, S3_BUCKET, "poster.bmp", content_type="image/bmp")
            logger.info("Uploaded poster: %s", url)
            orig_url = upload_file_to_s3(original_path, S3_BUCKET, "original_poster.png", content_type="image/png")
            logger.info("Uploaded original: %s", orig_url)
        else:
            sys.stdout.buffer.write(poster_path.read_bytes())
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
