import requests
import json
import os
import sys
import logging
import base64
import colorsys
from pathlib import Path
from typing import  Tuple, List

from PIL import Image
import numpy as np

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

AI_PROXY_TOKEN = os.environ.get("AI_PROXY_TOKEN")
AI_PROXY_BASE_URL = os.environ.get("AI_PROXY_BASE_URL")

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("S3_REGION")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")

def _proxy_base_url() -> str:
    base = _require_env(AI_PROXY_BASE_URL, "AI_PROXY_BASE_URL")
    return base.rstrip("/")


def _chat_url() -> str:
    return f"{_proxy_base_url()}/openai/v1/chat/completions"


def _ideogram_url() -> str:
    return f"{_proxy_base_url()}/ideogram/v1/ideogram-v3/generate"

DEFAULT_TIMEOUT_SEC = 600
TARGET_SIZE = (800, 480)

CHATGPT_SYSTEM_PROMPT = """
YOU ARE A MULTI-AWARD-WINNING ART DIRECTOR AND POSTER DESIGN VISIONARY, CELEBRATED FOR YOUR WORK IN FILM, BRANDING, SOCIAL IMPACT CAMPAIGNS, AND EVENTS.
YOU HAVE LED CREATIVE TEAMS AT TOP AGENCIES AND YOUR POSTERS ARE EXHIBITED IN MOMA AND FEATURED ON DESIGNBOOM, IT’S NICE THAT, AND BEHANCE.
YOUR TASK IS TO CREATE STRIKING, ORIGINAL POSTER CONCEPTS THAT IMMEDIATELY GRAB ATTENTION, COMMUNICATE A POWERFUL MESSAGE, AND ALIGN PERFECTLY WITH THE CLIENT’S OBJECTIVE.
"""

CHATGPT_PROMPT = """
You must generate **one new English quote** in the style of the examples (humor, irony, slightly philosophical, minimalistic). The quote must contain a surprising twist or reversal of expectation.
Then, adapt the **poster prompt** so that the composition the mood of the new quote. The cat must remain black and white.
Replace `<QUOTE OF THE DAY>` in the poster prompt with the generated quote.

**Output format:**
Return strictly in JSON:

```
{"quote": "<quote>", "prompt": "<poster prompt with the quote inserted>"}
```

---

### Quote rules:
- English only.
- 1–2 sentences maximum.
- Humor, irony, minimalism.
- Must contain a small unexpected twist or contrast.
- No emojis, smilies, or extra symbols.

---

### Poster-prompt rules:
Base style:

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
- Black and White cat.

Composition rules:
- <COMPOSITION>

Scale and framing:
- The cat must be large, dominating the composition and filling most of the frame.
- The cat must be black and white.
- Composition must use full bleed, edge-to-edge design, with minimal unused background space.

Typography requirement:
- At the very bottom of the composition, include EXACTLY the following quote as bold monospaced lettering:
“<QUOTE OF THE DAY>”
- The text must appear as part of the artwork (not metadata), spanning the full width as a clean strip, horizontally aligned, and clearly visible.

Atmosphere: sharp minimalism, bold irony, humor
```

Your task:
  - First, come up with a new quote (in English).
  - Then adapt the composition to match it.
  - Return JSON with two fields: "quote" and "prompt".
"""

PALETTE = np.array([
    [0, 0, 0],        # black
    [255, 255, 255],  # white
    [255, 255, 0],    # yellow
    [255, 0, 0],      # red
], dtype=np.uint8)


def quantize_to_palette(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape

    # Output initialized to white
    result = np.full_like(arr, (255, 255, 255))

    # Normalize to [0,1] for HSV conversion
    arr_norm = arr / 255.0

    # Compute HSV for all pixels
    hsv = np.zeros_like(arr_norm)
    for y in range(h):
        for x in range(w):
            r, g, b = arr_norm[y, x]
            hsv[y, x] = colorsys.rgb_to_hsv(r, g, b)  # (hue 0-1, sat 0-1, val 0-1)

    H = hsv[:, :, 0] * 360  # hue in degrees
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # --- White: low saturation + high brightness ---
    mask_white = (S < 0.2) & (V > 0.85)
    result[mask_white] = (255, 255, 255)

    # --- Black: low brightness ---
    mask_black = V < 0.25
    result[mask_black] = (0, 0, 0)

    # --- Red: hue near 0° or 360°, strong saturation ---
    mask_red = ((H < 20) | (H > 340)) & (S > 0.5) & (V > 0.25)
    result[mask_red] = (255, 0, 0)

    # --- Yellow: hue ~40–65°, strong saturation ---
    mask_yellow = (H >= 40) & (H <= 65) & (S > 0.5) & (V > 0.25)
    result[mask_yellow] = (255, 255, 0)

    return Image.fromarray(result)


def fill_borders(canvas: Image.Image, img: Image.Image) -> None:
    target_w, target_h = canvas.size
    x_offset = (target_w - img.width)//2
    y_offset = (target_h - img.height)//2

    # Paste top/bottom rows
    for x in range(x_offset, x_offset+img.width):
        top_pixel = img.getpixel((x-x_offset,0))
        bottom_pixel = img.getpixel((x-x_offset,img.height-1))
        for y in range(y_offset):
            canvas.putpixel((x,y), top_pixel)
        for y in range(y_offset+img.height, target_h):
            canvas.putpixel((x,y), bottom_pixel)

    # Paste left/right columns
    for y in range(target_h):
        left_pixel = canvas.getpixel((x_offset, y))
        right_pixel = canvas.getpixel((x_offset+img.width-1, y))
        for x in range(x_offset):
            canvas.putpixel((x, y), left_pixel)
        for x in range(x_offset+img.width, target_w):
            canvas.putpixel((x, y), right_pixel)


def upload_file_to_s3(local_path: str, bucket: str, key: str, content_type: str = "image/bmp") -> str:
    session = boto3.session.Session()
    s3_client = session.client(
        service_name="s3",
        region_name=_require_env(S3_REGION, "S3_REGION"),
        endpoint_url=_require_env(S3_ENDPOINT_URL, "S3_ENDPOINT_URL"),
        aws_access_key_id=_require_env(S3_ACCESS_KEY_ID, "S3_ACCESS_KEY_ID"),
        aws_secret_access_key=_require_env(S3_SECRET_ACCESS_KEY, "S3_SECRET_ACCESS_KEY"),
    )

    try:
        extra_args = {"ContentType": content_type}
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
    except Exception as e:
        logger.error("Failed to upload %s to s3://%s/%s: %s", local_path, bucket, key, e)
        raise

    return f"s3://{bucket}/{key}"


def process_image(input_path: str | Path, output_path_bmp: str | Path) -> None:
    img = Image.open(input_path).convert("RGB")

    target_size = TARGET_SIZE
    img.thumbnail(target_size, Image.LANCZOS)

    canvas = Image.new("RGB", target_size, (255, 255, 0))

    x = (target_size[0]-img.width)//2
    y = (target_size[1]-img.height)//2
    canvas.paste(img, (x,y))

    canvas = quantize_to_palette(canvas)

    # Not strictly needed, but preserves clean borders if any empty fill remains
    fill_borders(canvas, img)

    canvas.save(output_path_bmp, format="BMP")


def _require_env(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _image_uri_to_base64(url: str) -> str:
    logger.info("Downloading image: %s", url)
    with requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT_SEC) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        data = bytearray()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                data.extend(chunk)

    encoded = base64.b64encode(bytes(data)).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _download_file(url: str, dest_path: str | Path, chunk_size: int = 8192) -> None:
    with requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT_SEC) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as out_file:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    out_file.write(chunk)


def _gen_prompt() -> Tuple[str, str]:
    chat_payload = {
        "model": "gpt-5",
        "messages": [
            {
                "role": "system",
                "content": CHATGPT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": CHATGPT_PROMPT,
            },
        ],
    }

    logger.info("Sending prompt to Chat endpoint...")
    token = _require_env(AI_PROXY_TOKEN, "AI_PROXY_TOKEN")
    chat_resp = requests.post(
        _chat_url(),
        headers={
            "Authorization": f"OAuth {token}",
            "Content-Type": "application/json",
        },
        json=chat_payload,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    chat_resp.raise_for_status()
    chat_data = chat_resp.json()

    try:
        logger.info("Parsing Chat response JSON...")
        message_content = chat_data["response"]["choices"][0]["message"]["content"].strip()
        parsed = json.loads(message_content)
        quote = parsed["quote"]
        prompt = parsed["prompt"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise ValueError("Chat output parsing failed: " + json.dumps(chat_data)) from e

    logger.info("Generated quote: %s", quote)
    return quote, prompt


def _choose_image(quote: str, image_urls: List[str]) -> str:
    chat_payload = {
        "model": "gpt-5",
        "messages": [
            {
                "role": "system",
                "content": CHATGPT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":  [
                    {
                        "type": "text",
                        "text": (
                            f"Your task is to select the single best image from the list. The chosen image should be the clearest and best represent the quote: {quote}."
                            "  - Prefer images that look great in only for color (black, white, red, yellow)."
                            "  - Prefer images that include the quote text, but only if the text is clear and accurately matches the quote."
                            "  - If no image with the quote meets these conditions, choose the clearest image that visually represents the quote instead."
                            "  - Return only the number of the selected image, with no additional text or explanation."
                        )
                    }
                ]
            }
        ],
    }

    for image_url in image_urls:
        chat_payload["messages"][1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": _image_uri_to_base64(image_url)
            }
        })

    logger.info("Sending prompt to Chat endpoint...")
    token = _require_env(AI_PROXY_TOKEN, "AI_PROXY_TOKEN")
    chat_resp = requests.post(
        _chat_url(),
        headers={
            "Authorization": f"OAuth {token}",
            "Content-Type": "application/json",
        },
        json=chat_payload,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    chat_resp.raise_for_status()
    chat_data = chat_resp.json()

    try:
        logger.info("Parsing Chat response JSON...")
        image_number = chat_data["response"]["choices"][0]["message"]["content"].strip()
        index = int(image_number)
        return image_urls[index]
    except Exception as e:
        logger.error("Chat output parsing failed: %s", e)
        return image_urls[0]


def _gen_images_urls(prompt: str) -> List[str]:
    form = {
        "prompt": (None, prompt),
        "resolution": (None, "1152x704"),
        "rendering_speed": (None, "DEFAULT"),
        "num_images": (None, "3"),
    }

    logger.info("Sending prompt to Ideogram endpoint...")
    token = _require_env(AI_PROXY_TOKEN, "AI_PROXY_TOKEN")
    resp = requests.post(
        _ideogram_url(),
        headers={
            "Authorization": f"OAuth {token}",
        },
        files=form,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    ideogram_data = resp.json()

    try:
        return [ideogram_data["response"]["data"][i]["url"] for i in range(3)]
    except (KeyError, IndexError) as e:
        logger.error("Could not extract image URL: %s", e)
        raise


def main() -> None:
    quote, ideogram_prompt = _gen_prompt()
    image_urls = _gen_images_urls(ideogram_prompt)

    logger.info("Choose best image...")
    image_url = _choose_image(quote, image_urls)

    original_poster_path = "/tmp/original_poster.png"
    logger.info("Downloading generated image: %s -> %s", image_url, original_poster_path)
    _download_file(image_url, original_poster_path)

    poster_path = "/tmp/poster.bmp"
    logger.info("Postprocess image from: %s -> %s", original_poster_path, poster_path)
    process_image(original_poster_path, poster_path)
    
    logger.info("Poster saved %s", poster_path)

    if S3_BUCKET:
        s3_url = upload_file_to_s3(poster_path, S3_BUCKET, "poster.bmp", content_type="image/bmp")
        logger.info("Uploaded final image to %s", s3_url)

        orig_s3_url = upload_file_to_s3(original_poster_path, S3_BUCKET, "original_poster.png", content_type="image/png")
        logger.info("Uploaded original image to %s", orig_s3_url)
    else:
        with open(poster_path, "rb") as f:
            sys.stdout.buffer.write(f.read())
            sys.stdout.buffer.flush()

if __name__ == "__main__":
    main()
