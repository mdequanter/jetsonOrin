#!/usr/bin/env python3
"""
Create a QR code image and HTML page for testing nl_launchv2.py.

The QR content is entered in the terminal by default.
"""

import argparse
import html
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import qrcode
    from qrcode.constants import ERROR_CORRECT_H
except ImportError as e:
    missing = getattr(e, "name", None) or str(e)
    print(f"[ERROR] Missing Python package: {missing}", file=sys.stderr)
    print("[INFO] Install with: pip install qrcode[pil] opencv-python", file=sys.stderr)
    raise SystemExit(1)


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())[:40].strip("_")
    return cleaned or "qr"


def read_content(args) -> str:
    if args.text is not None:
        return args.text.strip()

    content = input("QR content: ").strip()
    return content


def build_qr(content: str, png_path: Path, box_size: int, border: int) -> None:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=box_size,
        border=border,
    )
    qr.add_data(content)
    qr.make(fit=True)

    image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    image.save(png_path)


def build_html_page(content: str, png_name: str, html_path: Path) -> None:
    escaped_content = html.escape(content)
    escaped_png_name = html.escape(png_name, quote=True)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FaceAssist QR Code</title>
  <style>
    html, body {{
      margin: 0;
      min-height: 100%;
      background: #f4f4f4;
      color: #111;
      font-family: Arial, sans-serif;
    }}
    body {{
      display: grid;
      place-items: center;
      padding: 32px;
      box-sizing: border-box;
    }}
    main {{
      width: min(720px, 100%);
      text-align: center;
    }}
    img {{
      width: min(80vmin, 560px);
      height: auto;
      image-rendering: pixelated;
      background: #fff;
      border: 24px solid #fff;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }}
    pre {{
      margin: 24px auto 0;
      padding: 16px;
      max-width: 640px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      text-align: left;
      background: #fff;
      border: 1px solid #ddd;
    }}
  </style>
</head>
<body>
  <main>
    <img src="{escaped_png_name}" alt="QR code">
    <pre>{escaped_content}</pre>
  </main>
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")


def verify_with_opencv(png_path: Path, expected: str) -> bool:
    image = cv2.imread(str(png_path))
    if image is None:
        return False

    detector = cv2.QRCodeDetector()
    decoded, _, _ = detector.detectAndDecode(image)
    return decoded == expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a QR code image and HTML page.")
    parser.add_argument("--text", type=str, default=None, help="QR content. If omitted, prompt in the terminal.")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "qr_codes"))
    parser.add_argument("--name", type=str, default="", help="Output base filename without extension.")
    parser.add_argument("--box_size", type=int, default=14, help="Pixel size of each QR module.")
    parser.add_argument("--border", type=int, default=4, help="QR quiet-zone border in modules.")
    args = parser.parse_args()

    content = read_content(args)
    if not content:
        print("[ERROR] QR content is empty.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = sanitize_filename(args.name or content)
    png_path = out_dir / f"{stamp}_{base_name}.png"
    html_path = out_dir / f"{stamp}_{base_name}.html"

    build_qr(
        content=content,
        png_path=png_path,
        box_size=max(2, int(args.box_size)),
        border=max(4, int(args.border)),
    )
    build_html_page(content=content, png_name=png_path.name, html_path=html_path)

    print(f"[OK] QR image: {png_path}")
    print(f"[OK] HTML page: {html_path}")

    if verify_with_opencv(png_path, content):
        print("[OK] OpenCV can detect this QR code.")
    else:
        print("[WARN] OpenCV did not decode the saved PNG directly. Try increasing --box_size.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
