#!/usr/bin/env python3
"""
ocr_paper.py — run Mistral OCR on every task paper.pdf (tasks/paper{i}/paper.pdf).

For each <root>/paperK/ directory containing paper.pdf, this script writes:

    paperK/
        paper.md                     # OCR'd Markdown (image refs rewritten)
        images/
            image1.{png,jpg,...}     # extracted images, sequentially numbered
            image2.{png,jpg,...}
            ...

Image references inside the Markdown are rewritten so that the original
Mistral image IDs (e.g. ``img-1.jpeg``) are replaced with the new
``images/imageN.<ext>`` paths.

Usage:
    export MISTRAL_API_KEY=...
    python ocr_paper.py                        # default: ./tasks
    python ocr_paper.py path/to/tasks
    python ocr_paper.py --only 1 5 12          # restrict to paper indices
    python ocr_paper.py --overwrite            # re-OCR papers that already have paper.md
    python ocr_paper.py --list                 # show plan, do not call API

Requirements:
    pip install mistralai
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None  # imported lazily; main() validates before using.

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

DEFAULT_INPUT_DIR = str(Path(__file__).resolve().parent / "tasks")
MODEL = "mistral-ocr-latest"
TOTAL_PAPERS = 15
RETRY_COUNT = 3
RETRY_BACKOFF_SECONDS = 5.0


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def encode_pdf_base64(pdf_path: Path) -> str:
    return base64.b64encode(pdf_path.read_bytes()).decode("utf-8")


def detect_image_extension(img_bytes: bytes) -> str:
    """Detect image format from magic bytes; default to .png."""
    if len(img_bytes) >= 4 and img_bytes[:4] == b"\x89PNG":
        return ".png"
    if len(img_bytes) >= 3 and img_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(img_bytes) >= 6 and img_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if (len(img_bytes) >= 12
            and img_bytes[:4] == b"RIFF"
            and img_bytes[8:12] == b"WEBP"):
        return ".webp"
    if len(img_bytes) >= 4 and img_bytes[:4] == b"%PDF":
        return ".pdf"  # rare embedded PDF page
    return ".png"


def strip_data_uri_prefix(b64_or_data_uri: str) -> str:
    """Mistral may return either raw base64 or a 'data:image/...;base64,...' URI."""
    if b64_or_data_uri.startswith("data:") and "," in b64_or_data_uri:
        return b64_or_data_uri.split(",", 1)[1]
    return b64_or_data_uri


# --------------------------------------------------------------------------- #
# OCR for one paper                                                            #
# --------------------------------------------------------------------------- #

def ocr_one_paper(
    client: Mistral,
    paper_dir: Path,
    paper_idx: int,
    overwrite: bool,
) -> Tuple[str, str]:
    """OCR a single paper directory.

    Returns (status, message), where status is one of
    'ok', 'skipped', 'missing-pdf', 'missing-dir', 'error'.
    """
    pdf_path = paper_dir / "paper.pdf"
    md_path = paper_dir / "paper.md"
    images_dir = paper_dir / "images"

    if not paper_dir.exists():
        return "missing-dir", f"{paper_dir} does not exist"
    if not pdf_path.exists():
        return "missing-pdf", f"{pdf_path} does not exist (download it first)"

    if md_path.exists() and not overwrite:
        return "skipped", f"{md_path} already exists (use --overwrite to redo)"

    pdf_size_kib = pdf_path.stat().st_size / 1024
    print(f"  encoding {pdf_path.name} ({pdf_size_kib:.1f} KiB)")
    base64_pdf = encode_pdf_base64(pdf_path)
    document = {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}",
    }

    last_error: Optional[Exception] = None
    response = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            print(f"  calling Mistral OCR (attempt {attempt}/{RETRY_COUNT})...")
            response = client.ocr.process(
                model=MODEL,
                document=document,
                include_image_base64=True,
            )
            break
        except Exception as e:  # broad: SDK exceptions vary by version
            last_error = e
            if attempt < RETRY_COUNT:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(f"  attempt {attempt} failed: {e}; retrying in {wait:.0f}s")
                time.sleep(wait)
    if response is None:
        return "error", f"Mistral OCR failed after {RETRY_COUNT} attempts: {last_error}"

    pages = list(response.pages)
    print(f"  OCR returned {len(pages)} pages")

    # ---- collect images and assign sequential names ----
    images_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any old image files when overwriting, to avoid stale leftovers
    if overwrite:
        for existing in images_dir.glob("image*"):
            try:
                existing.unlink()
            except OSError:
                pass

    id_to_relpath: Dict[str, str] = {}  # original Mistral id -> "images/imageN.ext"
    image_counter = 0

    for page in pages:
        page_images = getattr(page, "images", None) or []
        for img in page_images:
            if img.id in id_to_relpath:
                continue  # de-duplicate across pages if any
            image_counter += 1

            written = False
            ext = ".png"
            if getattr(img, "image_base64", None):
                try:
                    raw_b64 = strip_data_uri_prefix(img.image_base64)
                    img_bytes = base64.b64decode(raw_b64)
                    ext = detect_image_extension(img_bytes)
                    new_name = f"image{image_counter}{ext}"
                    (images_dir / new_name).write_bytes(img_bytes)
                    id_to_relpath[img.id] = f"images/{new_name}"
                    written = True
                except Exception as e:
                    print(f"    WARN: could not decode {img.id}: {e}")

            if not written:
                # No bytes available, but we still register a placeholder name
                new_name = f"image{image_counter}{ext}"
                id_to_relpath[img.id] = f"images/{new_name}"

    print(f"  wrote {image_counter} image(s) to {images_dir}")

    # ---- assemble markdown and rewrite image references ----
    md_parts: List[str] = []
    for page in pages:
        page_md = page.markdown or ""
        # Mistral typically embeds images as ![<id>](<id>) and may use the same id
        # in multiple syntactic positions. Replace all occurrences.
        # Sort by descending id length so that longer ids are replaced first
        # (avoids partial-prefix collisions).
        for orig_id in sorted(id_to_relpath.keys(), key=len, reverse=True):
            new_rel = id_to_relpath[orig_id]
            page_md = page_md.replace(orig_id, new_rel)
        md_parts.append(page_md)

    full_md = "\n\n".join(md_parts)
    md_path.write_text(full_md, encoding="utf-8")
    print(f"  wrote {md_path} ({len(full_md):,} chars)")

    return "ok", f"{len(pages)} pages, {image_counter} images"


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Mistral OCR on every task paper.pdf.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    parser.add_argument("input_dir", nargs="?", default=DEFAULT_INPUT_DIR,
                        help=f"Root directory containing paperK/ subdirs (default: {DEFAULT_INPUT_DIR}).")
    parser.add_argument("--only", type=int, nargs="+", metavar="N",
                        help="Restrict to specific paper indices (1..20).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-OCR papers that already have paper.md.")
    parser.add_argument("--list", action="store_true",
                        help="Print the planned actions but don't call the API.")
    parser.add_argument("--api-key", default=None,
                        help="Mistral API key (default: $MISTRAL_API_KEY).")
    parser.add_argument("--base-url", default=None,
                        help="Custom Mistral server/base URL, e.g. an internal gateway "
                             "or proxy (default: $MISTRAL_BASE_URL). Passed to the SDK as "
                             "server_url.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    base_url = args.base_url or os.environ.get("MISTRAL_BASE_URL")
    if not args.list and not api_key:
        print("ERROR: set MISTRAL_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist (expected the repository's tasks/ directory).", file=sys.stderr)
        return 2

    if args.only:
        bad = [i for i in args.only if not (1 <= i <= TOTAL_PAPERS)]
        if bad:
            print(f"ERROR: paper indices out of range (must be 1..{TOTAL_PAPERS}): {bad}",
                  file=sys.stderr)
            return 2
        indices = sorted(set(args.only))
    else:
        indices = list(range(1, TOTAL_PAPERS + 1))

    # Dry-run mode
    if args.list:
        print(f"Plan ({len(indices)} paper(s)):")
        for i in indices:
            paper_dir = root / f"paper{i}"
            pdf = paper_dir / "paper.pdf"
            md = paper_dir / "paper.md"
            if not paper_dir.exists():
                tag = "MISSING-DIR"
            elif not pdf.exists():
                tag = "MISSING-PDF (download first)"
            elif md.exists() and not args.overwrite:
                tag = "SKIP (paper.md exists)"
            else:
                tag = "WILL OCR"
            print(f"  paper{i:>2}: {tag}")
        return 0

    if Mistral is None:
        print("ERROR: mistralai is not installed. Run: pip install mistralai",
              file=sys.stderr)
        return 2
    client = Mistral(api_key=api_key, server_url=base_url) if base_url else Mistral(api_key=api_key)
    if base_url:
        print(f"[ocr] using custom server_url: {base_url}", file=sys.stderr)

    results: List[Tuple[int, str, str]] = []
    for i in indices:
        paper_dir = root / f"paper{i}"
        print(f"\n[paper{i}] {paper_dir}")
        try:
            status, msg = ocr_one_paper(client, paper_dir, i, args.overwrite)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception:
            traceback.print_exc()
            status, msg = "error", "unhandled exception (see traceback above)"
        results.append((i, status, msg))
        print(f"  -> {status}: {msg}")

    # ---- summary ----
    print("\n" + "=" * 72)
    print(f"{'paper':>5}  {'status':<14}  detail")
    print("-" * 72)
    fail_count = 0
    for idx, status, msg in results:
        if status in ("error", "missing-pdf", "missing-dir"):
            fail_count += 1
        print(f"paper{idx:>2}  {status:<14}  {msg}")
    print("=" * 72)
    print(f"Processed: {len(results)}; OK or skipped: {len(results)-fail_count}; failed: {fail_count}")
    if fail_count:
        print("Tip: every tasks/paper{i}/ ships its paper.pdf; a missing-pdf entry means the file was removed.")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())