"""
Fix images with incorrect EXIF orientation by applying exif_transpose in-place.
Usage: python fix_rotated_images.py <folder> [--recursive] [--dry-run]
"""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

EXIF_ORIENTATION_TAG = 274  # Tag ID for 'Orientation'
SUPPORTED_EXTS = {'.jpg', '.jpeg'}


def get_orientation(img: Image.Image) -> int:
    try:
        exif = img._getexif()
        if exif:
            return exif.get(EXIF_ORIENTATION_TAG, 1)
    except Exception:
        pass
    return 1


def fix_image(path: Path, dry_run: bool) -> bool:
    """Returns True if image was (or would be) fixed."""
    try:
        img = Image.open(path)
        orientation = get_orientation(img)
        if orientation == 1:
            return False

        tag_name = TAGS.get(EXIF_ORIENTATION_TAG, 'Orientation')
        print(f"  [FIX] {path}  ({tag_name}={orientation}  {img.size[0]}x{img.size[1]}", end="")

        if not dry_run:
            fixed = ImageOps.exif_transpose(img)
            print(f" -> {fixed.size[0]}x{fixed.size[1]})", end="")
            # Save preserving original format; fall back to JPEG
            fmt = img.format or "JPEG"
            fixed.save(path, format=fmt)

        print()
        return True

    except Exception as e:
        print(f"  [ERR] {path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix EXIF-rotated images in-place.")
    parser.add_argument("folder", help="Path to image folder")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search subfolders too")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Only report, don't modify")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory", file=sys.stderr)
        sys.exit(1)

    glob = folder.rglob("*") if args.recursive else folder.glob("*")
    paths = sorted(p for p in glob if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)

    if not paths:
        print(f"No images found in {folder}")
        return

    print(f"Scanning {len(paths)} images in {folder}" + (" (dry-run)" if args.dry_run else "") + "...\n")

    fixed = sum(fix_image(p, args.dry_run) for p in paths)

    action = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{action} {fixed}/{len(paths)} images.")

# python3 -m tools.fix_rotated_images /data/now/amr_img_test/test --recursive --dry-run
if __name__ == "__main__":
    main()
