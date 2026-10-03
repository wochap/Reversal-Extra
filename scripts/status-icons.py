#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python3 resvg
"""
Keep Reversal-Extra's status icons in one folder, status/scalable, where every
icon renders at the same visible size.

Icon themes ship status icons in several size folders (16, 22, 24, 32,
symbolic) whose drawings have different padding, so the same icon renders at a
different visible size depending on the requested size. Normalizing fixes that:
each SVG is rendered with resvg, the bounding box of its visible pixels is
measured, and the root <svg> gets a square viewBox centered on that box, sized
so the shape's longest side is RATIO of the canvas. Running it again on
normalized icons changes nothing.

Commands:
  import <theme-dir>   Replace status/scalable with the status icons of another
                       theme (folders merged in SOURCE_DIRS order, first match
                       wins), rewrite index.theme, then normalize.
  normalize            Normalize every SVG in status/scalable in place. Run this
                       after adding or editing icons.

Usage:
  ./scripts/status-icons.py import ~/Sandboxes/sandbox/vendor/Colloid-icon-theme/build/Colloid
  ./scripts/status-icons.py normalize
  ./scripts/status-icons.py normalize --ratio 0.75
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIRS = ["22", "24", "16", "32", "symbolic"]
OUT_DIR = "status/scalable"
CANVAS = 24
RENDER_WIDTH = 256
# Re-normalizing measures the bbox again with pixel rounding; viewBoxes that
# differ by less than this fraction of their side are kept as-is
TOLERANCE = 0.02

STATUS_SECTION = f"""[{OUT_DIR}]
Size={CANVAS}
Context=Status
MinSize=8
MaxSize=512
Type=Scalable
"""


def read_png_alpha_bbox(path):
    """Return (width, height, bbox) of a RGBA8 PNG, bbox = (x0, y0, x1, y1) or None."""
    with open(path, "rb") as f:
        data = f.read()
    pos, idat = 8, b""
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", chunk[:10])
            if depth != 8 or color != 6:
                raise ValueError(f"unsupported PNG format depth={depth} color={color}")
        elif kind == b"IDAT":
            idat += chunk
        pos += 12 + length

    raw = zlib.decompress(idat)
    stride, bpp = width * 4, 4
    prev = bytearray(stride)
    x0, y0, x1, y1 = width, height, -1, -1
    for y in range(height):
        start = y * (stride + 1)
        kind = raw[start]
        line = bytearray(raw[start + 1:start + 1 + stride])
        if kind == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
                line[i] = (line[i] + pred) & 0xFF
        alpha = line[3::4]
        if any(alpha):
            first = next(i for i, v in enumerate(alpha) if v)
            last = width - 1 - next(i for i, v in enumerate(reversed(alpha)) if v)
            x0, x1 = min(x0, first), max(x1, last)
            y0 = min(y0, y)
            y1 = y
        prev = line
    bbox = (x0, y0, x1 + 1, y1 + 1) if x1 >= 0 else None
    return width, height, bbox


def parse_length(value):
    match = re.match(r"\s*([0-9.]+)\s*(px)?\s*$", value or "")
    return float(match.group(1)) if match else None


def normalize(src, ratio):
    """Return the normalized SVG text of `src`, or None when it renders empty."""
    text = open(src, encoding="utf-8").read()
    root = re.search(r"<svg\b[^>]*>", text)
    if not root:
        raise ValueError("no <svg> root")
    tag = root.group(0)

    attr = lambda name: (re.search(rf'\s{name}=["\']([^"\']*)["\']', tag) or [None, None])[1]
    view_box = attr("viewBox")
    if view_box:
        vx, vy, vw, vh = (float(v) for v in re.split(r"[\s,]+", view_box.strip()))
    else:
        vx, vy = 0.0, 0.0
        vw, vh = parse_length(attr("width")), parse_length(attr("height"))
        if not vw or not vh:
            raise ValueError("no viewBox and no usable width/height")

    with tempfile.NamedTemporaryFile(suffix=".png") as png:
        subprocess.run(["resvg", "-w", str(RENDER_WIDTH), src, png.name],
                       check=True, capture_output=True)
        width, height, bbox = read_png_alpha_bbox(png.name)
    if not bbox:
        return None

    # Pixel bbox -> user units of the original viewBox
    sx, sy = vw / width, vh / height
    bx0, by0 = vx + bbox[0] * sx, vy + bbox[1] * sy
    bx1, by1 = vx + bbox[2] * sx, vy + bbox[3] * sy
    side = max(bx1 - bx0, by1 - by0) / ratio
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    new_box = (cx - side / 2, cy - side / 2, side, side)

    old_box = (vx, vy, vw, vh) if view_box else None
    if old_box and all(abs(a - b) <= TOLERANCE * side for a, b in zip(old_box, new_box)):
        return text

    new_tag = re.sub(r"""\s(width|height|viewBox|preserveAspectRatio)=("[^"]*"|'[^']*')""", "", tag)
    new_tag = new_tag.replace(
        "<svg",
        f'<svg width="{CANVAS}" height="{CANVAS}" '
        'viewBox="{:.4f} {:.4f} {:.4f} {:.4f}"'.format(*new_box),
        1,
    )
    return text[:root.start()] + new_tag + text[root.end():]


def collect(theme):
    """Map icon file name -> real source path, first SOURCE_DIRS match wins."""
    chosen = {}
    for sub in SOURCE_DIRS:
        folder = os.path.join(theme, "status", sub)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if name.endswith(".svg") and name not in chosen and os.path.exists(path):
                chosen[name] = os.path.realpath(path)
    return chosen


def write_index_theme():
    path = os.path.join(REPO, "index.theme")
    text = open(path, encoding="utf-8").read()
    sections = re.split(r"\n(?=\[)", text.strip())
    kept = [s for s in sections if not s.startswith("[status")]
    dirs = []
    for s in kept[1:]:
        dirs.append(s.split("]", 1)[0][1:])
    dirs.append(OUT_DIR)
    kept[0] = re.sub(r"^Directories=.*$", "Directories=" + ",".join(dirs), kept[0], flags=re.M)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(s.strip() for s in kept) + "\n\n" + STATUS_SECTION)


def normalize_folder(folder, ratio):
    """Normalize every regular SVG in `folder` in place (symlinks are aliases, left alone)."""
    paths = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".svg") and not os.path.islink(os.path.join(folder, name))
    )
    changed, empty, failed = 0, 0, []
    with concurrent.futures.ProcessPoolExecutor() as pool:
        futures = {path: pool.submit(normalize, path, ratio) for path in paths}
        for path, future in futures.items():
            try:
                content = future.result()
            except Exception as error:  # keep going, report at the end
                failed.append((path, error))
                continue
            if content is None:
                empty += 1
                continue
            if content != open(path, encoding="utf-8").read():
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                changed += 1
    print(f"{os.path.relpath(folder, REPO)}: {len(paths)} icons, {changed} changed, {empty} empty (skipped)")
    for path, error in failed:
        print(f"  failed: {os.path.relpath(path, REPO)}: {error}", file=sys.stderr)
    return not failed


def import_theme(theme):
    """Replace status/scalable with the raw status icons of `theme`."""
    chosen = collect(theme)
    if not chosen:
        sys.exit(f"no status icons found in {theme}/status/{{{','.join(SOURCE_DIRS)}}}")

    for stale in ("status", "status@2x", "status@3x"):
        path = os.path.join(REPO, stale)
        if os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    out = os.path.join(REPO, OUT_DIR)
    os.makedirs(out)

    # Aliases of the same real file become symlinks to the first name
    first_name = {}
    for name, real in chosen.items():
        if real in first_name:
            os.symlink(first_name[real], os.path.join(out, name))
        else:
            first_name[real] = name
            shutil.copyfile(real, os.path.join(out, name))
    write_index_theme()
    print(f"{OUT_DIR}: imported {len(first_name)} icons, {len(chosen) - len(first_name)} aliases")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ratio", type=float, default=16 / 24,
                        help="longest side of the shape relative to the canvas (default 16/24)")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    import_parser = commands.add_parser("import", parents=[common], help="import status icons from another theme")
    import_parser.add_argument("theme", help="source icon theme directory (contains status/)")
    commands.add_parser("normalize", parents=[common], help="normalize status/scalable in place")
    args = parser.parse_args()

    if args.command == "import":
        import_theme(args.theme)
    ok = normalize_folder(os.path.join(REPO, OUT_DIR), args.ratio)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
