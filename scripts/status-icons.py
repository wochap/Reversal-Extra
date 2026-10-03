#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python3 resvg
"""
Normalize the viewBox of Reversal-Extra's status icons so every icon in a size
folder renders at the same visible size.

The status folders (status/16, 22, 24, 32, symbolic) come from another theme
whose drawings use different padding: in status/24 one shape is 14x12, another
20x16, and symbolic icons fill the whole canvas. Normalizing fixes that per
folder: each SVG is rendered with resvg, the bounding box of its visible pixels
is measured, and the root <svg> gets a square viewBox centered on that box,
sized so the shape's longest side is RATIO of the folder's canvas (the Size= of
its index.theme section). Running it again on normalized icons changes nothing.

Commands:
  normalize            Normalize every status folder in place. Run this after
                       adding or editing icons.
  import <theme-dir>   Replace the status folders, their @2x/@3x links and their
                       index.theme sections with another theme's, then normalize.

Usage:
  ./scripts/status-icons.py normalize
  ./scripts/status-icons.py normalize --ratio 0.75
  ./scripts/status-icons.py import ~/Sandboxes/sandbox/vendor/Colloid-icon-theme/build/Colloid
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
RENDER_WIDTH = 256
# Re-normalizing measures the bbox again with pixel rounding; viewBoxes that
# differ by less than this fraction of their side are kept as-is
TOLERANCE = 0.02

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


def normalize(src, ratio, canvas):
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
        f'<svg width="{canvas}" height="{canvas}" '
        'viewBox="{:.4f} {:.4f} {:.4f} {:.4f}"'.format(*new_box),
        1,
    )
    return text[:root.start()] + new_tag + text[root.end():]


def theme_sections(text):
    """Split index.theme text into (name, body) pairs, name None for the preamble."""
    sections = []
    for chunk in re.split(r"\n(?=\[)", text.strip()):
        match = re.match(r"\[([^\]]+)\]", chunk)
        sections.append((match.group(1) if match else None, chunk.strip()))
    return sections


def status_folders():
    """Map status folder -> canvas size, from index.theme (the @2x/@3x links are skipped)."""
    folders = {}
    for name, body in theme_sections(open(os.path.join(REPO, "index.theme"), encoding="utf-8").read()):
        if name and name.startswith("status/"):
            size = re.search(r"^Size=(\d+)", body, re.M)
            folders[name] = int(size.group(1))
    return folders


def normalize_folder(folder, ratio, canvas):
    """Normalize every regular SVG in `folder` in place (symlinks are aliases, left alone)."""
    paths = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".svg") and not os.path.islink(os.path.join(folder, name))
    )
    changed, empty, failed = 0, 0, []
    with concurrent.futures.ProcessPoolExecutor() as pool:
        futures = {path: pool.submit(normalize, path, ratio, canvas) for path in paths}
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
    print(f"{os.path.relpath(folder, REPO)} ({canvas}px): {len(paths)} icons, {changed} changed, {empty} empty (skipped)")
    for path, error in failed:
        print(f"  failed: {os.path.relpath(path, REPO)}: {error}", file=sys.stderr)
    return not failed


def import_theme(theme):
    """Replace the status folders, @2x/@3x links and index.theme sections with `theme`'s."""
    source = os.path.join(theme, "status")
    if not os.path.isdir(source):
        sys.exit(f"no status folder in {theme}")

    for stale in ("status", "status@2x", "status@3x"):
        path = os.path.join(REPO, stale)
        if os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    # Keep links inside status/, copy the files behind links that leave it
    out = os.path.join(REPO, "status")
    shutil.copytree(os.path.realpath(source), out, symlinks=True)
    for folder, _, names in os.walk(out):
        for name in names:
            path = os.path.join(folder, name)
            if os.path.islink(path) and "/" in os.readlink(path):
                target = os.path.join(os.path.realpath(source), os.path.relpath(path, out))
                os.remove(path)
                shutil.copyfile(os.path.realpath(target), path)
    for scale in ("status@2x", "status@3x"):
        if os.path.lexists(os.path.join(theme, scale)):
            os.symlink("status", os.path.join(REPO, scale))

    # index.theme: drop our status sections, append the source theme's
    path = os.path.join(REPO, "index.theme")
    ours = [s for s in theme_sections(open(path, encoding="utf-8").read()) if not (s[0] or "").startswith("status")]
    theirs = [s for s in theme_sections(open(os.path.join(theme, "index.theme"), encoding="utf-8").read())
              if (s[0] or "").startswith("status")]
    dirs = [name for name, _ in ours[1:] + theirs]
    preamble = re.sub(r"^Directories=.*$", "Directories=" + ",".join(dirs), ours[0][1], flags=re.M)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join([preamble] + [body for _, body in ours[1:] + theirs]) + "\n")
    print(f"imported status folders from {theme}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ratio", type=float, default=16 / 24,
                        help="longest side of the shape relative to the canvas (default 16/24)")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("normalize", parents=[common], help="normalize every status folder in place")
    import_parser = commands.add_parser("import", parents=[common], help="import status folders from another theme")
    import_parser.add_argument("theme", help="source icon theme directory (contains status/ and index.theme)")
    args = parser.parse_args()

    if args.command == "import":
        import_theme(args.theme)
    ok = True
    for folder, canvas in status_folders().items():
        ok = normalize_folder(os.path.join(REPO, folder), args.ratio, canvas) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
