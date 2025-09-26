from pathlib import Path
from PIL import Image
import json

def tile_image(img_path, label_path, out_img_dir, out_lbl_dir, tile=1024, overlap=0.2):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    stride = int(tile * (1 - overlap))
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # YOLO label: class x_center y_center width height (normalized)
    boxes = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                c, x, y, bw, bh = map(float, line.strip().split())
                boxes.append((int(c), x, y, bw, bh))

    tiles = []
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1, y1 = min(x0+tile, w), min(y0+tile, h)
            if x1 - x0 < tile or y1 - y0 < tile:
                continue
            crop = img.crop((x0, y0, x1, y1))
            tiles.append((crop, x0, y0, x1, y1))

    for i, (crop, x0, y0, x1, y1) in enumerate(tiles):
        tw, th = crop.size
        lbl_lines = []
        for (c, xc, yc, bw, bh) in boxes:
            # denormalize
            bx = xc * w; by = yc * h; bwp = bw * w; bhp = bh * h
            x_min = bx - bwp/2; y_min = by - bhp/2
            x_max = bx + bwp/2; y_max = by + bhp/2
            # intersection with tile
            ix_min = max(x_min, x0); iy_min = max(y_min, y0)
            ix_max = min(x_max, x1); iy_max = min(y_max, y1)
            if ix_max <= ix_min or iy_max <= iy_min:
                continue
            # clip and renormalize to tile
            cx = (max(ix_min, x0) + min(ix_max, x1)) / 2 - x0
            cy = (max(iy_min, y0) + min(iy_max, y1)) / 2 - y0
            cw = (ix_max - ix_min); ch = (iy_max - iy_min)
            # filter very small leftovers
            if cw < 4 or ch < 4:
                continue
            lbl_lines.append(f"{c} {cx/tw:.6f} {cy/th:.6f} {cw/tw:.6f} {ch/th:.6f}")

        out_name = f"{Path(img_path).stem}_{i:04d}"
        crop.save(out_img_dir / f"{out_name}.jpg", quality=95)
        with open(out_lbl_dir / f"{out_name}.txt", "w") as f:
            f.write("\n".join(lbl_lines))
