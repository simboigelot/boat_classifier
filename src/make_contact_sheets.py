"""
Build contact sheets (thumbnail grids with IDs) of the training images for visual review.

Output (in output/contact_sheets/):
  - <class>_<n>.png : grids of COLS x ROWS thumbnails, each labelled with an ID
  - index.csv       : ID -> file path mapping, used to act on the review afterwards

IDs are M### for motor_boat and S### for sail_boat, assigned in sorted filename order,
so they are stable as long as the input/data/ folder is not modified between runs.

Usage (from the project root): python src/make_contact_sheets.py
"""

import csv

from PIL import Image, ImageDraw, ImageFont

from paths import DATA_DIR, OUTPUT_DIR, ROOT

OUT_DIR = OUTPUT_DIR / "contact_sheets"
PREFIX = {"motor_boat": "M", "sail_boat": "S"}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

THUMB = 200          # thumbnail size in pixels
LABEL_H = 28         # height of the label strip under each thumbnail
COLS, ROWS = 6, 5    # 30 images per sheet

OUT_DIR.mkdir(exist_ok=True)
try:
    font = ImageFont.load_default(size=22)
except TypeError:     # Pillow < 10.1
    font = ImageFont.load_default()

rows_csv = []
for cls, prefix in PREFIX.items():
    files = sorted(p for p in (DATA_DIR / cls).iterdir() if p.suffix.lower() in EXTS)
    per_sheet = COLS * ROWS
    for sheet_no, start in enumerate(range(0, len(files), per_sheet), 1):
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * (THUMB + LABEL_H)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, path in enumerate(files[start:start + per_sheet]):
            img_id = f"{prefix}{start + k + 1:03d}"
            rows_csv.append((img_id, cls, path.relative_to(ROOT).as_posix()))
            x, y = (k % COLS) * THUMB, (k // COLS) * (THUMB + LABEL_H)
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((THUMB, THUMB))
                    sheet.paste(im, (x + (THUMB - im.width) // 2, y + (THUMB - im.height) // 2))
            except Exception:
                draw.text((x + 10, y + THUMB // 2), "UNREADABLE", fill="red", font=font)
            draw.text((x + 6, y + THUMB + 2), img_id, fill="black", font=font)
        out = OUT_DIR / f"{cls}_{sheet_no}.png"
        sheet.save(out)
        print(f"saved {out.name}")

with open(OUT_DIR / "index.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "class", "path"])
    w.writerows(rows_csv)
print(f"{len(rows_csv)} images indexed in {OUT_DIR / 'index.csv'}")
