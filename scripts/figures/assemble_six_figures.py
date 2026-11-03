import json
import shutil
from pathlib import Path

import fitz

WORK = Path(__file__).resolve().parent
OUT = WORK / "figures/six_main"
CURRENT = Path("/public/home/mengxl/dzy/ad_omics_v3/delivery/figures/pdf")
SOURCES = {
    1: CURRENT / "Figure1.pdf",
    2: CURRENT / "Figure2.pdf",
    3: WORK / "figures/Figure3_real_action.pdf",
    4: WORK / "figures/Figure4_dynamic.pdf",
    5: WORK / "figures/updated_genetics/pdf/Figure4.pdf",
    6: CURRENT / "Figure5.pdf",
}
EXPECTED = {1: (1080, 1323), 2: (1080, 1458), 3: (1080, 1620), 4: (1080, 1512), 5: (1080, 1512), 6: (1080, 1620)}


def main():
    (OUT / "pdf").mkdir(parents=True, exist_ok=True)
    (OUT / "preview").mkdir(parents=True, exist_ok=True)
    combined = fitz.open()
    manifest = []
    for number, source in SOURCES.items():
        with fitz.open(source) as document:
            page = document[0]
            width, height = page.rect.width, page.rect.height
            assert abs(width - EXPECTED[number][0]) < 0.01 and abs(height - EXPECTED[number][1]) < 0.01
            combined.insert_pdf(document)
        target = OUT / "pdf" / f"Figure{number}.pdf"
        shutil.copy2(source, target)
        with fitz.open(target) as document:
            pixmap = document[0].get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
            pixmap.save(OUT / "preview" / f"Figure{number}.png")
        manifest.append({"figure": number, "source": str(source), "width_pt": width, "height_pt": height})
    combined.save(OUT / "AD_omics_v3_六张主图.pdf", garbage=4, deflate=True)
    combined.close()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
