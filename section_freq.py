import json
from pathlib import Path
from collections import Counter
import pandas as pd

def main():
    base = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    sections_jsonl = base / "sections.jsonl"
    if not sections_jsonl.exists():
        raise FileNotFoundError(f"Missing: {sections_jsonl}")

    c = Counter()

    with open(sections_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            if (rec.get("abstract") or "").strip():
                c["Abstract"] += 1

            for sec in rec.get("sections", []):
                name = (sec.get("norm_head", "Untitled") or "Untitled").strip()
                txt = (sec.get("text", "") or "").strip()
                if not txt:
                    continue
                if name == "References":
                    continue
                c[name] += 1

    df = pd.DataFrame([{"section": k, "count": v} for k, v in c.most_common()])
    out = base / "section_name_frequencies.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    main()
