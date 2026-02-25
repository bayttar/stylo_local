import json
from pathlib import Path
import pandas as pd

def main():
    base = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    sections_jsonl = base / "sections.jsonl"

    if not sections_jsonl.exists():
        raise FileNotFoundError("sections.jsonl not found")

    rows = []
    with open(sections_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rows.append({"tei_file": rec.get("tei_file", "")})

    df = pd.DataFrame(rows)
    out = base / "TEST_output.csv"
    df.to_csv(out, index=False)

    print("Saved:", out)

if __name__ == "__main__":
    main()
