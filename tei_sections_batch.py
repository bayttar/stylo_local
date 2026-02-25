from pathlib import Path
from typing import Dict, Any, List
from lxml import etree
import json
import re

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def norm_head(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"^\d+(\.\d+)*\s+", "", s)
    s_low = s.lower()

    if "abstract" in s_low:
        return "Abstract"
    if any(k in s_low for k in ["introduction", "background"]):
        return "Introduction"
    if any(k in s_low for k in ["method", "materials", "methodology", "data", "corpus"]):
        return "Methods"
    if any(k in s_low for k in ["result", "findings"]):
        return "Results"
    if any(k in s_low for k in ["discussion", "analysis"]):
        return "Discussion"
    if any(k in s_low for k in ["conclusion", "concluding", "limitations", "future work"]):
        return "Conclusion"
    if any(k in s_low for k in ["references", "bibliography"]):
        return "References"
    return s if s else "Untitled"

def tei_to_sections(tei_path: Path) -> Dict[str, Any]:
    tree = etree.parse(str(tei_path))
    root = tree.getroot()

    abstract_nodes = root.xpath(".//tei:profileDesc/tei:abstract", namespaces=NS)
    abstract = "\n".join([" ".join(a.itertext()) for a in abstract_nodes]).strip()

    sections: List[Dict[str, str]] = []
    divs = root.xpath(".//tei:text/tei:body//tei:div", namespaces=NS)

    for i, div in enumerate(divs, start=1):
        head = div.xpath("./tei:head", namespaces=NS)
        head_text = (" ".join(head[0].itertext()).strip() if head else f"SECTION_{i}")
        body_text = " ".join(div.itertext()).strip()
        if body_text:
            sections.append({
                "raw_head": head_text,
                "norm_head": norm_head(head_text),
                "text": body_text
            })

    return {
        "tei_file": tei_path.name,
        "abstract": abstract,
        "sections": sections
    }

def main():
    tei_dir = Path.home() / "stylo_local" / "stylo_out" / "grobid_tei"
    out_dir = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    out_dir.mkdir(parents=True, exist_ok=True)

    tei_files = sorted(tei_dir.glob("*.tei.xml"))
    if not tei_files:
        raise FileNotFoundError(f"No TEI files found in: {tei_dir}")

    jsonl_path = out_dir / "sections.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for tei in tei_files:
            record = tei_to_sections(tei)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Wrote:", jsonl_path)
    print("TEI count:", len(tei_files))

if __name__ == "__main__":
    main()
