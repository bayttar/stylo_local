from pathlib import Path
from lxml import etree

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def tei_sections(tei_path: str):
    tree = etree.parse(tei_path)
    root = tree.getroot()

    abstract_nodes = root.xpath(".//tei:profileDesc/tei:abstract", namespaces=NS)
    abstract = "\n".join([" ".join(a.itertext()) for a in abstract_nodes]).strip()

    sections = {}
    divs = root.xpath(".//tei:text/tei:body//tei:div", namespaces=NS)

    for i, div in enumerate(divs, start=1):
        head = div.xpath("./tei:head", namespaces=NS)
        head_text = (" ".join(head[0].itertext()).strip() if head else f"SECTION_{i}")
        body_text = " ".join(div.itertext()).strip()
        if body_text:
            sections[head_text] = body_text

    return {
        "abstract": abstract,
        "sections": sections
    }

if __name__ == "__main__":
    tei_dir = Path.home() / "stylo_local" / "stylo_out" / "grobid_tei"
    files = sorted(tei_dir.glob("*.tei.xml"))

    if not files:
        print("No TEI files found in:", tei_dir)
        exit(1)

    sample = files[0]
    data = tei_sections(str(sample))

    print("Sample file:", sample.name)
    print("Abstract chars:", len(data["abstract"]))
    print("Section count:", len(data["sections"]))
    print("First 5 section titles:", list(data["sections"].keys())[:5])
