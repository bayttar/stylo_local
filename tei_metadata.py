from pathlib import Path
from lxml import etree
import pandas as pd
import re

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def first_text(root, xpath: str) -> str:
    nodes = root.xpath(xpath, namespaces=NS)
    if not nodes:
        return ""
    return clean(" ".join(nodes[0].itertext()))

def all_texts(root, xpath: str):
    nodes = root.xpath(xpath, namespaces=NS)
    out = []
    for n in nodes:
        out.append(clean(" ".join(n.itertext())))
    return [x for x in out if x]

def extract_one(tei_path: Path):
    tree = etree.parse(str(tei_path))
    root = tree.getroot()

    title = first_text(root, ".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title")
    doi = first_text(root, ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblStruct//tei:idno[@type='DOI']")

    # year: prefer date/@when, fallback to text
    year = ""
    date_when = root.xpath(".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblStruct//tei:imprint//tei:date/@when", namespaces=NS)
    if date_when:
        year = clean(date_when[0])[:4]
    else:
        date_text = first_text(root, ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblStruct//tei:imprint//tei:date")
        year = date_text[:4] if date_text else ""

    journal = first_text(root, ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblStruct//tei:monogr//tei:title[@level='j']")
    publisher = first_text(root, ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblStruct//tei:monogr//tei:imprint//tei:publisher")

    # authors: from analytic/author
    author_nodes = root.xpath(".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblStruct//tei:analytic//tei:author", namespaces=NS)
    authors = []
    for a in author_nodes:
        pers = a.xpath(".//tei:persName", namespaces=NS)
        if pers:
            authors.append(clean(" ".join(pers[0].itertext())))
        else:
            authors.append(clean(" ".join(a.itertext())))
    authors = [x for x in authors if x]

    return {
        "tei_file": tei_path.name,
        "file_stem": tei_path.name.replace(".tei.xml", ""),
        "title": title,
        "authors": "; ".join(authors),
        "author_count": len(authors),
        "year": year,
        "doi": doi,
        "journal": journal,
        "publisher": publisher,
    }

def main():
    tei_dir = Path.home() / "stylo_local" / "stylo_out" / "grobid_tei"
    out_dir = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    out_dir.mkdir(parents=True, exist_ok=True)

    tei_files = sorted(tei_dir.glob("*.tei.xml"))
    if not tei_files:
        raise FileNotFoundError(f"No TEI files found in: {tei_dir}")

    rows = [extract_one(p) for p in tei_files]
    df = pd.DataFrame(rows)

    out_csv = out_dir / "metadata_from_tei.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print("Rows:", len(df))

if __name__ == "__main__":
    main()
