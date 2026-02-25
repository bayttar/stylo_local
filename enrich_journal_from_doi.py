from pathlib import Path
import pandas as pd
import requests
import time

CROSSREF_URL = "https://api.crossref.org/works/"

def fetch_journal(doi: str):
    try:
        r = requests.get(CROSSREF_URL + doi, timeout=15)
        if r.status_code != 200:
            return "", ""
        msg = r.json().get("message", {})
        container = msg.get("container-title", []) or []
        journal = container[0] if container else ""
        publisher = msg.get("publisher", "") or ""
        return journal, publisher
    except Exception:
        return "", ""

def main():
    base = Path.home() / "stylo_local" / "stylo_out" / "grobid_sections"
    meta_path = base / "metadata_from_tei.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    df = pd.read_csv(meta_path)
    if "doi" not in df.columns:
        raise ValueError("metadata_from_tei.csv has no 'doi' column")

    journals = []
    publishers = []

    for doi in df["doi"].astype(str).tolist():
        doi = doi.strip()
        j, p = fetch_journal(doi) if doi else ("", "")
        journals.append(j)
        publishers.append(p)
        time.sleep(0.12)

    df["journal_crossref"] = journals
    df["publisher_crossref"] = publishers

    out = base / "metadata_enriched.csv"
    df.to_csv(out, index=False)

    print("Saved:", out)
    print("Unique journals:", pd.Series(journals).replace("", pd.NA).dropna().nunique())

if __name__ == "__main__":
    main()
