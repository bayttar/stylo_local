import os, re, json, argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import spacy
from pypdf import PdfReader
from lexicalrichness import LexicalRichness

DEFAULT_PDF_DIR = str(Path.home() / "stylo_local" / "pdfs")
DEFAULT_OUT_DIR = str(Path.home() / "stylo_local" / "stylo_out")

SPACY_MODEL = "en_core_web_sm"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", default=DEFAULT_PDF_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--extractor", default="pypdf",
                   choices=["pypdf","pymupdf4llm","marker","grobid"])
    return p.parse_args()

args = parse_args()
PDF_DIR = args.pdf_dir
OUT_DIR = args.out_dir
EXTRACTOR = args.extractor

print("PDF_DIR:", PDF_DIR)
print("OUT_DIR:", OUT_DIR)
print("EXTRACTOR:", EXTRACTOR)

# ---------- spaCy ----------
nlp = spacy.load(SPACY_MODEL)
print("spaCy loaded:", SPACY_MODEL)

# ---------- Regex ----------
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EMDASH_RE = re.compile(r"—|--")
SEMICOLON_RE = re.compile(r";")
COLON_RE = re.compile(r":")
MID_SENT_COLON_RE = re.compile(r":(?!\s*$)")

# ---------- Extraction ----------
def clean(text):
    text = (text or "").replace("\r\n","\n").replace("\r","\n")
    return re.sub(r"\n{3,}","\n\n",text).strip()

def extract_pypdf(path):
    reader = PdfReader(path)
    return clean("\n".join(p.extract_text() or "" for p in reader.pages))

def extract_pymupdf(path):
    import pymupdf4llm
    return clean(pymupdf4llm.to_markdown(path))

def extract_marker(path):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    conv = PdfConverter(artifact_dict=create_model_dict())
    rendered = conv(path)
    text,_,_ = text_from_rendered(rendered)
    return clean(text)

def extract_grobid(path):
    from lxml port etree
    NS = {"tei":"http://www.tei-c.org/ns/1.0"}
    tei_dir = Path.home()/"stylo_local"/"stylo_out"/"grobid_tei"
    tei = tei_dir/(Path(path).stem+".tei.xml")
    if not tei.exists():
        raise FileNotFoundError(f"Missing TEI: {tei}")
    root = etree.parse(str(tei)).getroot()
    body = " ".join(" ".join(n.itertext())
                    for n in root.xpath(".//tei:text/tei:body",namespaces=NS))
    absn = " ".join(" ".join(n.itertext())
                    for n in root.xpath(".//tei:profileDesc/tei:abstract",namespaces=NS))
    return clean(absn+"\n\n"+body)

def extract(path):
    if EXTRACTOR=="pypdf": return extract_pypdf(path)
    if EXTRACTOR=="pymupdf4llm": return extract_pymupdf(path)
    if EXTRACTOR=="marker": return extract_marker(path)
    if EXTRACTOR=="grobid": return extract_grobid(path)
    raise ValueError("Unknown extractor")

# ---------- Load corpus ----------
pdfs=[f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
if not pdfs:
    raise FileNotFoundError("No PDFs found.")

corpus={}
for fn in sorted(pdfs):
    txt=extract(os.path.join(PDF_DIR,fn))
    if not txt.strip():
        raise ValueError(f"Empty extraction: {fn}")
    corpus[fn]=txt

print("PDF count:",len(corpus))

# ---------- Simple Metrics ----------
results=[]
for fn,txt in tqdm(corpus.items()):
    doc=nlp(txt)
    sents=[sum(1 for t in s if t.is_alpha) for s in doc.sents]
    words=WORD_RE.findall(txt)
    results.append({
        "file":fn,
        "total_words":len(words),
        "total_sentences":len(sents),
        "avg_sentence_len":float(np.mean(sents)) if sents else 0,
        "emdash_total":len(EMDASH_RE.findall(txt)),
        "semicolons_total":len(SEMICOLON_RE.findall(txt)),
        "colons_total":len(COLON_RE.findall(txt)),
        "mtld":float(LexicalRichness(" ".join(words)).mtld(0.72)) if words else 0
    })

df=pd.DataFrame(results)

Path(OUT_DIR).mkdir(parents=True,exist_ok=True)
df.to_csv(Path(OUT_DIR)/"per_article_metrics.csv",index=False)

print("DONE")
