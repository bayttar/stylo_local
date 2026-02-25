#!/bin/bash
set -euo pipefail

PDF_DIR="$HOME/stylo_local/pdfs"
OUT_DIR="$HOME/stylo_local/stylo_out/grobid_tei"
mkdir -p "$OUT_DIR"

count=0
for pdf in "$PDF_DIR"/*.pdf; do
  base="$(basename "$pdf" .pdf)"
  out="$OUT_DIR/$base.tei.xml"

  echo "Processing: $base.pdf"
  curl -s \
    -F "input=@$pdf" \
    http://localhost:8070/api/processFulltextDocument \
    -o "$out"

  # basic sanity check (file exists and not empty)
  if [ ! -s "$out" ]; then
    echo "ERROR: Empty TEI for $pdf"
    exit 1
  fi

  count=$((count+1))
done

echo "Done. TEI files: $count"
echo "Output folder: $OUT_DIR"
