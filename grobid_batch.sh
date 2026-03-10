#!/bin/bash
set -euo pipefail

PDF_DIR="$HOME/stylo_local/pdfs"
OUT_DIR="$HOME/stylo_local/stylo_out/grobid_tei"
mkdir -p "$OUT_DIR"

if ! command -v xmllint >/dev/null 2>&1; then
  echo "ERROR: xmllint is required to validate GROBID XML responses."
  exit 1
fi

count=0
for pdf in "$PDF_DIR"/*.pdf; do
  base="$(basename "$pdf" .pdf)"
  out="$OUT_DIR/$base.tei.xml"
  tmp_out="$(mktemp)"
  http_code=""

  echo "Processing: $base.pdf"
  http_code="$(curl -sS \
    -w "%{http_code}" \
    -F "input=@$pdf" \
    http://localhost:8070/api/processFulltextDocument \
    -o "$tmp_out")"

  if [ "$http_code" != "200" ]; then
    echo "ERROR: GROBID returned HTTP $http_code for $pdf"
    rm -f "$tmp_out"
    exit 1
  fi

  if [ ! -s "$tmp_out" ]; then
    echo "ERROR: Empty TEI for $pdf"
    rm -f "$tmp_out"
    exit 1
  fi

  if ! xmllint --noout "$tmp_out" >/dev/null 2>&1; then
    echo "ERROR: Invalid XML returned for $pdf"
    rm -f "$tmp_out"
    exit 1
  fi

  mv "$tmp_out" "$out"

  count=$((count+1))
done

echo "Done. TEI files: $count"
echo "Output folder: $OUT_DIR"
