#!/usr/bin/env bash
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Build the paper PDF and an arXiv-ready source tarball (with the .bbl) into a build directory OUTSIDE docs/paper/.
#
# OWNER: this script owns how the paper is built; the sources are seabraid.tex, refs.bib and figures/*.tex.
# CONTRACT: build.sh [BUILD_DIR]   (default: ${TMPDIR:-/tmp}/seabraid-paper-build)
#   writes BUILD_DIR/seabraid.pdf and BUILD_DIR/seabraid-arxiv.tar.gz.  Needs pdflatex and bibtex (a stock TeX Live).
# FAILURE: fail-fast and loud.  No TeX engine: exit 2 with a message, nothing is built.  Any LaTeX or BibTeX error, an undefined
#   reference or citation, or a missing bibliography result: exit 1.  This script never downloads anything.
# CHECKER: the log scan below (undefined references, missing citations, overfull errors) and, when pdffonts exists, the
#   font check (every font embedded).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${TMPDIR:-/tmp}/seabraid-paper-build}"
mkdir -p "$OUT"
case "$(cd "$OUT" && pwd -P)/" in
  "$(cd "$HERE" && pwd -P)/"*) echo "build.sh: BUILD_DIR must be outside docs/paper/" >&2; exit 2 ;;
esac

for tool in pdflatex bibtex; do
  command -v "$tool" >/dev/null 2>&1 || { echo "build.sh: '$tool' not found; install a TeX Live (pdflatex, bibtex, tikz, booktabs, hyperref, lmodern)" >&2; exit 2; }
done

WORK="$OUT/work"
rm -rf "$WORK"
mkdir -p "$WORK/figures"
cp "$HERE/seabraid.tex" "$HERE/refs.bib" "$WORK/"
cp "$HERE"/figures/*.tex "$WORK/figures/"

cd "$WORK"
run() { "$@" >/dev/null 2>run.err || { echo "build.sh: '$*' failed; see $WORK/seabraid.log" >&2; tail -n 20 seabraid.log >&2 || true; exit 1; }; }
run pdflatex -interaction=nonstopmode -halt-on-error seabraid.tex
run bibtex seabraid
run pdflatex -interaction=nonstopmode -halt-on-error seabraid.tex
run pdflatex -interaction=nonstopmode -halt-on-error seabraid.tex

[ -s seabraid.bbl ] || { echo "build.sh: bibtex produced no seabraid.bbl" >&2; exit 1; }
if grep -E "undefined|Citation .* undefined|Label\(s\) may have changed|LaTeX Error" seabraid.log; then
  echo "build.sh: undefined reference, citation or LaTeX error in the log (lines above)" >&2
  exit 1
fi
if command -v pdffonts >/dev/null 2>&1; then
  if pdffonts seabraid.pdf | awk 'NR>2 && $(NF-4)!="yes" {bad=1; print} END {exit bad}'; then :; else
    echo "build.sh: a font is not embedded (lines above)" >&2; exit 1
  fi
fi

cp seabraid.pdf "$OUT/seabraid.pdf"
# arXiv wants source, not the PDF; include the .bbl (arXiv does not run bibtex) and leave out refs.bib, the log and the PDF.
mkdir -p arxiv/figures
cp seabraid.tex seabraid.bbl arxiv/
cp figures/*.tex arxiv/figures/
tar -czf "$OUT/seabraid-arxiv.tar.gz" -C arxiv .
echo "built: $OUT/seabraid.pdf and $OUT/seabraid-arxiv.tar.gz"
command -v pdfinfo >/dev/null 2>&1 && pdfinfo "$OUT/seabraid.pdf" | grep -E "^Pages:" || true
