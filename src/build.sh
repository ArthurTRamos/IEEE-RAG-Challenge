#!/bin/bash
# Builds a submission zip, ready to upload to Codabench.
# Usage: ./build.sh [--offline] [output.zip]
# --offline reuses existing auxiliary models without deleting/downloading them.
#
# Docling and sentence-transformers (the libraries) are already in the
# challenge's evaluation environment, but two model weight files they need —
# Docling's "layout" model, and the "all-MiniLM-L6-v2" embedding model — are
# not pre-downloaded there, and that environment has no internet access to
# fetch them on its own. This script downloads both once, here, where there
# IS internet, and bundles them into the submission zip so the evaluation
# container never needs to reach out for either.
#
# Needs `docling-tools` (from the `docling` package) and `sentence-transformers`
# on PATH/importable, matching the versions pinned in the challenge's
# docker/requirements-participant.txt, plus `zip`. If you don't have them
# installed locally: pip install docling sentence-transformers (matching
# those pinned versions).
set -euo pipefail
cd "$(dirname "$0")"

offline=0
if [[ "${1:-}" == "--offline" ]]; then
    offline=1
    shift
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--offline] [output.zip]" >&2
    exit 2
fi
output_zip="${1:-../02_rag_submission.zip}"

if [[ "$offline" == 0 ]]; then
    rm -rf docling_models
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 docling-tools models download layout -o ./docling_models
    rm -rf embed_model
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('embed_model')
"
fi

for required in embed_model/model.safetensors embed_model/modules.json \
    docling_models/docling-project--docling-layout-heron/model.safetensors; do
    if [[ ! -s "$required" ]]; then
        echo "Missing auxiliary model file: $required" >&2
        exit 1
    fi
done
if [[ -e "$output_zip" ]]; then
    echo "Output already exists; choose a new zip name: $output_zip" >&2
    exit 1
fi
mkdir -p "$(dirname "$output_zip")"
zip -q -r "$output_zip" main.py retrieval.py docling_models embed_model \
    -x '*/__pycache__/*' '*.pyc' '*/.cache/*' '*.Identifier'
python3 - "$output_zip" <<'PY'
from pathlib import Path
import sys
import zipfile

path = Path(sys.argv[1])
if path.stat().st_size > 1_000_000_000:
    raise SystemExit("Submission exceeds 1 GB")
with zipfile.ZipFile(path) as archive:
    bad = archive.testzip()
    if bad:
        raise SystemExit(f"Corrupt ZIP entry: {bad}")
print(f"-> {path.resolve()} ({path.stat().st_size:,} bytes)")
PY
