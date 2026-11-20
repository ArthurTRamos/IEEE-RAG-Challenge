#!/bin/bash
# Package CURRENT local source/models without downloads or replacements.
# Usage: ./build.sh [--offline] [output.zip]
# --offline is accepted for compatibility; every build is now offline.
# Output paths are relative to src, as in the previous build script.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--offline" ]]; then
    shift
fi
if [[ $# -gt 1 || "${1:-}" == -* ]]; then
    echo "Usage: $0 [--offline] [output.zip]" >&2
    exit 2
fi
output_zip="${1:-../submissions/rag-submission-$(date +%Y%m%d-%H%M%S).zip}"

for required in main.py retrieval.py build.sh \
    embed_model/model.safetensors embed_model/modules.json \
    docling_models/docling-project--docling-layout-heron/model.safetensors \
    spacy_model/config.cfg spacy_model/meta.json; do
    if [[ ! -s "$required" ]]; then
        echo "Missing local submission file: $required" >&2
        exit 1
    fi
done
if [[ -e "$output_zip" ]]; then
    echo "Output already exists; choose a new zip name: $output_zip" >&2
    exit 1
fi
mkdir -p "$(dirname "$output_zip")"
zip -q -X -r "$output_zip" main.py retrieval.py build.sh docling_models embed_model spacy_model \
    -x '*/__pycache__/*' '*.pyc' '*/.cache/*' '*.Identifier' '*:Zone.Identifier'

python3 - "$output_zip" <<'PY'
import hashlib
from pathlib import Path
import sys
import zipfile

path = Path(sys.argv[1])
if path.stat().st_size > 1_000_000_000:
    raise SystemExit("Submission exceeds 1 GB")

def included(file):
    return (file.is_file() and not {"__pycache__", ".cache"}.intersection(file.parts)
            and file.suffix != ".pyc" and not file.name.endswith(".Identifier"))

expected = {p.as_posix() for p in Path('.').rglob('*') if included(p)
            and (p.parts[0] in {"docling_models", "embed_model", "spacy_model"}
                 or p.as_posix() in {"main.py", "retrieval.py", "build.sh"})}
with zipfile.ZipFile(path) as archive:
    bad = archive.testzip()
    if bad:
        raise SystemExit(f"Corrupt ZIP entry: {bad}")
    actual = {item.filename for item in archive.infolist() if not item.is_dir()}
    if actual != expected:
        raise SystemExit(f"ZIP file set differs: missing={expected - actual}, extra={actual - expected}")
    for name in sorted(expected):
        if hashlib.sha256(archive.read(name)).digest() != hashlib.sha256(Path(name).read_bytes()).digest():
            raise SystemExit(f"ZIP file differs from current source: {name}")
    for name in ("main.py", "retrieval.py"):
        compile(archive.read(name), name, "exec")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"-> {path.resolve()} ({path.stat().st_size:,} bytes)")
print(f"Verified {len(expected)} files against current src, CRC, Python syntax and 1 GB limit")
print(f"SHA-256: {digest}")
PY
