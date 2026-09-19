#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${UF850_COLLECTION_PYTHON:-python3}" -B "$ROOT/launch.py" collection --site-ready "$@"
