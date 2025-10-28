#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d bin ]; then
  echo "bin/ not found. Compile first: ./compile.sh" >&2
  exit 1
fi
exec java -cp bin com.example.gradesystem.Main "$@"