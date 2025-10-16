#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
shopt -s globstar nullglob
sources=(src/main/java/**/*.java)
if [ ${#sources[@]} -eq 0 ]; then
  echo "No Java sources found under src/main/java" >&2
  exit 1
fi
javac -encoding UTF-8 -Xlint:unchecked -d bin "${sources[@]}"
echo "Compiled ${#sources[@]} sources to bin/"