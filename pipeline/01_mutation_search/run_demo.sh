#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python search_mutations.py -s KSMLKSMK -k 1 -o outputs/demo
