#!/usr/bin/env python3
"""Replay the current 15-method AMP classification benchmark."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from amp_classification.replay import main
if __name__ == '__main__': main()
