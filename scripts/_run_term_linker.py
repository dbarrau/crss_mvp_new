#!/usr/bin/env python3
"""Quick script to run the term linker.

Usage:
    PYTHONPATH=. python scripts/_run_term_linker.py          # write edges
    PYTHONPATH=. python scripts/_run_term_linker.py --dry-run # preview only
"""
import argparse
import logging
import sys
import traceback

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

parser = argparse.ArgumentParser(description="Run the USES_TERM term linker.")
parser.add_argument("--dry-run", action="store_true", help="Preview only")
args = parser.parse_args()

try:
    from canonicalization.term_linker import link_terms
    result = link_terms(dry_run=args.dry_run)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
