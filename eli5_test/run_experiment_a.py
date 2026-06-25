"""
run_experiment_a.py
===================
Convenience entry point that runs generation first and evaluation second.
"""

from __future__ import annotations

from generate_experiment_a import main as generate_main
from evaluate_experiment_a import main as evaluate_main


def main() -> None:
    generate_main()
    evaluate_main()


if __name__ == "__main__":
    main()
