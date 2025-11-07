"""
Output Formatter
Formats and displays generated attacks in a readable format
"""

import json
from typing import Any


def pretty_print_batch(batch: list[dict[str, Any]]):
    """
    Print generated batch neatly.
    """
    for i, item in enumerate(batch, start=1):
        print(f"\n🧨 Example {i}: {item['metadata']['attack_type'].upper()}")
        print(json.dumps(item, indent=2))
