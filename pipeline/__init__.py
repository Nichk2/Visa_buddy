"""Visa Buddy v2 pipeline: corpus -> index -> eval -> (sft) -> serve.

Every stage records what it read, what it wrote, and the numbers it computed
in runs/ledger.jsonl (see pipeline/tracking.py). Reports and charts are built
only from that ledger and the files it points to.
"""
