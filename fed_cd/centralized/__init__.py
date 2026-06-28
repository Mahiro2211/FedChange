"""Centralized (non-federated) change detection training.

Provides the performance upper bound for federated experiments. Mirrors
``fed_cd.federated.fed_main`` but trains a single model on the union of all
client data (the same sample pool the federated partitions draw from).
"""
