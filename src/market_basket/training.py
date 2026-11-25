from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "basket_analysis.csv"
DEFAULT_RULES_PATH = ROOT / "artifacts" / "rules.csv"
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.5
RULE_COLUMNS = ["antecedents", "consequents", "support", "confidence", "lift"]


def train_rules(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_RULES_PATH,
) -> int:
    """Mine association rules and atomically replace the API's rules artifact."""
    data_path = Path(data_path)
    output_path = Path(output_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Transaction data not found: {data_path}")

    transactions = pd.read_csv(data_path)
    if transactions.empty or len(transactions.columns) == 0:
        raise ValueError(f"Transaction data is empty: {data_path}")
    if transactions.isna().any().any():
        raise ValueError(f"Transaction data contains missing values: {data_path}")

    for column in transactions.columns:
        values = set(transactions[column].unique())
        if not values.issubset({True, False, 1, 0}):
            raise ValueError(
                f"Column {column!r} must contain only boolean or 0/1 values."
            )
    transactions = transactions.astype(bool)

    frequent_itemsets = apriori(
        transactions,
        min_support=MIN_SUPPORT,
        use_colnames=True,
    )
    if frequent_itemsets.empty:
        raise ValueError(
            f"No frequent itemsets found at minimum support {MIN_SUPPORT}."
        )

    mined_rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )
    if mined_rules.empty:
        raise ValueError(
            f"No association rules found at minimum confidence {MIN_CONFIDENCE}."
        )

    rules = mined_rules[RULE_COLUMNS].copy()
    rules["antecedents"] = rules["antecedents"].map(
        lambda items: json.dumps(sorted(items, key=str.casefold))
    )
    rules["consequents"] = rules["consequents"].map(
        lambda items: json.dumps(sorted(items, key=str.casefold))
    )
    rules = rules.sort_values(
        ["confidence", "lift"], ascending=False
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            rules.to_csv(temporary_file, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return len(rules)
