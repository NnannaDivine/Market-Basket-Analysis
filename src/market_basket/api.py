from __future__ import annotations

import ast
import csv
import json
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "artifacts" / "rules.csv"
REQUIRED_COLUMNS = {
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift",
}


class Recommendation(TypedDict):
    based_on: list[str]
    buy_with: list[str]
    confidence: float
    lift: float
    support: float


@dataclass
class RecommendationCatalog:
    products: list[str]
    rule_count: int
    recommendations: dict[str, list[Recommendation]]

    def recommend(
        self,
        product: str,
        *,
        min_confidence: float = 0.0,
        min_lift: float = 0.0,
        sort_by: Literal["confidence", "lift"] = "confidence",
        limit: int = 10,
    ) -> list[Recommendation]:
        matches = [
            recommendation
            for recommendation in self.recommendations.get(
                product.strip().casefold(), []
            )
            if recommendation["confidence"] >= min_confidence
            and recommendation["lift"] >= min_lift
        ]
        matches.sort(
            key=lambda recommendation: (
                recommendation[sort_by],
                recommendation["confidence" if sort_by == "lift" else "lift"],
            ),
            reverse=True,
        )
        return matches[:limit]


def parse_itemset(value: str) -> list[str]:
    """Read JSON itemsets and legacy frozenset strings without evaluating code."""
    if not isinstance(value, str):
        raise ValueError(f"Expected an itemset string, got {value!r}")
    raw = value.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        if raw.startswith("frozenset(") and raw.endswith(")"):
            raw = raw[len("frozenset(") : -1]
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid itemset in rules file: {value!r}") from exc

    if not isinstance(parsed, (list, tuple, set, frozenset)):
        raise ValueError(f"Expected an itemset in rules file, got {value!r}")
    if any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"Itemsets must contain only product names: {value!r}")
    return sorted(set(parsed), key=str.casefold)


def load_catalog(path: Path = RULES_PATH) -> RecommendationCatalog:
    if not path.is_file():
        raise FileNotFoundError(
            f"Rules file not found at {path}. Run `python run.py --train` first."
        )

    recommendations: dict[str, list[Recommendation]] = {}
    products_by_key: dict[str, str] = {}
    rule_count = 0

    with path.open(newline="", encoding="utf-8") as rules_file:
        reader = csv.DictReader(rules_file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"Rules file {path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                antecedents = parse_itemset(row["antecedents"])
                consequents = parse_itemset(row["consequents"])
                support = float(row["support"])
                confidence = float(row["confidence"])
                lift = float(row["lift"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid rule data on line {line_number} of {path}: {exc}"
                ) from exc

            if not all(
                math.isfinite(metric) for metric in (support, confidence, lift)
            ):
                raise ValueError(f"Non-finite rule metric on line {line_number} of {path}")
            if not 0 <= support <= 1 or not 0 <= confidence <= 1 or lift < 0:
                raise ValueError(f"Invalid rule metric on line {line_number} of {path}")
            if not antecedents or not consequents:
                raise ValueError(f"Empty itemset on line {line_number} of {path}")

            recommendation: Recommendation = {
                "based_on": antecedents,
                "buy_with": consequents,
                "confidence": confidence,
                "lift": lift,
                "support": support,
            }
            rule_count += 1
            for antecedent in antecedents:
                product_key = antecedent.casefold()
                products_by_key.setdefault(product_key, antecedent)
                recommendations.setdefault(product_key, []).append(recommendation)
            for product in consequents:
                products_by_key.setdefault(product.casefold(), product)

    for product_recommendations in recommendations.values():
        product_recommendations.sort(
            key=lambda result: (result["confidence"], result["lift"]),
            reverse=True,
        )

    return RecommendationCatalog(
        products=sorted(products_by_key.values(), key=str.casefold),
        rule_count=rule_count,
        recommendations=recommendations,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.catalog = load_catalog()
    yield


app = FastAPI(
    title="Market Basket Analysis API",
    description="Product recommendations mined from market basket transactions.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_catalog(request: Request) -> RecommendationCatalog:
    return request.app.state.catalog


@app.get("/", tags=["status"])
def home() -> dict[str, str]:
    return {"message": "Market Basket API is working", "docs": "/docs"}


@app.get("/health", tags=["status"])
def health(request: Request) -> dict[str, int | str]:
    catalog = get_catalog(request)
    return {
        "status": "ok",
        "products": len(catalog.products),
        "rules": catalog.rule_count,
    }


@app.get("/stats", tags=["recommendations"])
def get_stats(request: Request) -> dict[str, int]:
    catalog = get_catalog(request)
    return {"products": len(catalog.products), "rules": catalog.rule_count}


@app.get("/products", tags=["recommendations"])
def get_products(request: Request) -> dict[str, list[str]]:
    return {"products": get_catalog(request).products}


@app.get("/recommend", tags=["recommendations"])
def recommend(
    request: Request,
    product: str = Query(..., min_length=1, description="Product to recommend alongside"),
    min_confidence: float = Query(0.0, ge=0, le=1),
    min_lift: float = Query(0.0, ge=0),
    sort_by: Literal["confidence", "lift"] = Query("confidence"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, str | list[Recommendation]]:
    product = product.strip()
    if not product:
        raise HTTPException(status_code=422, detail="Product must not be blank.")
    return {
        "product": product,
        "recommendations": get_catalog(request).recommend(
            product,
            min_confidence=min_confidence,
            min_lift=min_lift,
            sort_by=sort_by,
            limit=limit,
        ),
    }
