import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from matplotlib.ticker import PercentFormatter


API_URL = os.getenv("MARKET_BASKET_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Basketwise | Retail Insights",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(120, 130, 150, .16);}
    .eyebrow {
        color: #0f766e;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .hero {
        background: linear-gradient(115deg, #102a43 0%, #174c57 68%, #167d77 100%);
        border-radius: 18px;
        color: #ffffff;
        margin: .5rem 0 1.5rem;
        padding: 1.6rem 1.8rem;
    }
    .hero h1 {color: #ffffff; font-size: 2rem; margin: 0 0 .45rem;}
    .hero p {color: #d7e8e8; font-size: 1rem; margin: 0; max-width: 760px;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid rgba(120, 130, 150, .18);
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_json(path: str, params: dict[str, object] | None = None) -> dict:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=8)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=45)
def fetch_catalog() -> tuple[list[str], dict[str, int]]:
    products = get_api_json("/products")["products"]
    stats = get_api_json("/stats")
    return products, stats


@st.cache_data(ttl=30)
def fetch_recommendations(
    product: str,
    min_confidence: float,
    min_lift: float,
    sort_by: str,
    limit: int,
) -> list[dict[str, object]]:
    result = get_api_json(
        "/recommend",
        params={
            "product": product,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
            "sort_by": sort_by,
            "limit": limit,
        },
    )
    return result["recommendations"]


with st.sidebar:
    st.markdown("## 🛒 Basketwise")
    st.caption("Retail association explorer")
    st.divider()

    try:
        products, stats = fetch_catalog()
    except requests.RequestException as exc:
        st.error("The recommendation service is not available.")
        st.caption("Start or restart the project with `python run.py`.")
        st.caption(f"Connection detail: {exc}")
        st.stop()

    if not products:
        st.warning("No product data is available. Run `python run.py --train`.")
        st.stop()

    selected_product = st.selectbox("Product in the basket", products)
    st.markdown("#### Recommendation filters")
    confidence_percent = st.slider(
        "Minimum confidence",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        format="%d%%",
        help="How often the recommended item appears when the rule's trigger set is present.",
    )
    min_lift = st.slider(
        "Minimum lift",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Lift above 1 indicates a positive association versus the item's baseline frequency.",
    )
    sort_label = st.selectbox("Rank results by", ["Confidence", "Lift"])
    result_limit = st.slider(
        "Maximum recommendations",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )
    st.divider()
    st.caption("Confidence and lift describe observed associations, not guaranteed sales.")

sort_by = sort_label.casefold()
try:
    recommendations = fetch_recommendations(
        selected_product,
        confidence_percent / 100,
        min_lift,
        sort_by,
        result_limit,
    )
except requests.RequestException as exc:
    st.error("Could not retrieve recommendations from the API.")
    st.caption(f"Connection detail: {exc}")
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow" style="color:#9fe2d7">MERCHANDISING INTELLIGENCE</div>
      <h1>Turn baskets into better pairings.</h1>
      <p>Explore the products shoppers buy together and uncover evidence-led
      cross-selling opportunities for your store.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

health_col, _ = st.columns([1, 4])
with health_col:
    st.success("Recommendation engine online", icon="✅")

metric_columns = st.columns(3)
metric_columns[0].metric("Products in catalog", f"{stats['products']:,}")
metric_columns[1].metric("Association rules", f"{stats['rules']:,}")
metric_columns[2].metric("Selected product", selected_product)

recommendation_tab, methodology_tab = st.tabs(
    ["Explore recommendations", "How to read the results"]
)

with recommendation_tab:
    st.subheader(f"Pairings for {selected_product}")
    st.caption(
        f"Showing rules with confidence ≥ {confidence_percent}% and lift ≥ "
        f"{min_lift:.1f}, ranked by {sort_label.lower()}."
    )

    if not recommendations:
        st.info(
            "No rules meet these filters. Lower the confidence or lift threshold "
            "in the sidebar to broaden the results."
        )
    else:
        average_confidence = sum(
            float(rule["confidence"]) for rule in recommendations
        ) / len(recommendations)
        best_lift = max(float(rule["lift"]) for rule in recommendations)
        result_columns = st.columns(3)
        result_columns[0].metric("Matching rules", f"{len(recommendations):,}")
        result_columns[1].metric(
            "Average confidence", f"{average_confidence:.0%}"
        )
        result_columns[2].metric("Strongest lift", f"{best_lift:.2f}×")

        rows = [
            {
                "Rule trigger": " + ".join(rule["based_on"]),
                "Recommended pairing": " + ".join(rule["buy_with"]),
                "Confidence": float(rule["confidence"]),
                "Lift": float(rule["lift"]),
                "Basket support": float(rule["support"]),
            }
            for rule in recommendations
        ]
        results = pd.DataFrame(rows)

        table_column, chart_column = st.columns([1.5, 1], gap="large")
        with table_column:
            st.markdown("#### Recommendation detail")
            st.dataframe(
                results,
                hide_index=True,
                column_config={
                    "Rule trigger": st.column_config.TextColumn(
                        help="Products that must be present for this association rule."
                    ),
                    "Recommended pairing": st.column_config.TextColumn(),
                    "Confidence": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=1,
                        format="percent",
                    ),
                    "Lift": st.column_config.NumberColumn(format="%.2f×"),
                    "Basket support": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=1,
                        format="percent",
                    ),
                },
            )
            st.download_button(
                "Download filtered rules",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_product.lower().replace(' ', '-')}-recommendations.csv",
                mime="text/csv",
            )

        with chart_column:
            st.markdown("#### Confidence by association rule")
            chart_rows = []
            for rule in recommendations:
                context = " + ".join(
                    item
                    for item in rule["based_on"]
                    if item.casefold() != selected_product.casefold()
                ) or selected_product
                chart_rows.append(
                    (
                        f"{' + '.join(rule['buy_with'])}  ·  {context}",
                        float(rule["confidence"]),
                    )
                )
            chart_rows.sort(key=lambda row: row[1])
            labels, confidence_values = zip(*chart_rows)
            figure, axis = plt.subplots(
                figsize=(6.2, max(3.2, len(chart_rows) * 0.42))
            )
            bars = axis.barh(
                labels,
                confidence_values,
                color="#167d77",
                height=0.58,
            )
            axis.set_xlim(0, 1.08)
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.set_xlabel("Confidence")
            axis.set_ylabel("")
            axis.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.25)
            axis.set_axisbelow(True)
            axis.bar_label(
                bars,
                labels=[f"{value:.0%}" for value in confidence_values],
                padding=5,
                fontsize=9,
                color="#294152",
            )
            for spine in ("top", "right", "left"):
                axis.spines[spine].set_visible(False)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
            st.caption(
                "Higher confidence means the pairing appeared more often in "
                "transactions that matched the rule trigger."
            )

with methodology_tab:
    st.subheader("What these numbers mean")
    definition_columns = st.columns(3)
    definition_columns[0].markdown(
        "**Support**  \n"
        "The share of all baskets that contain both the rule trigger and "
        "the recommended item."
    )
    definition_columns[1].markdown(
        "**Confidence**  \n"
        "Among baskets containing the trigger, the share that also contain "
        "the recommended item."
    )
    definition_columns[2].markdown(
        "**Lift**  \n"
        "How much more often the pairing occurs than expected from the "
        "recommended item's overall frequency. Values above 1 indicate a "
        "positive association."
    )
    st.info(
        "Rules are generated with Apriori from the one-hot transaction dataset. "
        "These are associations, not evidence that placing or discounting a "
        "product will cause additional sales."
    )
