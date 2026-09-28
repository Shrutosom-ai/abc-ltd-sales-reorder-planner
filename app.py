"""
ABC Ltd · Sales & Reorder Planner
=================================
A Streamlit app for managers. For a planned dispatch it answers two questions:
  1. How much of it will sell?                       -> linear regression
  2. Will stock fall below the minimum (reorder)?    -> logistic regression

The models are trained in the Colab notebook, which saves them to abc_ltd_models.joblib.
Run on your own computer:   pip install -r requirements.txt   then   streamlit run app.py
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Paste the link to your feedback form (for example a Google Form) for the user study.
FEEDBACK_FORM_URL = ""

MODEL_FILE = Path(__file__).parent / "abc_ltd_models.joblib"
BLUE, ORANGE, GREY = "#2a6fb0", "#d9480f", "#8a8a8a"

st.set_page_config(page_title="ABC Ltd · Sales & Reorder Planner", page_icon="📦", layout="wide")


# ----------------------------------------------------------------------------- load models
@st.cache_resource
def load_models():
    return joblib.load(MODEL_FILE)


try:
    bundle = load_models()
except Exception as err:  # file missing, or library versions differ from the Colab run
    st.error(
        "Could not load **abc_ltd_models.joblib**. Check that it is in the GitHub repository next to "
        "app.py, and that requirements.txt came from the same Colab run as the model file."
    )
    st.exception(err)
    st.stop()

sales_model = bundle["sales_model"]
reorder_model = bundle["reorder_model"]
meta = bundle["meta"]
S, R = meta["sales"], meta["reorder"]
ALERT = R["alert_cutoff"]
Q_MAX = int(np.ceil(meta["qty_range"][1] / 10) * 10)


# ----------------------------------------------------------------------------- prediction helpers
def expected_sales(qty):
    """Linear regression: expected quantity sold, kept between 0 and the quantity available."""
    qty = np.atleast_1d(np.asarray(qty, dtype=float))
    pred = sales_model.predict(pd.DataFrame({"qty_available": qty}))
    return np.clip(pred, 0, qty)


def likely_range(qty, pred):
    """Range that held 8 in 10 actual results for similar quantities in the training data."""
    qty = np.atleast_1d(np.asarray(qty, dtype=float))
    low = np.clip(pred + np.interp(qty, S["range_qty"], S["range_low"]), 0, qty)
    high = np.clip(pred + np.interp(qty, S["range_qty"], S["range_high"]), 0, qty)
    return low, high


def reorder_chance(qty, min_stock):
    """Logistic regression: probability that closing stock ends below the minimum level.
    Business rule on top: if no more than the minimum level is made available, a reorder is certain."""
    qty = np.atleast_1d(np.asarray(qty, dtype=float))
    X = pd.DataFrame({"qty_available": qty, "min_stock": np.full_like(qty, float(min_stock))})
    chance = reorder_model.predict_proba(X)[:, 1]
    return np.where(qty <= min_stock, 1.0, chance)


def pct(p):
    """Format a probability for managers: '<1%' rather than '0%'."""
    return "less than 1%" if p < 0.005 else f"{p:.0%}"


# ----------------------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Plan a dispatch")
    qty = st.slider(
        "Quantity made available (L/kg)", min_value=10, max_value=Q_MAX, value=400, step=10,
        help="How much of the product you will stock or send to this market.",
    )
    min_stock = st.slider(
        "Minimum stock level (L/kg)", min_value=10, max_value=100,
        value=int(round(meta["defaults"]["min_stock"] / 5) * 5), step=5,
        help="The level below which the outlet or depot has to reorder.",
    )
    price = st.number_input(
        "Selling price (₹ per L/kg)", min_value=1.0, max_value=2000.0, value=55.0, step=1.0,
        help="Used only to put a rupee value on the forecast. In ABC Ltd's records, "
             "price had no measurable effect on the quantity sold.",
    )
    st.divider()
    st.caption(
        "**Why only these inputs?** Price, product, brand, channel, region and month were all tested. "
        "In ABC Ltd's records none of them changed sales or reorders by a meaningful amount. "
        "See *Model accuracy* for details."
    )

# ----------------------------------------------------------------------------- predictions
sold_arr = expected_sales(qty)
low_arr, high_arr = likely_range(qty, sold_arr)
sold, low, high = float(sold_arr[0]), float(low_arr[0]), float(high_arr[0])
closing = qty - sold
chance = float(reorder_chance(qty, min_stock)[0])

grid = np.arange(10, Q_MAX + 1, 10, dtype=float)
grid_sold = expected_sales(grid)
grid_low, grid_high = likely_range(grid, grid_sold)
grid_chance = reorder_chance(grid, min_stock)
safe = grid[grid_chance < ALERT]
safe_qty = float(safe[0]) if len(safe) else None  # smallest quantity below the alert level
SHARES = expected_sales([100, 500, 1000]) / np.array([100, 500, 1000])

# ----------------------------------------------------------------------------- header and results
st.title("ABC Ltd · Sales & Reorder Planner")
st.caption(
    f"Forecasts built on {meta['rows_clean']:,} ABC Ltd sales and inventory records (2019–2022). "
    "Linear regression estimates how much will sell; logistic regression estimates the chance "
    "that stock falls below the minimum level."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Expected quantity sold", f"{sold:,.0f} L/kg", border=True,
              help="Linear regression forecast, tested on 2022 data.")
    st.caption(f"Likely range {low:,.0f}–{high:,.0f} L/kg (8 in 10 similar dispatches)")
with c2:
    st.metric("Expected closing stock", f"{closing:,.0f} L/kg", border=True)
    st.caption(f"{closing / qty:.0%} of what you make available stays unsold")
with c3:
    if qty <= min_stock:
        level = "Certain (no more than the minimum is available)"
    else:
        level = "High" if chance >= 0.5 else ("Medium" if chance >= ALERT else "Low")
    st.metric("Chance of a reorder", "<1%" if chance < 0.005 else f"{chance:.0%}", border=True,
              help=f"Logistic regression. An alert is raised at {ALERT:.0%} or above.")
    st.caption(f"Risk level: **{level}**")
with c4:
    st.metric("Expected sales value", f"₹{sold * price:,.0f}", border=True)
    st.caption(f"At ₹{price:,.0f} per L/kg")


def safe_advice():
    if safe_qty is None:
        return "Even the largest quantity in the data keeps the chance above the alert level, so plan a reorder."
    unsold_then = safe_qty - float(expected_sales(safe_qty)[0])
    return (f"Making at least **{safe_qty:,.0f} L/kg** available brings it under the alert level, "
            f"with about {unsold_then:,.0f} L/kg expected to stay unsold.")


if qty <= min_stock:
    st.error(f"**Reorder certain.** You are making available no more than the minimum stock level of "
             f"{min_stock} L/kg, so stock falls to or below it as soon as anything sells. {safe_advice()}")
elif chance >= 0.5:
    st.error(f"**Reorder likely.** The chance that closing stock ends below the minimum level of {min_stock} L/kg "
             f"is {pct(chance)}. {safe_advice()}")
elif chance >= ALERT:
    st.warning(f"**Reorder alert.** The chance that closing stock ends below the minimum level of {min_stock} L/kg "
               f"is {pct(chance)}, above the alert level of {ALERT:.0%}. {safe_advice()}")
else:
    st.success(f"**No reorder alert.** The chance that closing stock ends below the minimum level of {min_stock} L/kg "
               f"is {pct(chance)}, under the alert level of {ALERT:.0%}.")

# ----------------------------------------------------------------------------- tabs
tab_whatif, tab_acc, tab_data, tab_fb = st.tabs(
    ["What if I change the quantity?", "Model accuracy", "About the data", "Feedback"]
)

with tab_whatif:
    curve = pd.DataFrame({"qty": grid, "sold": grid_sold, "low": grid_low, "high": grid_high,
                          "chance": grid_chance})
    now = pd.DataFrame({"qty": [qty], "sold": [sold], "chance": [chance]})
    x = alt.X("qty:Q", title="Quantity made available (L/kg)")
    tips = [alt.Tooltip("qty:Q", title="Available", format=",.0f"),
            alt.Tooltip("sold:Q", title="Expected sold", format=",.0f"),
            alt.Tooltip("low:Q", title="Likely low", format=",.0f"),
            alt.Tooltip("high:Q", title="Likely high", format=",.0f"),
            alt.Tooltip("chance:Q", title="Reorder chance", format=".0%")]

    left, right = st.columns(2)
    with left:
        band = alt.Chart(curve).mark_area(opacity=0.2, color=BLUE).encode(
            x=x, y=alt.Y("low:Q", title="Quantity sold (L/kg)"), y2="high:Q")
        line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=3).encode(x=x, y="sold:Q", tooltip=tips)
        marker = alt.Chart(now).mark_rule(color=ORANGE, strokeDash=[4, 4]).encode(x="qty:Q")
        dot = alt.Chart(now).mark_point(filled=True, size=100, color=ORANGE).encode(x="qty:Q", y="sold:Q")
        st.altair_chart(alt.layer(band, line, marker, dot).properties(
            height=330, title="Expected sales (shaded: likely range)"), width="stretch")
    with right:
        y = alt.Y("chance:Q", title="Chance of a reorder", axis=alt.Axis(format="%"),
                  scale=alt.Scale(domain=[0, 1]))
        risk = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=3).encode(x=x, y=y, tooltip=tips)
        alert_df = pd.DataFrame({"qty": [Q_MAX * 0.7], "chance": [ALERT], "label": [f"Alert level {ALERT:.0%}"]})
        alert_rule = alt.Chart(alert_df).mark_rule(color=GREY, strokeDash=[2, 3]).encode(y="chance:Q")
        alert_text = alt.Chart(alert_df).mark_text(align="left", dy=-8, color=GREY).encode(
            x="qty:Q", y="chance:Q", text="label:N")
        marker = alt.Chart(now).mark_rule(color=ORANGE, strokeDash=[4, 4]).encode(x="qty:Q")
        dot = alt.Chart(now).mark_point(filled=True, size=100, color=ORANGE).encode(x="qty:Q", y="chance:Q")
        st.altair_chart(alt.layer(risk, alert_rule, alert_text, marker, dot).properties(
            height=330, title=f"Chance of a reorder at minimum stock {min_stock} L/kg"), width="stretch")

    st.markdown("**Quick reference** (at your minimum stock level)")
    tq = np.arange(100, Q_MAX + 1, 100, dtype=float)
    ts = expected_sales(tq)
    table = pd.DataFrame({
        "Quantity made available (L/kg)": tq,
        "Expected sold (L/kg)": ts,
        "Share sold (%)": 100 * ts / tq,
        "Expected closing stock (L/kg)": tq - ts,
        "Chance of a reorder (%)": 100 * reorder_chance(tq, min_stock),
    })
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in table.columns})

with tab_acc:
    st.subheader("How far to trust the forecasts")
    st.caption(f"Both models learned from {meta['rows_train']:,} records (2019–2021) and were tested on "
               f"{meta['rows_test']:,} records from 2022 that they had never seen.")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Sales forecast · linear regression")
        st.markdown(
            f"- Explains **{S['r2']:.0%}** of the ups and downs in quantity sold (R² = {S['r2']:.2f}).\n"
            f"- Typical miss: **±{S['mae']:.0f} L/kg** (mean absolute error).\n"
            f"- The likely range contained the actual figure for **{S['range_coverage']:.0%}** of 2022 dispatches.\n"
            f"- Expected share sold: about **{SHARES[0]:.0%}** at 100 L/kg, **{SHARES[1]:.0%}** at 500 L/kg "
            f"and **{SHARES[2]:.0%}** at 1,000 L/kg."
        )
    with b:
        st.markdown("#### Reorder alert · logistic regression")
        st.markdown(
            f"- Given one dispatch that needed a reorder and one that did not, it gives the first a higher "
            f"chance **{R['auc']:.0%}** of the time (AUC = {R['auc']:.2f}).\n"
            f"- At the alert level of {ALERT:.0%}, it caught **{R['recall']:.0%}** of the reorders in 2022, "
            f"and **{R['precision']:.0%}** of its alerts were right.\n"
            f"- Each extra 100 L/kg made available cuts the odds of a reorder by "
            f"**{1 - R['odds_per_100_qty']:.0%}**.\n"
            f"- Each extra 10 L/kg of minimum stock raises the odds by **{R['odds_per_10_min_stock'] - 1:.0%}**.\n"
            f"- Common-sense rule on top of the model: if no more than the minimum stock level is made "
            f"available, a reorder is shown as certain."
        )
    st.markdown("#### What the models leave out, and why")
    st.markdown(
        f"These inputs were tested and made no meaningful difference in ABC Ltd's records: "
        f"{', '.join(meta['no_effect_inputs'])}. A sales model using all of them scored R² = "
        f"{S['r2_all_inputs']:.2f}, no better than quantity alone, and a reorder model using all of them "
        f"scored AUC = {R['auc_all_inputs']:.2f}. So the tool asks only for what matters."
    )

with tab_data:
    st.subheader("About the data")
    st.markdown(
        f"- **Source:** ABC Ltd dairy sales and inventory records, 2019–2022, across 15 Indian states, "
        f"10 products and 3 sales channels (company and brand names anonymised).\n"
        f"- **Cleaning:** {meta['rows_removed']:,} of {meta['rows_raw']:,} records were removed because they broke "
        f"the basic stock rule: more was sold, or more was left over, than was available. "
        f"{meta['rows_clean']:,} records remain.\n"
        f"- **Training and testing:** trained on {meta['train_period']}, tested on {meta['test_period']}.\n"
        f"- **Reorder** means closing stock ended below the minimum stock level."
    )
    st.markdown("**Limitations**")
    st.markdown(
        "- The records contain values a real business would not produce, such as expiry dates before "
        "production dates and every product priced across the same ₹10–₹100 range. Treat the tool as a "
        "demonstration of the method, not as a tested operating policy.\n"
        f"- The sales forecast explains about {S['r2']:.0%} of the variation in quantity sold. Plan with the "
        "likely range, not the single number.\n"
        "- The alert level trades false alarms for catching more real reorders."
    )

with tab_fb:
    st.subheader("Try it, then tell us what you think")
    st.markdown(
        "1. You plan to make **200 L/kg** available with a minimum stock level of **60 L/kg**. "
        "Would you expect to reorder?\n"
        "2. For that minimum level, what is the smallest quantity that avoids a reorder alert?\n"
        "3. If you make **800 L/kg** available, how much is likely to stay unsold?\n"
        "4. Would you use a tool like this when planning dispatches? What would you change?"
    )
    if FEEDBACK_FORM_URL:
        st.link_button("Open the feedback form", FEEDBACK_FORM_URL, type="primary")
    else:
        st.info("Feedback form link not added yet. Paste it into FEEDBACK_FORM_URL at the top of app.py.")

st.divider()
st.caption("ABC Ltd · Sales & Reorder Planner · linear and logistic regression · data anonymised")
