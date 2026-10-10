"""A small neural network to explore Norway's interest rate and NOK exchange rate.

Run: streamlit run nok_neural_network.py
Requires: streamlit, numpy, pandas, matplotlib

The learning algorithm is written entirely in NumPy. The data are daily
Norges Bank observations republished as convenient CSV files by frefrik.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import streamlit as st


# ---------- 1. Get two real time series ----------
REPOSITORY = "https://raw.githubusercontent.com/frefrik/norway-finance-statistics/master/data/"
RATE_URL = REPOSITORY + "no_keyPolicyRate.csv"
EUR_URL = REPOSITORY + "no_exchangeRates.csv"
FEATURE_NAMES = ("Policy rate (%)", "EUR/NOK change this month (%)")
HERE = Path(__file__).resolve().parent


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading Norges Bank data …")
def get_daily_series():
    """Prefer nearby CSV files (offline); otherwise download both CSVs."""
    local_rate = HERE / "no_keyPolicyRate.csv"
    local_eur = HERE / "no_exchangeRates.csv"
    rate_source = local_rate if local_rate.is_file() else RATE_URL
    eur_source = local_eur if local_eur.is_file() else EUR_URL

    rate = pd.read_csv(rate_source, usecols=["Date", "Rate"])
    eur = pd.read_csv(eur_source, usecols=["Date", "EUR"])
    rate["Date"] = pd.to_datetime(rate["Date"], errors="coerce")
    eur["Date"] = pd.to_datetime(eur["Date"], errors="coerce")
    rate["Rate"] = pd.to_numeric(rate["Rate"], errors="coerce")
    eur["EUR"] = pd.to_numeric(eur["EUR"], errors="coerce")
    return rate.dropna(), eur.dropna()


def make_monthly(rate, eur):
    """Create monthly averages and a next-month classification target.

    The explanatory variables are known at the END of month t:
    monthly policy rate, and EUR/NOK percent change since previous month.
    The target is whether EUR/NOK rises in month t+1 (NOK weakens).
    """
    rate = rate.copy()
    eur = eur.copy()
    rate["month"] = rate["Date"].dt.to_period("M").dt.to_timestamp()
    eur["month"] = eur["Date"].dt.to_period("M").dt.to_timestamp()

    r = rate.groupby("month")["Rate"].mean().rename("rate_pct")
    x = eur.groupby("month")["EUR"].mean().rename("eur_nok")
    monthly = pd.concat([r, x], axis=1).sort_index()

    # Keep a complete calendar. Missing months remain missing, not compressed.
    calendar = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(calendar)

    # Exclude the current, incomplete calendar month.
    current_month = pd.Timestamp.now().to_period("M").to_timestamp()
    monthly = monthly.loc[monthly.index < current_month].copy()

    monthly["current_change_pct"] = monthly["eur_nok"].div(
        monthly["eur_nok"].shift(1)
    ).sub(1).mul(100)
    monthly["next_change_pct"] = monthly["eur_nok"].shift(-1).div(
        monthly["eur_nok"]
    ).sub(1).mul(100)
    monthly["depreciates_next_month"] = (
        monthly["next_change_pct"] > 0
    ).astype(int)
    return monthly


# ---------- 2. The neural network (just NumPy) ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def forward(X, weights):
    W1, b1, W2, b2 = weights
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)  # ReLU activation
    z2 = h @ W2 + b2
    p = sigmoid(z2)         # 1 = NOK depreciates next month
    return z1, h, z2, p


def binary_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradients(X, Y, weights):
    W1, b1, W2, b2 = weights
    z1, h, _z2, p = forward(X, weights)
    d2 = (p - Y) / len(X)             # sigmoid + cross-entropy derivative
    d1 = (d2 @ W2.T) * (z1 > 0)      # backpropagate through ReLU
    gW2 = h.T @ d2
    gb2 = d2.sum(axis=0)
    gW1 = X.T @ d1
    gb1 = d1.sum(axis=0)
    return gW1, gb1, gW2, gb2


def train_network(X, Y, epochs=300, lr=0.1, hidden=8):
    """Full-batch gradient descent; store weights after EVERY epoch."""
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((2, hidden)) * 0.4
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 1)) * 0.4
    b2 = np.zeros(1)
    weights = (W1, b1, W2, b2)
    snapshots, losses = [], []
    for epoch in range(epochs + 1):
        # Snapshot 0 is the network BEFORE any training.
        snapshots.append(tuple(w.copy() for w in weights))
        losses.append(binary_loss(Y, forward(X, weights)[3]))
        if epoch == epochs:
            break
        gW1, gb1, gW2, gb2 = gradients(X, Y, weights)
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2
    return snapshots, np.asarray(losses)


def class_probability(X, weights):
    return forward(X, weights)[3].ravel()


# ---------- 3. Illustrations ----------
def decision_plot(X_raw, y, train_cut, mean, scale, weights, selected):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    xmin, xmax = np.quantile(X_raw[:, 0], [0.01, 0.99]); xmin -= 0.5; xmax += 0.5
    ymin, ymax = np.quantile(X_raw[:, 1], [0.01, 0.99]); ymin -= 0.65; ymax += 0.65
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 110), np.linspace(ymin, ymax, 110))
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    probs = class_probability((mesh - mean) / scale, weights).reshape(xx.shape)
    surf = ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.49)
    if np.min(probs) < 0.5 < np.max(probs):
        ax.contour(xx, yy, probs, levels=[0.5], colors=["#222222"], linewidths=1.6)
    for val, color, label in [(0, "#276EB3", "NOK strengthens next month"),
                              (1, "#B13B35", "NOK weakens next month")]:
        mask = (y[:train_cut] == val)
        ax.scatter(X_raw[:train_cut, 0][mask], X_raw[:train_cut, 1][mask],
                   c=color, s=19, marker="o", edgecolors="white", linewidth=0.3,
                   alpha=0.75, label=label)
    for val, color in [(0, "#276EB3"), (1, "#B13B35")]:
        mask = y[train_cut:] == val
        ax.scatter(X_raw[train_cut:, 0][mask], X_raw[train_cut:, 1][mask],
                   c=color, marker="x", s=32, linewidths=1.1, alpha=0.8)
    ax.scatter(X_raw[selected, 0], X_raw[selected, 1], c="#111111", marker="*",
               s=230, edgecolors="white", linewidth=1.0, zorder=6, label="Selected month")
    cb = fig.colorbar(surf, ax=ax, pad=0.02)
    cb.set_label("Model P(NOK depreciates next month)")
    ax.set(xlabel=FEATURE_NAMES[0], ylabel=FEATURE_NAMES[1],
           title="Decision surface — dots = training, × = later observations")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def network_plot(x, weights, selected_hidden):
    W1, b1, W2, b2 = weights
    z1, h, _z2, p = forward(x.reshape(1, -1), weights)
    n = W1.shape[1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.65 * n + 1)))
    inp_y = np.array([0.75, 0.25]); hid_y = np.linspace(0.94, 0.06, n)
    out_y = 0.5
    def edge(xa, ya, xb, yb, w):
        ax.plot([xa, xb], [ya, yb], color=("#218B73" if w >= 0 else "#D07634"),
                linewidth=0.4 + min(3.2, abs(w) * 1.8), alpha=0.40, zorder=1)
    for i in range(2):
        for j in range(n):
            edge(0, inp_y[i], 1, hid_y[j], W1[i, j])
    for j in range(n):
        edge(1, hid_y[j], 2, out_y, W2[j, 0])
    for i in range(2):
        ax.scatter(0, inp_y[i], s=950, c="#377EB8", edgecolors="white", zorder=3)
        ax.text(0, inp_y[i], f"x{i+1}\n{x[i]:+.2f}", color="white", ha="center", va="center", fontsize=9, zorder=5)
    for j in range(n):
        active = h[0, j] > 0
        col = "#159979" if active else "#9AA4AE"
        outline = "#141414" if j == selected_hidden else "white"
        ax.scatter(1, hid_y[j], s=(800 if j == selected_hidden else 620),
                   c=col, edgecolors=outline, linewidths=(2.4 if j == selected_hidden else 1), zorder=3)
        ax.text(1, hid_y[j], f"h{j+1}\n{h[0,j]:.2f}", color="white", ha="center", va="center", fontsize=8, zorder=5)
    ax.scatter(2, out_y, s=1300, c="#B48435", edgecolors="white", zorder=3)
    ax.text(2, out_y, f"p\n{p[0,0]:.2f}", color="white", ha="center", va="center", fontsize=10, zorder=5)
    ax.set_xlim(-0.35, 2.45); ax.set_ylim(-0.04, 1.09); ax.axis("off")
    ax.text(0, 1.04, "2 normalized inputs", ha="center", fontsize=10)
    ax.text(1, 1.04, f"{n} ReLU neurons", ha="center", fontsize=10)
    ax.text(2, 1.04, "Sigmoid output", ha="center", fontsize=10)
    ax.set_title("Green connections: positive weights   ·   Orange: negative weights", fontsize=10, pad=18)
    fig.tight_layout()
    return fig


# ---------- 4. Streamlit interface ----------
st.set_page_config(page_title="NOK Neural Network Lab", page_icon="🧠", layout="wide")
st.title("🧠 NOK neural network lab")
st.caption("A real-data experiment: Can today's interest rate and EUR/NOK movement tell us whether NOK will weaken next month?")
st.info("This is a **learning exercise**, not an economic or currency forecasting tool. "
        "The model has just two explanatory variables, and exchange rates are difficult to predict.")

with st.sidebar:
    st.header("Experiment settings")
    start_year = st.select_slider("Use data from", options=[2000, 2005, 2010, 2015, 2020], value=2005)
    hidden = st.slider("Hidden neurons", 2, 16, 8)
    epochs = st.slider("Training epochs", 0, 500, 300, 10)
    lr = st.slider("Learning rate", 0.01, 1.0, 0.10, 0.01)
    st.caption("Weights start with a fixed random seed (0). Moving an epoch slider does not change the starting weights.")
    if st.button("Refresh source data"):
        get_daily_series.clear()
        st.rerun()

try:
    rate_daily, eur_daily = get_daily_series()
except Exception as exc:
    st.error(f"Could not load the CSV sources: {exc}")
    st.markdown("**Manual fallback:** Download the two CSV files linked below and place them "
                "in the same folder as `nok_neural_network.py`, then restart the app.")
    st.markdown(f"[Policy rate CSV]({RATE_URL}) · [EUR/NOK CSV]({EUR_URL})")
    st.stop()

monthly = make_monthly(rate_daily, eur_daily)
monthly = monthly.loc[monthly.index.year >= start_year].copy()
model_rows = monthly[["rate_pct", "current_change_pct", "next_change_pct",
                      "depreciates_next_month"]].dropna().copy()
if len(model_rows) < 40:
    st.error("Not enough complete monthly observations. Try an earlier start year.")
    st.stop()

X_raw = model_rows[["rate_pct", "current_change_pct"]].to_numpy(dtype=float)
Y_all = model_rows["depreciates_next_month"].to_numpy(dtype=int)
Y = Y_all.reshape(-1, 1).astype(float)
cut = int(0.8 * len(X_raw))  # chronological split; no randomly mixing future with past
mean = X_raw[:cut].mean(axis=0)
scale = X_raw[:cut].std(axis=0)
scale = np.where(scale < 1e-10, 1, scale)
X = (X_raw - mean) / scale

snapshots, losses = train_network(X[:cut], Y[:cut], epochs, lr, hidden)

st.caption(f"{len(X)} months ({model_rows.index.min():%b %Y}–{model_rows.index.max():%b %Y}); "
           f"{cut} earlier months for training, {len(X)-cut} later months for testing. "
           "Monthly averages of daily observations. Data source: Norges Bank, republished by frefrik.")

bar_left, bar_right = st.columns([1.4, 1])
with bar_left:
    shown_epoch = st.slider("Show the network after this many epochs", min_value=0,
                            max_value=epochs, value=epochs, step=1,
                            help="0 is the random, untrained network.")
with bar_right:
    month_index = st.selectbox("Inspect a month", options=list(range(len(model_rows))),
                               index=min(cut, len(model_rows)-1),
                               format_func=lambda i: model_rows.index[i].strftime("%B %Y"))

w = snapshots[shown_epoch]
probabilities = class_probability(X, w)
predictions = (probabilities >= .5).astype(int)
train_acc = (predictions[:cut] == Y_all[:cut]).mean()
test_acc = (predictions[cut:] == Y_all[cut:]).mean()
majority_baseline = max(np.mean(Y_all[cut:]), 1 - np.mean(Y_all[cut:]))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Training accuracy", f"{train_acc:.1%}")
m2.metric("Later-period accuracy", f"{test_acc:.1%}")
m3.metric("Simple test baseline", f"{majority_baseline:.1%}", help="Accuracy from always predicting the most common test-period class.")
m4.metric("P(NOK weakens), chosen month", f"{probabilities[month_index]:.1%}")

view_data, view_boundary, view_neuron, view_learning, view_code = st.tabs(
    ["📈 Economic data", "🗺️ Decision boundary", "🔬 Inside a neuron", "📉 Learning", "⌨️ The NumPy code"]
)

with view_data:
    st.write("Each dot used by the network is a month. Inputs use information available "
             "at the **end of that month**. The label tells us what happened **the following month**.")
    a, b = st.columns(2)
    with a:
        st.markdown("**EUR/NOK (NOK per euro)**")
        st.line_chart(monthly["eur_nok"].rename("EUR/NOK"))
    with b:
        st.markdown("**Norges Bank policy rate (%)**")
        st.line_chart(monthly["rate_pct"].rename("Rate (%)"))
    st.dataframe(model_rows.rename(columns={"rate_pct": "Interest rate (%)",
                                             "current_change_pct": "EUR/NOK change (%)",
                                             "next_change_pct": "Next-month EUR/NOK change (%)",
                                             "depreciates_next_month": "Next month: NOK weakens? (1=yes)"}),
                 use_container_width=True)
    st.download_button("Download prepared monthly dataset (CSV)",
                       data=model_rows.to_csv(index_label="month").encode("utf-8"),
                       file_name="nok_monthly_network_data.csv", mime="text/csv")

with view_boundary:
    st.markdown("The background is the network's predicted probability. "
                "The thick black line, when visible, is its 50% classification boundary. "
                "Circles were used for training; crosses are later months, not used for fitting.")
    fig = decision_plot(X_raw, Y_all, cut, mean, scale, w, month_index)
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.caption("The selected month is marked by a star. Changing the epoch slider shows how the learned surface changes.")

with view_neuron:
    chosen = st.slider("Select hidden neuron", 1, hidden, 1) - 1
    st.markdown("Inputs are standardized using **only the training sample's** mean and standard deviation. "
                "This changes units but not the observations' economic information.")
    xv = X[month_index]
    W1, b1, W2, b2 = w
    z1, h, z2, p = forward(xv.reshape(1, 2), w)
    left, right = st.columns([1.25, 1])
    with left:
        fig = network_plot(xv, w, chosen)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with right:
        st.markdown(f"**Month:** {model_rows.index[month_index]:%B %Y}")
        st.write(f"Actual rate: **{X_raw[month_index, 0]:.3f}%**")
        st.write(f"Actual current-month EUR/NOK change: **{X_raw[month_index, 1]:+.3f}%**")
        st.write(f"Normalized inputs: x₁ = **{xv[0]:+.3f}**, x₂ = **{xv[1]:+.3f}**")
        st.markdown(f"**Neuron {chosen + 1}: weighted sum**")
        st.latex(r"z_j = x_1w_{1j}+x_2w_{2j}+b_j")
        st.code(f"z = ({xv[0]:+.4f} × {W1[0,chosen]:+.4f}) "
                f"+ ({xv[1]:+.4f} × {W1[1,chosen]:+.4f}) "
                f"+ ({b1[chosen]:+.4f})\n"
                f"  = {z1[0,chosen]:+.4f}\n"
                f"ReLU(z) = max(0, z) = {h[0,chosen]:.4f}", language="text")
        st.markdown("**Output:** all hidden activations contribute")
        st.latex(r"p = \sigma\!\left(\sum_j h_jW_{2,j}+b_2\right)")
        st.write(f"Output logit = **{z2[0,0]:+.4f}**; sigmoid(logit) = **{p[0,0]:.4f}**")
        st.write("Observed next month:", "**NOK weakened**" if Y_all[month_index] else "**NOK strengthened or held**")

    st.markdown("**Look inside the matrices**")
    mat = pd.DataFrame({
        "Neuron": np.arange(1, hidden + 1),
        "W1: x1": W1[0], "W1: x2": W1[1], "bias": b1,
        "weighted sum": z1[0], "ReLU output": h[0], "W2 to output": W2[:, 0]
    }).set_index("Neuron")
    st.dataframe(mat.style.format("{:+.4f}"), use_container_width=True)

    st.markdown("**A real gradient update (computed on the training months)**")
    gW1, gb1, gW2, gb2 = gradients(X[:cut], Y[:cut], w)
    dweight = -lr * gW1[0, chosen]
    st.code(f"∂loss/∂W1[0,{chosen}] = {gW1[0,chosen]:+.7f}\n"
            f"update = −learning_rate × gradient = {dweight:+.7f}\n"
            f"weight before = {W1[0,chosen]:+.7f}\n"
            f"weight after  = {W1[0,chosen]+dweight:+.7f}", language="text")
    st.caption("The gradient is an average over ALL training observations. This displays what the next update "
               "would do to the selected weight; it does not train on the inspected test month.")
    if st.button("Check that gradient numerically (finite differences)"):
        eps = 1e-5
        wp = [v.copy() for v in w]; wm = [v.copy() for v in w]
        wp[0][0, chosen] += eps; wm[0][0, chosen] -= eps
        ngrad = (binary_loss(Y[:cut], forward(X[:cut], wp)[3]) -
                 binary_loss(Y[:cut], forward(X[:cut], wm)[3])) / (2*eps)
        st.success(f"Backprop gradient: {gW1[0,chosen]:+.9f}   |   "
                   f"Numerical gradient: {ngrad:+.9f}   |   "
                   f"Absolute difference: {abs(gW1[0,chosen]-ngrad):.2g}")

with view_learning:
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(np.arange(len(losses)), losses, linewidth=2.0)
    ax.axvline(shown_epoch, linestyle="--", color="#CB772F", label=f"Selected epoch: {shown_epoch}")
    ax.set(xlabel="Epoch", ylabel="Training cross-entropy loss", title="Learning by gradient descent")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.write(f"At epoch {shown_epoch}: training loss **{losses[shown_epoch]:.4f}**. "
             "The loss measures fit to the training months, not out-of-sample forecast accuracy.")
    st.markdown("**Training versus later months**")
    chart = pd.DataFrame({"Date": model_rows.index, "Actual": Y_all,
                          "Predicted probability": probabilities,
                          "Sample": np.where(np.arange(len(X)) < cut, "Training", "Later/test")})
    st.dataframe(chart.tail(36), hide_index=True, use_container_width=True)
    st.caption("With small time-series datasets, a network can fit the training period without "
               "predicting unseen months. Try different numbers of hidden neurons and learning rates.")

with view_code:
    st.markdown("**This is the core learning algorithm in the app.** It uses NumPy, "
                "not a ready-made neural-network package.")
    st.code('''W1 = rng.standard_normal((2, hidden)) * 0.4
b1 = np.zeros(hidden)
W2 = rng.standard_normal((hidden, 1)) * 0.4
b2 = np.zeros(1)
for epoch in range(epochs):
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)              # ReLU
    p = 1 / (1 + np.exp(-(h @ W2 + b2)))  # sigmoid
    d2 = (p - Y) / len(X)               # error at output
    d1 = (d2 @ W2.T) * (z1 > 0)        # error at hidden layer
    W2 -= lr * (h.T @ d2)
    b2 -= lr * d2.sum(axis=0)
    W1 -= lr * (X.T @ d1)
    b1 -= lr * d1.sum(axis=0)''', language="python")
    st.caption("For numerical stability, the executable version clips the sigmoid exponent. "
               "It also saves intermediate weights for the epoch slider and separates gradients "
               "from weight updates so you can inspect each one.")
    st.markdown("**Variable shapes**")
    st.code(f"X:   ({cut}, 2)       training observations × 2 inputs\n"
            f"W1:  (2, {hidden})       input → hidden weights\n"
            f"b1:  ({hidden},)        hidden biases\n"
            f"h:   ({cut}, {hidden})      hidden activations\n"
            f"W2:  ({hidden}, 1)       hidden → output weights\n"
            f"b2:  (1,)          output bias\n"
            f"Y:   ({cut}, 1)       correct classifications", language="text")

st.divider()
st.caption("Sources: Norges Bank's open data (original observations), "
           "frefrik/norway-finance-statistics (CSV redistribution). "
           "These two economic variables alone cannot identify the effect of monetary policy "
           "on exchange rates, and no causal interpretation should be made.")
st.markdown(f"[Exchange-rate CSV]({EUR_URL}) · [Policy-rate CSV]({RATE_URL}) · "
            "[Norges Bank's official data portal](https://app.norges-bank.no/query/)")
