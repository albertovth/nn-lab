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
BASE_FEATURE_COLUMNS = ("rate_pct", "current_change_pct")
MAX_HISTORY_MONTHS = 3
MOVEMENT_WINDOWS = (0, 3, 6)
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


def add_history_features(monthly):
    """Add past monthly inputs without using information after the target month."""
    monthly = monthly.copy()
    for lag in range(1, MAX_HISTORY_MONTHS + 1):
        monthly[f"rate_pct_lag{lag}"] = monthly["rate_pct"].shift(lag)
        monthly[f"current_change_pct_lag{lag}"] = monthly["current_change_pct"].shift(lag)
    return monthly


def add_movement_features(monthly):
    """Summarize changes known by the end of each month."""
    monthly = monthly.copy()
    for window in MOVEMENT_WINDOWS[1:]:
        changes = monthly["current_change_pct"].rolling(window, min_periods=window)
        monthly[f"eur_trend_{window}m"] = changes.mean()
        monthly[f"eur_volatility_{window}m"] = changes.std(ddof=0)
    return monthly


def selected_features(history_months, movement_window=0):
    columns = list(BASE_FEATURE_COLUMNS)
    names = list(FEATURE_NAMES)
    for lag in range(1, history_months + 1):
        when = "previous month" if lag == 1 else f"{lag} months earlier"
        columns.extend((f"rate_pct_lag{lag}", f"current_change_pct_lag{lag}"))
        names.extend((f"Policy rate, {when} (%)", f"EUR/NOK change, {when} (%)"))
    if movement_window:
        columns.extend((f"eur_trend_{movement_window}m",
                        f"eur_volatility_{movement_window}m"))
        names.extend((f"EUR/NOK mean monthly change, last {movement_window} months (%)",
                      f"EUR/NOK change volatility, last {movement_window} months (%)"))
    return columns, names


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


def objective_loss(Y, p, weights, penalty):
    W1, _b1, W2, _b2 = weights
    return binary_loss(Y, p) + 0.5 * penalty * (np.sum(W1**2) + np.sum(W2**2))


def gradients(X, Y, weights, penalty=0.0):
    W1, b1, W2, b2 = weights
    z1, h, _z2, p = forward(X, weights)
    d2 = (p - Y) / len(X)             # sigmoid + cross-entropy derivative
    d1 = (d2 @ W2.T) * (z1 > 0)      # backpropagate through ReLU
    gW2 = h.T @ d2 + penalty * W2
    gb2 = d2.sum(axis=0)
    gW1 = X.T @ d1 + penalty * W1
    gb1 = d1.sum(axis=0)
    return gW1, gb1, gW2, gb2


def backtracking_update(X, Y, weights, grads, max_step, current_loss, penalty=0.0):
    """Try smaller steps until the training objective does not rise."""
    step = max_step
    for _ in range(21):
        trial = tuple(w - step * g for w, g in zip(weights, grads))
        if objective_loss(Y, forward(X, trial)[3], trial, penalty) <= current_loss:
            return trial, step
        step *= 0.5
    return weights, 0.0


def train_network(X, Y, epochs=300, lr=0.1, hidden=8, penalty=0.0):
    """Full-batch descent with backtracking; store weights after every epoch."""
    input_rng = np.random.default_rng(0)
    output_rng = np.random.default_rng(1)
    W1 = input_rng.standard_normal((X.shape[1], hidden)) * 0.4
    b1 = np.zeros(hidden)
    W2 = output_rng.standard_normal((hidden, 1)) * 0.4
    b2 = np.zeros(1)
    weights = (W1, b1, W2, b2)
    snapshots, losses, step_sizes = [], [], []
    for epoch in range(epochs + 1):
        # Snapshot 0 is the network BEFORE any training.
        snapshots.append(tuple(w.copy() for w in weights))
        current_loss = objective_loss(Y, forward(X, weights)[3], weights, penalty)
        losses.append(current_loss)
        if epoch == epochs:
            break
        grads = gradients(X, Y, weights, penalty)
        weights, step = backtracking_update(X, Y, weights, grads, lr,
                                            current_loss, penalty)
        step_sizes.append(step)
    return snapshots, np.asarray(losses), np.asarray(step_sizes)


def class_probability(X, weights):
    return forward(X, weights)[3].ravel()


def walk_forward_check(X_raw, labels, training_end, epochs, lr, hidden, penalty):
    """Evaluate three expanding-window folds inside the training period."""
    edges = np.linspace(training_end // 2, training_end, 4, dtype=int)
    correct, majority_correct, persistence_correct, fold_scores = [], [], [], []
    for train_end, validation_end in zip(edges[:-1], edges[1:]):
        mean = X_raw[:train_end].mean(axis=0)
        scale = X_raw[:train_end].std(axis=0)
        scale = np.where(scale < 1e-10, 1, scale)
        X_train = (X_raw[:train_end] - mean) / scale
        X_validation = (X_raw[train_end:validation_end] - mean) / scale
        snapshots, _, _ = train_network(X_train, labels[:train_end].reshape(-1, 1),
                                        epochs, lr, hidden, penalty)
        actual = labels[train_end:validation_end]
        predicted = class_probability(X_validation, snapshots[-1]) >= 0.5
        hits = predicted == actual
        correct.extend(hits)
        fold_scores.append(float(np.mean(hits)))
        majority_class = int(np.mean(labels[:train_end]) >= 0.5)
        majority_correct.extend(actual == majority_class)
        persistence_correct.extend((X_raw[train_end:validation_end, 1] > 0) == actual)
    return (float(np.mean(correct)), float(np.mean(majority_correct)),
            float(np.mean(persistence_correct)), fold_scores, len(correct))


# ---------- 3. Illustrations ----------
def series_plot(series, ylabel):
    values = series.dropna()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(values.index, values.to_numpy(), color="#276EB3", linewidth=1.8)
    low, high = float(values.min()), float(values.max())
    padding = 0.08 * (high - low) if high > low else max(abs(low) * 0.08, 0.1)
    ax.set_ylim(low - padding, high + padding)
    ax.set_xlim(values.index.min(), values.index.max())
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def decision_mesh(X_raw, selected, xx, yy, feature_columns, movement_window):
    """Vary two current inputs while keeping the chosen month's past fixed."""
    mesh = np.tile(X_raw[selected], (xx.size, 1))
    mesh[:, 0], mesh[:, 1] = xx.ravel(), yy.ravel()
    if movement_window:
        trend_index = feature_columns.index(f"eur_trend_{movement_window}m")
        volatility_index = feature_columns.index(f"eur_volatility_{movement_window}m")
        chosen_change = X_raw[selected, 1]
        chosen_mean = X_raw[selected, trend_index]
        chosen_volatility = X_raw[selected, volatility_index]
        past_sum = movement_window * chosen_mean - chosen_change
        past_squares = (movement_window * (chosen_volatility**2 + chosen_mean**2)
                        - chosen_change**2)
        new_mean = (past_sum + mesh[:, 1]) / movement_window
        new_variance = (past_squares + mesh[:, 1]**2) / movement_window - new_mean**2
        mesh[:, trend_index] = new_mean
        mesh[:, volatility_index] = np.sqrt(np.maximum(new_variance, 0))
    return mesh


def decision_plot(X_raw, y, train_cut, mean, scale, weights, selected,
                  feature_columns, movement_window):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    xmin, xmax = np.quantile(X_raw[:, 0], [0.01, 0.99]); xmin -= 0.5; xmax += 0.5
    ymin, ymax = np.quantile(X_raw[:, 1], [0.01, 0.99]); ymin -= 0.65; ymax += 0.65
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 110), np.linspace(ymin, ymax, 110))
    # Past inputs stay fixed; current-derived summaries follow the grid value.
    mesh = decision_mesh(X_raw, selected, xx, yy, feature_columns, movement_window)
    probs = class_probability((mesh - mean) / scale, weights).reshape(xx.shape)
    surf = ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.49)
    if np.min(probs) < 0.5 < np.max(probs):
        boundary = ax.contour(xx, yy, probs, levels=[0.5], colors=["#222222"], linewidths=1.6)
        ax.clabel(boundary, fmt={0.5: "50%"}, inline=True, fontsize=9)
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
           title="Black line = 50% predicted chance of NOK weakening")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def network_plot(x, weights, selected_hidden):
    W1, b1, W2, b2 = weights
    z1, h, _z2, p = forward(x.reshape(1, -1), weights)
    n_inputs, n = W1.shape
    fig, ax = plt.subplots(figsize=(9, max(4, 0.65 * max(n_inputs, n) + 1)))
    inp_y = np.linspace(0.92, 0.08, n_inputs)
    hid_y = np.linspace(0.94, 0.06, n)
    out_y = 0.5
    def edge(xa, ya, xb, yb, w):
        ax.plot([xa, xb], [ya, yb], color=("#218B73" if w >= 0 else "#D07634"),
                linewidth=0.4 + min(3.2, abs(w) * 1.8), alpha=0.40, zorder=1)
    for i in range(n_inputs):
        for j in range(n):
            edge(0, inp_y[i], 1, hid_y[j], W1[i, j])
    for j in range(n):
        edge(1, hid_y[j], 2, out_y, W2[j, 0])
    for i in range(n_inputs):
        ax.scatter(0, inp_y[i], s=(950 if n_inputs == 2 else 620), c="#377EB8", edgecolors="white", zorder=3)
        ax.text(0, inp_y[i], f"x{i+1}\n{x[i]:+.2f}", color="white", ha="center", va="center",
                fontsize=(9 if n_inputs == 2 else 7), zorder=5)
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
    ax.text(0, 1.04, f"{n_inputs} normalized inputs", ha="center", fontsize=10)
    ax.text(1, 1.04, f"{n} ReLU neurons", ha="center", fontsize=10)
    ax.text(2, 1.04, "Sigmoid output", ha="center", fontsize=10)
    ax.set_title("Green connections: positive weights   ·   Orange: negative weights", fontsize=10, pad=18)
    fig.tight_layout()
    return fig


# ---------- 4. Streamlit interface ----------
st.set_page_config(page_title="NOK Neural Network Lab",
                   page_icon=HERE / "blank_favicon.png", layout="wide")
st.markdown("""<style>
/* Leave room for the top of large letters at different browser zoom levels. */
[data-testid="stAppViewContainer"] h1 {
    line-height: 1.4;
    padding-top: 0.15em;
    overflow: visible;
}
</style>""", unsafe_allow_html=True)
st.title("NOK neural network lab")
st.caption("A real-data experiment: Can today's interest rate and EUR/NOK movement tell us whether NOK will weaken next month?")
st.write("This is a **learning exercise**, not an economic or currency forecasting tool. "
         "More input history may help or hurt later-period accuracy; exchange rates are difficult to predict.")

with st.sidebar:
    st.header("Experiment settings")
    start_year = st.select_slider("Use data from", options=[2000, 2005, 2010, 2015, 2020], value=2005)
    history_months = st.slider("Previous months of inputs", 0, MAX_HISTORY_MONTHS, 0,
                               help="Add the policy rate and EUR/NOK change from each previous month. "
                                    "The same observations are used for every setting, so test results can be compared.")
    movement_window = st.select_slider("Recent EUR/NOK summary",
                                        options=MOVEMENT_WINDOWS, value=0,
                                        format_func=lambda months: "Off" if months == 0 else f"{months} months",
                                        help="Add the average monthly EUR/NOK change and its volatility "
                                             "over the chosen window. Only current and past months are used.")
    hidden = st.slider("Hidden neurons", 2, 16, 8)
    penalty = st.slider("Weight penalty (L2)", 0.0, 0.10, 0.0, 0.005,
                        help="Discourages large weights. Zero turns it off. "
                             "The loss chart includes this penalty when it is above zero.")
    epochs = st.slider("Training epochs", 0, 500, 300, 10)
    lr = st.slider("Maximum learning rate", 0.01, 1.0, 0.10, 0.01,
                   help="Each update tries this step size first. If the training objective would rise, "
                        "the step is halved until it falls.")
    st.caption("The starting weights are repeatable across settings. Moving the epoch slider does not change them.")
    if st.button("Refresh source data"):
        get_daily_series.clear()
        st.rerun()

try:
    rate_daily, eur_daily = get_daily_series()
except Exception as exc:
    st.markdown(f"**Could not load the CSV sources:** {exc}")
    st.markdown("**Manual fallback:** Download the two CSV files linked below and place them "
                "in the same folder as `nok_neural_network.py`, then restart the app.")
    st.markdown(f"[Policy rate CSV]({RATE_URL}) · [EUR/NOK CSV]({EUR_URL})")
    st.stop()

monthly = add_movement_features(make_monthly(rate_daily, eur_daily))
monthly = monthly.loc[monthly.index.year >= start_year].copy()
monthly = add_history_features(monthly)
all_feature_columns, _ = selected_features(MAX_HISTORY_MONTHS)
for window in MOVEMENT_WINDOWS[1:]:
    all_feature_columns.extend((f"eur_trend_{window}m",
                                f"eur_volatility_{window}m"))
feature_columns, feature_names = selected_features(history_months, movement_window)
# Keep the same eligible months and split for every feature setting.
model_rows = monthly[all_feature_columns + ["next_change_pct",
                      "depreciates_next_month"]].dropna().copy()
if len(model_rows) < 40:
    st.markdown("**Not enough complete monthly observations.** Try an earlier start year.")
    st.stop()

X_raw = model_rows[feature_columns].to_numpy(dtype=float)
Y_all = model_rows["depreciates_next_month"].to_numpy(dtype=int)
Y = Y_all.reshape(-1, 1).astype(float)
cut = int(0.8 * len(X_raw))  # chronological split; no randomly mixing future with past
mean = X_raw[:cut].mean(axis=0)
scale = X_raw[:cut].std(axis=0)
scale = np.where(scale < 1e-10, 1, scale)
X = (X_raw - mean) / scale

snapshots, losses, step_sizes = train_network(X[:cut], Y[:cut], epochs, lr, hidden,
                                              penalty)

st.caption(f"{len(X)} months ({model_rows.index.min():%b %Y}–{model_rows.index.max():%b %Y}); "
           f"{cut} earlier months for training, {len(X)-cut} later months for testing. "
           f"{len(feature_columns)} inputs from the current month and {history_months} previous month(s). "
           f"Recent EUR/NOK summary: {'off' if movement_window == 0 else f'{movement_window} months'}. "
           "The same months are used for every feature setting. Data source: Norges Bank, republished by frefrik.")

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
training_majority_class = int(np.mean(Y_all[:cut]) >= 0.5)
majority_baseline = np.mean(Y_all[cut:] == training_majority_class)
persistence_baseline = np.mean((X_raw[cut:, 1] > 0) == Y_all[cut:])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Training accuracy", f"{train_acc:.1%}")
m2.metric("Later-period accuracy", f"{test_acc:.1%}")
m3.metric("Training-majority baseline", f"{majority_baseline:.1%}",
          help="Choose the more common outcome in training, then always predict it for later months.")
m4.metric("P(NOK weakens), chosen month", f"{probabilities[month_index]:.1%}")

view_data, view_boundary, view_neuron, view_learning, view_code = st.tabs(
    ["Economic data", "Decision boundary", "Inside a neuron", "Learning", "The NumPy code"]
)

with view_data:
    st.write("Each dot used by the network is a month. Inputs use information available "
             "at the **end of that month**. The label tells us what happened **the following month**.")
    st.caption(f"The two charts show observed monthly data through {monthly.index.max():%B %Y}.")
    st.caption("The table stops before the latest complete input month because that month's "
               "following-month outcome is not known yet. The rate values shown are observations, "
               "not rate forecasts.")
    if movement_window:
        st.write(f"The {movement_window}-month summary adds two inputs: the average of the monthly "
                 "EUR/NOK percentage changes and their standard deviation. It includes the selected "
                 "month and earlier months only. Higher volatility means the recent changes varied more.")
    a, b = st.columns(2)
    with a:
        st.markdown("**EUR/NOK (NOK per euro)**")
        fig = series_plot(monthly["eur_nok"], "NOK per euro")
        st.pyplot(fig, width="stretch"); plt.close(fig)
    with b:
        st.markdown("**Norges Bank policy rate (%)**")
        fig = series_plot(monthly["rate_pct"], "Policy rate (%)")
        st.pyplot(fig, width="stretch"); plt.close(fig)
    display_rows = model_rows[feature_columns + ["next_change_pct", "depreciates_next_month"]]
    display_rows = display_rows.rename(columns={**dict(zip(feature_columns, feature_names)),
                                                "next_change_pct": "Next-month EUR/NOK change (%)",
                                                "depreciates_next_month": "Next month: NOK weakens? (1=yes)"})
    st.dataframe(display_rows, width="stretch", row_height=42)
    st.download_button("Download prepared monthly dataset (CSV)",
                       data=display_rows.to_csv(index_label="month").encode("utf-8"),
                       file_name="nok_monthly_network_data.csv", mime="text/csv")

with view_boundary:
    st.markdown("The horizontal axis is the current policy rate; the vertical axis is this month's EUR/NOK change. "
                "The background color shows the network's predicted chance that NOK weakens **next month**. "
                "On the **red side** of the black line, that chance is above 50%, so the model predicts "
                "NOK will weaken. On the **blue side**, it is below 50%, so the model predicts NOK will "
                "strengthen or stay level. The black line itself is exactly 50%. "
                "Circles are training months; crosses are later months.")
    if history_months or movement_window:
        st.caption("This is a two-axis slice: past-month inputs are held at the selected month's values, "
                   "while recent-movement summaries follow the current-month change on the vertical axis. "
                   "Each dot has its own past, so its prediction may differ from the background at that position.")
    fig = decision_plot(X_raw, Y_all, cut, mean, scale, w, month_index,
                        feature_columns, movement_window)
    st.pyplot(fig, width="stretch"); plt.close(fig)
    st.caption("The selected month is marked by a star. Changing the epoch slider shows how the learned surface changes.")

with view_neuron:
    chosen = st.slider("Select hidden neuron", 1, hidden, 1) - 1
    st.markdown("Inputs are standardized using **only the training sample's** mean and standard deviation. "
                "This changes units but not the observations' economic information.")
    xv = X[month_index]
    W1, b1, W2, b2 = w
    z1, h, z2, p = forward(xv.reshape(1, -1), w)
    left, right = st.columns([1.25, 1])
    with left:
        fig = network_plot(xv, w, chosen)
        st.pyplot(fig, width="stretch"); plt.close(fig)
    with right:
        st.markdown(f"**Month:** {model_rows.index[month_index]:%B %Y}")
        st.write(f"Actual rate: **{X_raw[month_index, 0]:.3f}%**")
        st.write(f"Actual current-month EUR/NOK change: **{X_raw[month_index, 1]:+.3f}%**")
        st.write(f"The network uses **{len(xv)} standardized inputs**, listed below the diagram.")
        st.markdown(f"**Neuron {chosen + 1}: weighted sum**")
        st.latex(r"z_j = \sum_i x_iw_{ij}+b_j")
        st.code(f"z = sum(input × weight) + bias\n"
                f"  = {np.dot(xv, W1[:, chosen]):+.4f} + {b1[chosen]:+.4f}\n"
                f"  = {z1[0,chosen]:+.4f}\n"
                f"ReLU(z) = max(0, z) = {h[0,chosen]:.4f}", language="text")
        st.markdown("**Output:** all hidden activations contribute")
        st.latex(r"p = \sigma\!\left(\sum_j h_jW_{2,j}+b_2\right)")
        st.write(f"Output logit = **{z2[0,0]:+.4f}**; sigmoid(logit) = **{p[0,0]:.4f}**")
        st.write("Observed next month:", "**NOK weakened**" if Y_all[month_index] else "**NOK strengthened or held**")

    st.markdown("**Inputs to the selected neuron**")
    contributions = pd.DataFrame({"Input": feature_names,
                                  "Raw value": X_raw[month_index],
                                  "Standardized x": xv,
                                  "W1 weight": W1[:, chosen],
                                  "x × weight": xv * W1[:, chosen]})
    st.dataframe(contributions.style.format({"Raw value": "{:+.3f}",
                                             "Standardized x": "{:+.3f}",
                                             "W1 weight": "{:+.4f}",
                                             "x × weight": "{:+.4f}"}),
                 hide_index=True, width="stretch", row_height=42)

    st.markdown("**Look inside the matrices**")
    mat = pd.DataFrame({"Neuron": np.arange(1, hidden + 1)}).set_index("Neuron")
    for i, name in enumerate(feature_names):
        mat[f"W1: x{i + 1} ({name})"] = W1[i]
    mat["bias"] = b1
    mat["weighted sum"] = z1[0]
    mat["ReLU output"] = h[0]
    mat["W2 to output"] = W2[:, 0]
    st.dataframe(mat.style.format("{:+.4f}"), width="stretch", row_height=42)

    st.markdown("**A real gradient update (computed on the training months)**")
    gW1, gb1, gW2, gb2 = gradients(X[:cut], Y[:cut], w, penalty)
    _, next_step = backtracking_update(X[:cut], Y[:cut], w,
                                       (gW1, gb1, gW2, gb2), lr,
                                       objective_loss(Y[:cut], forward(X[:cut], w)[3],
                                                      w, penalty), penalty)
    dweight = -next_step * gW1[0, chosen]
    st.code(f"∂objective/∂W1[0,{chosen}] = {gW1[0,chosen]:+.7f}\n"
            f"accepted learning rate = {next_step:.5f} (maximum {lr:.2f})\n"
            f"update = −accepted_rate × gradient = {dweight:+.7f}\n"
            f"weight before = {W1[0,chosen]:+.7f}\n"
            f"weight after  = {W1[0,chosen]+dweight:+.7f}", language="text")
    st.caption("The data part of the gradient is averaged over all training observations; the optional "
               "weight penalty is then added. This displays the next proposed update to the selected "
               "weight and does not train on the inspected test month.")
    if st.button("Check that gradient numerically (finite differences)"):
        eps = 1e-5
        wp = [v.copy() for v in w]; wm = [v.copy() for v in w]
        wp[0][0, chosen] += eps; wm[0][0, chosen] -= eps
        ngrad = (objective_loss(Y[:cut], forward(X[:cut], wp)[3], wp, penalty) -
                 objective_loss(Y[:cut], forward(X[:cut], wm)[3], wm, penalty)) / (2*eps)
        st.write(f"Backprop gradient: {gW1[0,chosen]:+.9f}   |   "
                 f"Numerical gradient: {ngrad:+.9f}   |   "
                 f"Absolute difference: {abs(gW1[0,chosen]-ngrad):.2g}")
        training_z1 = forward(X[:cut], w)[0][:, chosen]
        if np.any(np.abs(training_z1) <= eps * np.abs(X[:cut, 0])):
            st.caption("The tiny numerical step crosses ReLU's corner at zero for at least one "
                       "training month. A central finite difference can disagree with backpropagation "
                       "there because the slope is not unique. Try an earlier epoch or another neuron.")

with view_learning:
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(np.arange(len(losses)), losses, linewidth=2.0)
    ax.axvline(shown_epoch, linestyle="--", color="#CB772F", label=f"Selected epoch: {shown_epoch}")
    loss_label = "Training objective (includes weight penalty)" if penalty else "Training cross-entropy loss"
    ax.set(xlabel="Epoch", ylabel=loss_label, title="Learning by gradient descent")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)
    cross_entropy = binary_loss(Y[:cut], forward(X[:cut], w)[3])
    st.write(f"At epoch {shown_epoch}: training objective **{losses[shown_epoch]:.4f}** "
             f"(cross-entropy **{cross_entropy:.4f}**, weight penalty "
             f"**{losses[shown_epoch]-cross_entropy:.4f}**). "
             "These values use training months, not later-period predictions.")
    reduced_steps = np.count_nonzero((step_sizes[:shown_epoch] > 0) &
                                     (step_sizes[:shown_epoch] < lr))
    st.caption(f"The maximum learning rate is {lr:.2f}. The app halved the step "
               f"on {reduced_steps} of {shown_epoch} updates to keep the training objective from rising. "
               "A smoother training curve does not guarantee better later-period predictions.")
    st.write(f"**Simple benchmarks on the later months:** training-majority "
             f"**{majority_baseline:.1%}**; predict the same direction as the current month "
             f"**{persistence_baseline:.1%}**. Both rules can be applied without seeing the later labels.")
    if history_months or movement_window:
        # Compare on identical months, split and training settings; only the
        # additional inputs differ. Neither model is selected on test results.
        current_raw = model_rows[list(BASE_FEATURE_COLUMNS)].to_numpy(dtype=float)
        current_mean = current_raw[:cut].mean(axis=0)
        current_scale = current_raw[:cut].std(axis=0)
        current_scale = np.where(current_scale < 1e-10, 1, current_scale)
        current_X = (current_raw - current_mean) / current_scale
        current_snapshots, _, _ = train_network(current_X[:cut], Y[:cut], epochs,
                                                lr, hidden, penalty)
        current_predictions = (class_probability(current_X[cut:],
                                                 current_snapshots[shown_epoch]) >= 0.5)
        current_test_acc = (current_predictions == Y_all[cut:]).mean()
        st.write(f"**Same-month comparison:** current-month inputs alone: **{current_test_acc:.1%}** "
                 f"later-period accuracy; with the chosen extra inputs: **{test_acc:.1%}**. "
                 "Adding inputs can reduce accuracy, especially with a small sample.")
    st.caption("Trying many settings against the same later period uses it for model selection. "
               "The best observed score is exploratory and needs a fresh period for independent evaluation.")
    if st.button("Run walk-forward check on training years"):
        wf_acc, wf_majority, wf_persistence, fold_scores, checked_months = walk_forward_check(
            X_raw, Y_all, cut, epochs, lr, hidden, penalty)
        st.write(f"**Walk-forward accuracy:** network **{wf_acc:.1%}** across {checked_months} "
                 f"validation months; training-majority **{wf_majority:.1%}**; "
                 f"same-direction rule **{wf_persistence:.1%}**. "
                 "Each fold trains on earlier months and predicts the next block.")
        st.caption("Three fold scores, oldest to newest: " +
                   ", ".join(f"{score:.1%}" for score in fold_scores) +
                   ". These months are all inside the original training period.")
    st.markdown("**Training versus later months**")
    chart = pd.DataFrame({"Date": model_rows.index, "Actual": Y_all,
                          "Predicted probability": probabilities,
                          "Sample": np.where(np.arange(len(X)) < cut, "Training", "Later/test")})
    st.dataframe(chart.tail(36), hide_index=True, width="stretch", row_height=42)
    st.caption("With small time-series datasets, a network can fit the training period without "
               "predicting unseen months. Try different input histories, recent-movement summaries, "
               "hidden neurons and learning rates.")

with view_code:
    st.markdown("**This is the core learning algorithm in the app.** It uses NumPy, "
                "not a ready-made neural-network package. Here `X` and `Y` mean the standardized "
                "training inputs and their observed labels. The app separates these steps into functions "
                "and saves weights after every epoch for the slider. `penalty` is the optional "
                "weight penalty from the sidebar.")
    st.code('''input_rng = np.random.default_rng(0)
output_rng = np.random.default_rng(1)
W1 = input_rng.standard_normal((X.shape[1], hidden)) * 0.4
b1 = np.zeros(hidden)
W2 = output_rng.standard_normal((hidden, 1)) * 0.4
b2 = np.zeros(1)
weights = (W1, b1, W2, b2)
for epoch in range(epochs):
    W1, b1, W2, b2 = weights
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)              # ReLU
    z2 = h @ W2 + b2
    p = sigmoid(z2)
    d2 = (p - Y) / len(X)               # error at output
    d1 = (d2 @ W2.T) * (z1 > 0)        # error at hidden layer
    gW2 = h.T @ d2 + penalty * W2
    gb2 = d2.sum(axis=0)
    gW1 = X.T @ d1 + penalty * W1
    gb1 = d1.sum(axis=0)
    grads = (gW1, gb1, gW2, gb2)
    loss = objective_loss(Y, p, weights, penalty)
    step = lr
    for _ in range(21):
        trial = tuple(w - step*g for w, g in zip(weights, grads))
        if objective_loss(Y, forward(X, trial)[3], trial, penalty) <= loss:
            weights = trial
            break
        step *= 0.5''', language="python")
    st.markdown("""**Every line, in order**

1. `input_rng = np.random.default_rng(0)` creates a seeded generator for input-to-hidden weights, making their starting values repeatable.
2. `output_rng = np.random.default_rng(1)` creates a separate seeded generator. This keeps the output weights' starting values the same when the history slider changes the input count.
3. `W1 = ...` creates one weight from each input to each hidden neuron, then multiplies the initial values by 0.4. The first two rows retain the same starting values across history settings.
4. `b1 = np.zeros(hidden)` gives each hidden neuron a bias, initially zero. A bias is an adjustable offset, not an embedding.
5. `W2 = ...` creates one initial weight from each hidden neuron to the single output.
6. `b2 = np.zeros(1)` gives the output one adjustable offset, initially zero.
7. `weights = (W1, b1, W2, b2)` groups the four arrays so each trial can update them together.
8. `for epoch in range(epochs):` repeats the forward pass, backpropagation and weight update for the chosen number of epochs. Each pass uses all training months.
9. `W1, b1, W2, b2 = weights` unpacks the current weights and biases.
10. `z1 = X @ W1 + b1` multiplies every input row by the first-layer weights and adds the hidden biases. `z1` is the value before activation.
11. `h = np.maximum(0, z1)` applies ReLU: negative values become zero; positive values pass through.
12. `z2 = h @ W2 + b2` combines the hidden activations into one raw output value, called a logit, for each month.
13. `p = sigmoid(z2)` converts each logit to a value between zero and one. The function clips extreme inputs before exponentiation for numerical stability.
14. `d2 = (p - Y) / len(X)` calculates the output error used by the gradient of binary cross-entropy, averaged over training months.
15. `d1 = (d2 @ W2.T) * (z1 > 0)` sends that error back through the output weights. The mask stops it at hidden neurons where ReLU was inactive.
16. `gW2 = h.T @ d2 + penalty * W2` calculates how the objective changes with each hidden-to-output weight. The extra term discourages large weights; it is zero when the penalty slider is zero.
17. `gb2 = d2.sum(axis=0)` calculates how the loss changes with the output bias.
18. `gW1 = X.T @ d1 + penalty * W1` calculates the input-to-hidden weight gradients, including the same optional penalty.
19. `gb1 = d1.sum(axis=0)` calculates how the loss changes with each hidden bias.
20. `grads = (gW1, gb1, gW2, gb2)` groups the gradients in the same order as the weights.
21. `loss = objective_loss(Y, p, weights, penalty)` records cross-entropy plus half the penalty strength times the sum of squared weights. Biases are excluded.
22. `step = lr` starts with the maximum learning rate chosen in the sidebar.
23. `for _ in range(21):` allows up to 21 attempts to find an update that does not raise the training objective.
24. `trial = ...` subtracts the proposed step times each gradient from its corresponding weight or bias.
25. `if objective_loss(...) <= loss:` checks the objective after the proposed update.
26. `weights = trial` accepts the update when its objective does not exceed the old objective.
27. `break` ends the retry loop after an accepted update.
28. `step *= 0.5` halves the step after a rejected update. If every attempt fails, the old weights remain in place.

`W1` and `W2` are weight matrices; `b1` and `b2` are biases. This network has no embedding matrix.""")
    st.markdown("**Variable shapes**")
    st.code(f"X:   ({cut}, {len(feature_columns)})       training observations × inputs\n"
            f"W1:  ({len(feature_columns)}, {hidden})       input → hidden weights\n"
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
