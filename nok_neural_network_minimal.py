"""Minimal NumPy neural network: Norwegian interest rates and EUR/NOK.
Run: python nok_neural_network_minimal.py
Install: pip install numpy pandas matplotlib
Requires internet the first time, unless you save source CSVs beside this file.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW = 'https://raw.githubusercontent.com/frefrik/norway-finance-statistics/master/data/'
folder = Path(__file__).resolve().parent

def source(name):
    local = folder / name
    return str(local) if local.exists() else RAW + name

# DATA: observations in month t predict the direction of EUR/NOK in month t+1.
rate = pd.read_csv(source('no_keyPolicyRate.csv'), usecols=['Date', 'Rate'])
euro = pd.read_csv(source('no_exchangeRates.csv'), usecols=['Date', 'EUR'])
for frame in (rate, euro):
    frame['month'] = pd.to_datetime(frame['Date']).dt.to_period('M').dt.to_timestamp()
r = rate.groupby('month')['Rate'].mean()
e = euro.groupby('month')['EUR'].mean()
data = pd.concat([r.rename('interest_rate'), e.rename('eur_nok')], axis=1)
data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='MS'))
data = data.loc[(data.index.year >= 2005) &
                (data.index < pd.Timestamp.now().to_period('M').to_timestamp())].copy()
data['nok_change'] = 100 * (data.eur_nok / data.eur_nok.shift(1) - 1)
data['next_change'] = 100 * (data.eur_nok.shift(-1) / data.eur_nok - 1)
data = data.dropna(subset=['interest_rate', 'nok_change', 'next_change'])
X_raw = data[['interest_rate', 'nok_change']].to_numpy()
Y = (data.next_change.to_numpy() > 0).astype(float).reshape(-1, 1)
cut = int(0.8 * len(X_raw))  # Fit on earlier months, inspect later months.
mu, sd = X_raw[:cut].mean(axis=0), X_raw[:cut].std(axis=0)
sd[sd < 1e-10] = 1
X = (X_raw - mu) / sd
Xt, Yt = X[:cut], Y[:cut]

# THE NEURAL NETWORK: two inputs -> eight ReLU neurons -> one sigmoid.
np.random.seed(0); lr = 0.1
W1, b1 = np.random.randn(2, 8) * 0.4, np.zeros(8)
W2, b2 = np.random.randn(8, 1) * 0.4, np.zeros(1)
losses = []
for epoch in range(300):
    h = np.maximum(0, Xt @ W1 + b1)
    p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -40, 40)))
    p_safe = np.clip(p, 1e-12, 1 - 1e-12)
    losses.append(-np.mean(Yt*np.log(p_safe) + (1-Yt)*np.log(1-p_safe)))
    d2 = (p - Yt) / len(Xt)
    d1 = (d2 @ W2.T) * (h > 0)
    W2 -= lr * (h.T @ d2); b2 -= lr * d2.sum(axis=0)
    W1 -= lr * (Xt.T @ d1); b1 -= lr * d1.sum(axis=0)

def predict(standardized_inputs):
    h = np.maximum(0, standardized_inputs @ W1 + b1)
    return 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -40, 40)))

pred = predict(X[cut:]).ravel()
true = Y[cut:].ravel().astype(int)
accuracy = ((pred >= 0.5) == true).mean()
baseline = max(true.mean(), 1-true.mean())
print(f'{len(X)} monthly observations ({len(Xt)} train, {len(true)} later/test)')
print(f'Test accuracy: {accuracy:.1%}; majority-class baseline: {baseline:.1%}')
print('Correlation and prediction here are NOT evidence of a causal effect.')

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(losses)
ax.set(xlabel='Epoch', ylabel='Training cross-entropy', title='How the network learns')
ax.grid(alpha=.3)
fig.tight_layout()

# Show the probability surface in the original economic units.
xx, yy = np.meshgrid(np.linspace(*np.quantile(X_raw[:,0], [0, 1]), 100),
                     np.linspace(*np.quantile(X_raw[:,1], [0, 1]), 100))
grid = np.column_stack([xx.ravel(), yy.ravel()])
zz = predict((grid-mu)/sd).reshape(xx.shape)
fig2, ax2 = plt.subplots(figsize=(8, 5.3))
surface = ax2.contourf(xx, yy, zz, levels=np.linspace(0,1,11), cmap='RdYlBu_r', alpha=.7)
if zz.min()<.5<zz.max():
    ax2.contour(xx, yy, zz, levels=[.5], colors='black')
ax2.scatter(X_raw[:cut,0], X_raw[:cut,1], c=Y[:cut,0], cmap='coolwarm',
            vmin=0,vmax=1, s=16, edgecolors='white', linewidth=.3, label='Earlier/train')
ax2.scatter(X_raw[cut:,0], X_raw[cut:,1], c=Y[cut:,0], cmap='coolwarm',
            vmin=0,vmax=1, s=35, marker='x', label='Later/test')
fig2.colorbar(surface, label='P(NOK weakens next month)')
ax2.set(xlabel='Policy rate (%)', ylabel='EUR/NOK change this month (%)',
        title='Learned decision surface — real Norwegian economic data')
ax2.legend(); fig2.tight_layout()
plt.show()
