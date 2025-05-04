import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd

# Black-Scholes call price (r = 0, S0 = 100)
def call_price(sigma, T, K):
    S0 = 100.0
    d1 = (np.log(S0/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * norm.cdf(d2)

def implied_vol(C_market, T, K, sigma_low=1e-6, sigma_high=5.0):
    func = lambda sigma: call_price(sigma, T, K) - C_market
    try:
        # Use Brent's method to find the root
        vol = brentq(func, sigma_low, sigma_high)
    except ValueError:
        vol = np.nan  # edge caese where no solution is found
    return vol

T_vals = [0.1, 0.5, 1, 2, 5]
K_vals = [80, 90, 100, 110, 120]
market_prices = np.array([
    [20.0004, 10.0000, 1.2618, 0.0022, 0.0001],
    [20.0107, 10.3038, 2.8358, 0.2273, 0.0039],
    [20.1117, 10.9033, 4.0219, 0.8194, 0.0843],
    [20.5035, 12.0277, 5.7094, 2.0770, 0.5651],
    [22.0009, 14.6868, 9.0662, 5.1655, 2.7213]
])
implied_vols = np.zeros(market_prices.shape)
for i, T in enumerate(T_vals):
    for j, K in enumerate(K_vals):
        C_market = market_prices[i, j]
        vol = implied_vol(C_market, T, K)
        implied_vols[i, j] = vol

implied_vols_df = pd.DataFrame(
    implied_vols,
    index=[f"T = {T}" for T in T_vals],
    columns=[f"K = {K}" for K in K_vals]
)
print("Implied Volatilities DataFrame:")
print(implied_vols_df)
