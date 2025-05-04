import numpy as np
from scipy.stats import norm

# Parameters
n_sim = 100000
I = 5
T = 5
r = 0.04
pi = 0.35
rho = 0.2
lambda_val = 0.01 / 0.65

# Correlation matrix
Sigma = np.full((I, I), rho)
np.fill_diagonal(Sigma, 1)
L = np.linalg.cholesky(Sigma)

# Storage
protection_legs = np.zeros(I)
premium_legs = np.zeros(I)
sum_times = np.zeros(I)
counts = np.zeros(I)

for _ in range(n_sim):
    Z = np.random.normal(0, 1, I)
    X = L @ Z
    U = norm.cdf(X)
    tau = -np.log(U) / lambda_val
    tau_sorted = np.sort(tau)
    
    for N in range(I):
        tau_N = tau_sorted[N] if N < I else np.inf
        if tau_N <= T:
            protection_legs[N] += (1 - pi) * np.exp(-r * tau_N)
            sum_times[N] += tau_N
            counts[N] += 1
        pv_premium = (1 - np.exp(-r * min(tau_N, T))) / r
        premium_legs[N] += pv_premium

# Compute spreads and expected times
s_NtD = (protection_legs / n_sim) / (premium_legs / n_sim)
expected_times = sum_times / counts

print("NtD Spreads:")
print(s_NtD)
print("\nExpected Times:")
print(expected_times)