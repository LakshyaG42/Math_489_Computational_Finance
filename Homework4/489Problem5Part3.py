import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
n_sim, I, T, r, pi, lambda_val = 100000, 5, 5, 0.04, 0.35, 0.01 / 0.65
rhos = np.linspace(0.1, 0.9, 9) 
spreads = np.zeros((5, len(rhos)))

for idx, rho in enumerate(rhos):
    Sigma = np.full((I, I), rho)
    np.fill_diagonal(Sigma, 1)

    L = np.linalg.cholesky(Sigma)

    Z = np.random.normal(0, 1, (n_sim, I)) 
    X = Z @ L.T  

    U = norm.cdf(X)  
    tau = -np.log(U) / lambda_val 

    sorted_tau = np.sort(tau, axis=1)  

    protection_legs = np.zeros(5)
    premium_legs = np.zeros(5)
    
    for N in range(5):
        tau_N = sorted_tau[:, N]
        mask = tau_N <= T

        protection = (1 - pi) * np.exp(-r * tau_N) * mask
        protection_legs[N] = protection.mean()

        min_tau_T = np.minimum(tau_N, T)
        premium = (1 - np.exp(-r * min_tau_T)) / r
        premium_legs[N] = premium.mean()
    s_NtD = protection_legs / premium_legs
    spreads[:, idx] = s_NtD

plt.figure(figsize=(10, 6))
for N in range(5):
    plt.plot(rhos, spreads[N], 'o-', label=f'{N+1}tD')
plt.xlabel('Correlation (ρ)')
plt.ylabel('Spread')
plt.title('NtD Spreads vs Correlation')
plt.legend()
plt.grid(True)
plt.show()
