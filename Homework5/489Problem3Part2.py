import numpy as np

def simulate_bond_price_log(T, scheme='euler', M=1_000_000):
    r     = 0.05  
    x0    = 0.1    
    theta = 0.1
    kappa = 0.25
    beta  = 0.2

    N = 200               
    Delta = T / N        
    clip_min, clip_max = -50, 50  

    Y = np.full((M, N+1), np.log(x0), dtype=np.float64)

    eps = np.random.randn(M, N)
    
    for n in range(N):
        Y_n = Y[:, n]
        drift = (kappa * (theta * np.exp(-Y_n) - 1) - 0.5 * beta**2 * np.exp(-Y_n)) * Delta
        diffusion = beta * np.exp(-Y_n / 2) * np.sqrt(Delta) * eps[:, n]
        if scheme == 'euler':
            Y_next = Y_n + drift + diffusion
        elif scheme == 'milstein':
            correction = -0.25 * beta**2 * np.exp(-Y_n) * Delta * (eps[:, n]**2 - 1)
            Y_next = Y_n + drift + diffusion + correction
        Y[:, n+1] = np.clip(Y_next, clip_min, clip_max)
    X = np.exp(np.clip(Y, clip_min, clip_max))
    I = np.sum((r + X[:, :-1]) * Delta, axis=1)
    discount_factors = np.exp(-I)
    bond_price = np.mean(discount_factors)
    return bond_price
T_values = [1, 10, 100]
schemes = ['euler', 'milstein']
for T in T_values:
    for scheme in schemes:
        price = simulate_bond_price_log(T, scheme=scheme, M=1_000_000)
        print(f"T = {T:3d}, {scheme.capitalize()} scheme bond price (log simulation): {price:.10f}")
