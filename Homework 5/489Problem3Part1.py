import numpy as np

def simulate_bond_price(T, scheme='euler', M=1_000_000):
    r = 0.05
    x0 = 0.1
    theta = 0.1
    kappa = 0.25
    beta = 0.2

    N = 200           
    Delta = T / N       

    X = np.full((M, N+1), x0, dtype=np.float64)
    eps = np.random.randn(M, N)
    for n in range(N):
        X_current = X[:, n]
        sqrt_X = np.sqrt(np.maximum(X_current, 0)) # to avoid negative values
        
        if scheme == 'euler':
            X_next = X_current + kappa*(theta - X_current)*Delta + beta * sqrt_X * np.sqrt(Delta) * eps[:, n]
        elif scheme == 'milstein':
            X_next = (X_current + kappa*(theta - X_current)*Delta +
                      beta * sqrt_X * np.sqrt(Delta) * eps[:, n] +
                      0.25 * beta**2 * Delta * (eps[:, n]**2 - 1))
        else:
            raise ValueError("scheme must be 'euler' or 'milstein'")

        X[:, n+1] = np.maximum(X_next, 0)

    I = np.sum((r + X[:, :-1]) * Delta, axis=1)
    discount_factors = np.exp(-I)
    bond_price = np.mean(discount_factors)
    return bond_price
T_list = [1, 10, 100]
schemes = ['euler', 'milstein']

results = {}
for T in T_list:
    results[T] = {}
    for scheme in schemes:
        price = simulate_bond_price(T, scheme=scheme, M=1_000_000)
        results[T][scheme] = price
        print(f"T = {T:3d}, {scheme.capitalize()} scheme bond price: {price}")

        