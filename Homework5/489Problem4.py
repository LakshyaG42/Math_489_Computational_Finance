import numpy as np

def simulate_pnl(T=1.0, N=100, M=1_000_000):
    dt = T / N
    Y = np.zeros((M, N+1))
    for n in range(N):
        eps = np.random.randn(M) 
        Y[:, n+1] = Y[:, n] - Y[:, n]*dt + np.sqrt(dt)*eps
    return Y[:, -1]
T = 1.0          
N = 100          
M = 1_000_000    
final_Y = simulate_pnl(T, N, M)
VaR_5 = np.percentile(final_Y, 5)
CVaR_5 = final_Y[final_Y <= VaR_5].mean()
print(f"VaR at 5%: {VaR_5:.4f}")
print(f"Conditional VaR at 5%: {CVaR_5:.4f}")
