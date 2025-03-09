import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def compute_bond_prices_from_swaps(swap_rates):
    N = len(swap_rates)
    B = np.zeros(N+1)
    B[0] = 1.0 
    
    for n in range(1, N+1):
        s_n = swap_rates[n-1] 
        if n == 1:
            # B(0,1) = 1 / (1 + s1)
            B[n] = 1.0 / (1.0 + s_n)
        else:
            # B(0,n) = [1 - s_n * sum_{k=1}^{n-1} B(0,k)] / [1 + s_n]
            sum_prev = np.sum(B[1:n])  # sum of B(0,1)...B(0,n-1)
            B[n] = (1.0 - s_n * sum_prev) / (1.0 + s_n)
    return B

def compute_yields_from_bonds(B):
    N = len(B) - 1
    yields = np.zeros(N+1)
    yields[0] = np.nan 
    for n in range(1, N+1):
        yields[n] = (1.0 / B[n])**(1.0 / n) - 1.0
    return yields

def main():

    swap_rates_percent = [3.25, 3.75, 4.0, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25]
    swap_rates = [r / 100.0 for r in swap_rates_percent]
    
    # Part B: Compute B(0,n)
    B = compute_bond_prices_from_swaps(swap_rates)
    # ^ list of length 13: B[0], B[1], ..., B[12]

    yields = compute_yields_from_bonds(B)
    # yields[n] = discrete annual yield for maturity n

    # Cubic spline on B(0,n) over n=0..12
    n_points = np.arange(0, len(B))
    cs_B = CubicSpline(n_points, B)
    
    # Cubic spline on yields over n=1..12
    n_points_y = np.arange(1, len(B))  # i.e. 1..12
    cs_y = CubicSpline(n_points_y, yields[1:])
    
    # Create a fine grid for t in [0,12] for B(0,t)
    t_fine = np.linspace(0, 12, 200)
    B_spline = cs_B(t_fine)
    
    t_fine_y = np.linspace(1, 12, 200)
    # "Yield from B(t)" => y(t) = (1 / B(0,t))^(1/t) - 1
    B_spline_y = cs_B(t_fine_y)
    yield_from_B = (1.0 / B_spline_y)**(1.0 / t_fine_y) - 1.0
    
    # "Yield from direct interpolation" => cs_y(t)
    yield_spline = cs_y(t_fine_y)
    
    plt.figure(figsize=(8, 5))
    plt.plot(n_points, B, 'o', label='Bond Price B(0, n)', markersize=5)
    plt.plot(t_fine, B_spline, '-', label='Cubic Spline B(0,t)')
    plt.title("Cubic Spline Interpolation of Zero-Coupon Bond Prices B(0,t)")
    plt.xlabel("t (years)")
    plt.ylabel("Price B(0,t)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.plot(n_points_y, yields[1:], 'o', label='Annual yields (data)', markersize=5)
    plt.plot(t_fine_y, yield_spline, '--', label='Spline of yields')
    plt.plot(t_fine_y, yield_from_B, '-', label='Yield from B-spline')
    
    plt.title("Two Yield Curves")
    plt.xlabel("t (years)")
    plt.ylabel("Yield")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
