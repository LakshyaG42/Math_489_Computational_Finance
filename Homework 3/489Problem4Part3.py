import numpy as np
import scipy.stats as stats
import math

def monte_carlo_option_expectation(n_simulations=100000, strike_price=1, mean_val=1, std_dev=1):
    # Here I am generating random samples from a normal distribution
    normal_samples = np.random.normal(mean_val, std_dev, n_simulations)

    # first expectation: E[max(0, K - X)]
    payoff_1 = np.maximum(strike_price - normal_samples, 0)
    expectation_estimate_1 = np.mean(payoff_1)

    # second expectation: E[max(0, K - exp(X))]
    exp_normal_samples = np.exp(normal_samples)
    payoff_2 = np.maximum(strike_price - exp_normal_samples, 0)
    expectation_estimate_2 = np.mean(payoff_2)

    std_error_1 = np.std(payoff_1) / np.sqrt(n_simulations)
    std_error_2 = np.std(payoff_2) / np.sqrt(n_simulations)

    # Here is the calculation for the 95% Confidence Intervals (using a Z-score of 1.96 for approx. 95% CI)
    confidence_interval_1 = (expectation_estimate_1 - 1.96 * std_error_1,
                             expectation_estimate_1 + 1.96 * std_error_1)
    confidence_interval_2 = (expectation_estimate_2 - 1.96 * std_error_2,
                             expectation_estimate_2 + 1.96 * std_error_2)

    return {
        "Monte Carlo Estimate (K-X)": expectation_estimate_1,
        "Confidence Interval (K-X)": confidence_interval_1,
        "Monte Carlo Estimate (K-exp(X))": expectation_estimate_2,
        "Confidence Interval (K-exp(X))": confidence_interval_2,
    }


if __name__ == "__main__":
    simulation_results = monte_carlo_option_expectation()

    print("\nProblem 4 Part 3 - Monte Carlo Simulation Results\n")
    for key, value in simulation_results.items():
        print(f"{key}: {value}")