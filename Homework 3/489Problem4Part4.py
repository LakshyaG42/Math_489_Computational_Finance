import numpy as np
import scipy.stats as stats
import math

def monte_carlo_antithetic_expectations(num_simulations=100000, strike_price=1, mean_val=1, std_dev=1):
    # I'm generating random samples from a normal distribution.
    normal_samples = np.random.normal(mean_val, std_dev, num_simulations)

    # Now, I'm creating my antithetic samples.
    # For each 'regular' sample, I'm creating an 'opposite' sample using the formula -X + 2.
    antithetic_samples = -normal_samples + 2

    exp_samples = np.exp(normal_samples)
    exp_antithetic_samples = np.exp(antithetic_samples)

    # Time to estimate the first expectation using antithetic variables.
    # I'm calculating the payoff for both 'regular' & 'antithetic' samples (K + X - 2)^+, then taking average
    expectation_estimate_1 = np.mean(
        (np.maximum(strike_price - normal_samples, 0) + np.maximum(strike_price + normal_samples - 2, 0)) / 2
    )
    expectation_estimate_2 = np.mean(
        (np.maximum(strike_price - exp_samples, 0) + np.maximum(strike_price - exp_antithetic_samples, 0)) / 2
    )
    std_error_1 = np.std(
        (np.maximum(strike_price - normal_samples, 0) + np.maximum(strike_price + normal_samples - 2, 0)) / 2
    ) / np.sqrt(num_simulations)
    std_error_2 = np.std(
        (np.maximum(strike_price - exp_samples, 0) + np.maximum(strike_price - exp_antithetic_samples, 0)) / 2
    ) / np.sqrt(num_simulations)

    #Finally, I'm calculating the 95% confidence intervals for both expectations.
    confidence_level = 1.96
    confidence_interval_1 = (expectation_estimate_1 - confidence_level * std_error_1,
                             expectation_estimate_1 + confidence_level * std_error_1)
    confidence_interval_2 = (expectation_estimate_2 - confidence_level * std_error_2,
                             expectation_estimate_2 + confidence_level * std_error_2)   
    return {
        "Monte Carlo Estimate (K-X) with Antithetic Variables": expectation_estimate_1,
        "Confidence Interval (K-X)": confidence_interval_1,
        "Monte Carlo Estimate (K-exp(X)) with Antithetic Variables": expectation_estimate_2,
        "Confidence Interval (K-exp(X))": confidence_interval_2,
    }
if __name__ == "__main__":
    simulation_results = monte_carlo_antithetic_expectations()
    print("\nProblem 4 Part 4 - Antithetic Variable Monte Carlo Simulation Results\n")
    for key, value in simulation_results.items():
        print(f"{key}: {value}")