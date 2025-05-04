import numpy as np
import scipy.stats as stats
import math

def monte_carlo_control_variates_estimation(num_sims=100000, strike=1, avg_val=1, std_dev=1):
    norm_samples = np.random.normal(avg_val, std_dev, num_sims)
    exp_samples_val = np.exp(norm_samples)
    # I'm finding the average of max(0, K - X) for my normal samples.
    mc_estimate_1 = np.mean(np.maximum(strike - norm_samples, 0))
    # And calculating the regular MC estimate for the second expectation: E[max(0, K - e^X)].
    mc_estimate_2 = np.mean(np.maximum(strike - exp_samples_val, 0))
    # Need to figure out the standard error for both of these regular MC estimates.
    std_error_1 = np.std(np.maximum(strike - norm_samples, 0)) / np.sqrt(num_sims)
    std_error_2 = np.std(np.maximum(strike - exp_samples_val, 0)) / np.sqrt(num_sims)

    conf_interval_1 = (mc_estimate_1 - 1.96 * std_error_1, mc_estimate_1 + 1.96 * std_error_1)
    conf_interval_2 = (mc_estimate_2 - 1.96 * std_error_2, mc_estimate_2 + 1.96 * std_error_2)

    closed_form_sol_1 = 1 / (2 * math.pi) **(1/2)
    closed_form_sol_2 = strike * stats.norm.cdf(-1) - np.exp(3/2) * stats.norm.cdf(-2)

    possible_c_values = np.arange(-0.5, 0.6, 0.1)
    best_control_c = None 
    lowest_variance = float('inf') 
    control_estimates = {}

    for c_val in possible_c_values:
        # I'm adjusting the estimate for expectation_2 using expectation_1 as a control.
        control_variate_val = np.maximum(strike - exp_samples_val, 0) + c_val * (np.maximum(strike - norm_samples, 0) - closed_form_sol_1)
        controlled_estimate = np.mean(control_variate_val) # This is my new, controlled estimate
        controlled_std_error = np.std(control_variate_val) / np.sqrt(num_sims)
        controlled_ci = (controlled_estimate - 1.96 * controlled_std_error, controlled_estimate + 1.96 * controlled_std_error)
        control_estimates[c_val] = {"Estimate": controlled_estimate, "Confidence Interval": controlled_ci} 

        # Check to see if 'c' value gave me a better (lower) variance than before.
        if controlled_std_error < lowest_variance:
            lowest_variance = controlled_std_error 
            best_control_c = c_val

    return {
        "Regular MC Estimate (K-X)": mc_estimate_1, 
        "Confidence Interval (K-X)": conf_interval_1, 
        "Closed-Form Solution (K-X)": closed_form_sol_1, 
        "Regular MC Estimate (K-exp(X))": mc_estimate_2, 
        "Confidence Interval (K-exp(X))": conf_interval_2,
        "Closed-Form Solution (K-exp(X))": closed_form_sol_2,
        "Control Variate Estimates (for K-exp(X))": control_estimates, 
        "Optimal c": round(best_control_c, 2) 
    }

results = monte_carlo_control_variates_estimation()

print("\nProblem 4 Part 5 - Control Variates Monte Carlo Results\n")
for key, value in results.items():
    if key == 'Control Variate Estimates (for K-exp(X))':
        print("Control Variate Estimates (for K-exp(X)):")
        for c_val, res in results[key].items():
            print(f"  c = {c_val:.1f} : {res}") 
    else:
        print(f"{key}: {value}")