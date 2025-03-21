import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Parameters (example values)
A_G = 1.0  # GenCo's risk-aversion factor
A_L = 1.0  # LSE's risk-aversion factor
zeta_1 = 100  # Threat point utility for GenCo
zeta_2 = 150  # Threat point utility for LSE
mu_lambda = 500  # Mean of lambda (sum of LMPs over contract period)
sigma_lambda = 50  # Standard deviation of lambda
T = 24  # Contract period (hours)
alpha = 0.95  # Confidence level for CVaR
M_lower = 0  # Lower bound for contract amount
M_upper = 600  # Upper bound for contract amount
S_lower = 15  # Lower bound for strike price
S_upper = 25  # Upper bound for strike price

# Expected net earnings from the day-ahead market (example values)
E_G_pi_G_lambda = 1000  # GenCo's expected net earnings
E_L_pi_L_0 = 1200  # LSE's expected net earnings

# CVaR calculation for a normal distribution
def calculate_cvar(mean, std, alpha):
    """Calculate CVaR for a normal distribution."""
    phi_alpha = norm.pdf(norm.ppf(alpha))  # PDF of standard normal at alpha quantile
    return mean + std * (phi_alpha / (1 - alpha))

# Objective function
def objective_function(x):
    """Objective function to maximize (u_G - zeta_1) * (u_L - zeta_2)."""
    M, S = x  # Unpack decision variables

    # Expected earnings
    E_G = E_G_pi_G_lambda + (T * S - mu_lambda) * M
    E_L = E_L_pi_L_0 + (mu_lambda - T * S) * M

    # CVaR calculations
    CVaR_G = calculate_cvar(E_G_pi_G_lambda, sigma_lambda, alpha) + (T * S - mu_lambda) * M
    CVaR_L = calculate_cvar(E_L_pi_L_0, sigma_lambda, alpha) + (mu_lambda - T * S) * M

    # Utility functions
    u_G = E_G - A_G * CVaR_G
    u_L = E_L - A_L * CVaR_L

    # Objective: maximize (u_G - zeta_1) * (u_L - zeta_2)
    return -(u_G - zeta_1) * (u_L - zeta_2)  # Negative for minimization

# Bounds for M and S
bounds = [(M_lower, M_upper), (S_lower, S_upper)]

# Initial guess for M and S
initial_guess = [300, 20]  # Example initial guess

# Constraints (if any)
# For example, you could add constraints on M and S here if needed
constraints = []  # No additional constraints in this example

# Optimize using scipy.optimize.minimize
result = minimize(
    objective_function,
    initial_guess,
    bounds=bounds,
    constraints=constraints,
    method='SLSQP'  # Sequential Least Squares Programming
)

# Output the results
if result.success:
    optimal_M, optimal_S = result.x
    max_objective_value = -result.fun  # Negate to get the original objective value
    print(f"Optimal Contract Amount (M): {optimal_M}")
    print(f"Optimal Strike Price (S): {optimal_S}")
    print(f"Maximum Objective Value: {max_objective_value}")
else:
    print("Optimization failed:", result.message)