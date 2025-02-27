import numpy as np
import scipy.optimize as opt

def cvar(losses, alpha=0.95):
    """ Compute Conditional Value-at-Risk (CVaR) """
    var = np.percentile(losses, (1 - alpha) * 100)
    cvar = losses[losses >= var].mean()
    return cvar

def utility(expected_earnings, risk_aversion, cvar):
    """ Return-risk utility function """
    return expected_earnings - risk_aversion * cvar

def nash_bargaining(q, p, expected_genco, expected_lse, risk_aversion_genco, risk_aversion_lse, cvar_genco, cvar_lse, q_bounds, p_bounds):
    """ Nash bargaining solution """
    def objective(x):
        q, p = x
        u_genco = utility(expected_genco(q, p), risk_aversion_genco, cvar_genco(q, p))
        u_lse = utility(expected_lse(q, p), risk_aversion_lse, cvar_lse(q, p))
        return -(u_genco - u_genco_no_contract) * (u_lse - u_lse_no_contract)  # Nash product
    
    bounds = [q_bounds, p_bounds]
    result = opt.minimize(objective, x0=[(q_bounds[0] + q_bounds[1]) / 2, (p_bounds[0] + p_bounds[1]) / 2], bounds=bounds)
    return result.x if result.success else None

# Example definitions of earnings and CVaR for GenCo and LSE
def expected_genco(q, p):
    return q * (p - 30)  # Example: selling electricity at strike price minus cost

def expected_lse(q, p):
    return q * (50 - p)  # Example: buying electricity at strike price lower than retail price

def cvar_genco(q, p):
    losses = np.random.normal(5, 2, 1000) * q  # Example loss distribution
    return cvar(losses)

def cvar_lse(q, p):
    losses = np.random.normal(4, 1.5, 1000) * q  # Example loss distribution
    return cvar(losses)

# Load profile with variability (MW per hour for a 24-hour period)
base_load_profile = np.array([300, 280, 270, 260, 250, 240, 230, 250, 300, 350, 400, 420, 
                              430, 420, 410, 400, 390, 380, 370, 360, 350, 340, 320, 310])
load_variability = np.random.normal(0, 30, (1000, 24))  # Variability in load
load_scenarios = base_load_profile + load_variability
load_scenarios = np.clip(load_scenarios, 200, 500)  # Ensuring reasonable limits

# No contract utility levels
u_genco_no_contract = 0  # Assume GenCo has 0 net earnings without contract
u_lse_no_contract = 0    # Assume LSE has 0 net earnings without contract

# Risk aversion factors
risk_aversion_genco = 1.0
risk_aversion_lse = 1.5

# Monte Carlo simulation to determine optimal contract parameters
q_samples = []
p_samples = []
for i in range(1000):
    q_opt, p_opt = nash_bargaining(q_samples, p_samples, expected_genco, expected_lse, risk_aversion_genco, risk_aversion_lse, cvar_genco, cvar_lse, q_samples, p_samples)
    q_samples.append(q_opt)
    p_samples.append(p_opt)

q_final = np.mean(q_samples)
p_final = np.mean(p_samples)

print(f"Optimal contract quantity (Monte Carlo): {q_final} MW, Optimal strike price: ${p_final}/MWh")

# Adjust contract based on load profile
contracted_power = np.minimum(base_load_profile, q_final)
print(f"Adjusted contract quantities based on load profile: {contracted_power}")
