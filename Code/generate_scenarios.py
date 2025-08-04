"""Generate Monte Carlo scenarios and save them to CSV files.

This script generates price, production, capture rate, and load scenarios using Monte Carlo
simulation and saves them to CSV files for later use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# ───────────────────────── helpers ──────────────────────────

def _yearly_index(start: str | pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(start), periods=periods, freq="YS")
def _monthly_index(start: str | pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(start), periods=periods, freq="MS")


def _save_matrix(folder: Path, kind: str, mat: np.ndarray, start: str | pd.Timestamp,resample: bool):
    """Save a (years × sims) matrix as `<kind>_scenarios_<y>y_<s>s.csv`."""
    n_months = 12
    years, sims = mat.shape
   
    if resample == True:
      
        df = pd.DataFrame(mat, index=_monthly_index(start, years),
                      columns=pd.RangeIndex(sims, name="sim"))
         # For prices, use mean (average) instead of sum
        if kind == "price":
            df = df.resample("YE").mean()
        else:
            df = df.resample("YE").sum()
        fname = f"{kind}_scenarios_{years//n_months}y_{sims}s.csv"
    else:
        df = pd.DataFrame(mat, index=_yearly_index(start, years),
                      columns=pd.RangeIndex(sims, name="sim"))
        df = df.resample("YE").sum()
        fname = f"{kind}_scenarios_{years}y_{sims}s.csv"
        

    df.to_csv(folder / fname, index_label="year")


# ───────────────────────── price model ──────────────────────

@dataclass
class PriceModel:
    rng: np.random.Generator
    s: Optional[float] = None  # Lognormal parameters
    loc: Optional[float] = None
    scale: Optional[float] = None
    start_value: Optional[float] = None  # Starting value for OU process
    kappa: Optional[float] = None  # OU parameters
    theta: Optional[float] = None
    theta_1 : Optional[float] = None  # Trend parameter for OU process
    sigma: Optional[float] = None


    @classmethod
    def from_csv(cls, sampling_type: str, csv_path: str, seed: Optional[int] = None) -> "PriceModel":
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        
        df.index = pd.to_datetime(df["HourUTC"])

        # Exclude 2025 data
        #df = df[df.index.year < 2025]
        mean_price = df['DK2_EUR/MWh'].mean()
        std_price = df['DK2_EUR/MWh'].std()
        df_clean = df[(df['DK2_EUR/MWh'] > mean_price - 3*std_price) & 
                    (df['DK2_EUR/MWh'] < mean_price + 3*std_price)]

        print("Data shape after outlier removal:", df_clean.shape)

        if sampling_type == "OU_Process":  # Actually OU process
            # Resample to monthly data
            monthly = df_clean["DK2_EUR/MWh"].resample("ME").mean().to_numpy(float) * 1e-3  # Convert to Mio EUR/GWh
            start_value = monthly[-1]

            # Step 1: Prepare time series for OU parameter estimation with time-varying mean
            X = monthly
            X_t = X[:-1]
            X_tp1 = X[1:]

            # Time index (normalized to [0,1] for numerical stability)
            t_raw = np.arange(len(X_t))
            t_normalized = t_raw / len(X_t)  # Normalize time to [0,1]

            # Time step is 1 month = 1/12 years
            dt = 1/12
            dX_dt = (X_tp1 - X_t) / dt  # Convert to annual rate

            # Step 2: Linear regression with time-varying mean
            # Model: dX/dt = κ[θ(t) - X_t] = κ[(θ₀ + θ₁*t) - X_t]
            # Rearranged: dX/dt = κθ₀ + κθ₁*t - κX_t
            # Regression: dX/dt = a + b*t + c*X_t
            
            regression_matrix = np.column_stack([
                np.ones(len(X_t)),    # Constant term (a = κθ₀)
                t_normalized,         # Time trend (b = κθ₁)
                X_t                   # Lagged price level (c = -κ)
            ])
            
            model = sm.OLS(dX_dt, regression_matrix)
            results = model.fit()

            # Step 3: Extract OU parameters with time-varying mean
            a = results.params[0]  # κθ₀
            b = results.params[1]  # κθ₁  
            c = results.params[2]  # -κ
            
            # Solve for OU parameters
            kappa = -c
            theta_0 = a / kappa if kappa != 0 else X.mean()
            theta_1_normalized = b / kappa if kappa != 0 else 0
            
            # Convert theta_1 back to original time scale (per year)
            theta_1_annual = theta_1_normalized / len(X_t) * 12  # Convert to per year
            
            # Estimate volatility from residuals
            sigma = results.resid.std() * np.sqrt(dt)  # Volatility per √year

            print(f"Estimated OU parameters with time-varying θ(t):")
            print(f"  κ (mean reversion speed) = {kappa:.4f} per year")
            print(f"  θ₀ (initial long-run mean) = {theta_0:.4f} Mio EUR/GWh")
            print(f"  θ₁ (trend slope) = {theta_1_annual:.6f} Mio EUR/GWh per year")
            print(f"  σ (volatility) = {sigma:.4f} Mio EUR/GWh per √year")
            

            return cls(
                rng=np.random.default_rng(seed),
                s=None, 
                loc=None, 
                scale=None,
                start_value=start_value,
                kappa=kappa, 
                theta=theta_0,  # Store initial theta
                theta_1=theta_1_annual,  # Store trend parameter
                sigma=sigma
            )

        else:
            # Lognormal distribution fitting
            monthly = df_clean["DK2_EUR/MWh"][1:].resample("ME").mean().to_numpy(float) * 1e-3
            s, loc, scale = stats.lognorm.fit(monthly)
            return cls(
                rng=np.random.default_rng(seed),
                s=s, 
                loc=loc, 
                scale=scale,
                start_value=None,
                kappa=None,
                theta=None,
                theta_1=None,
                sigma=None
            )

    def simulate(self, sampling_type: str, years: int, sims: int) -> np.ndarray:
        if sampling_type == "OU_Process":  # Actually OU process
            # Generate monthly data points
            n_steps = years * 12  # Total number of monthly steps
            dt = 1/12  # Monthly time step in years
            
            all_simulations = []
            
            for i in range(sims):
                # Initialize the process with the starting value
                process_values = [self.start_value]
                
                for j in range(n_steps):
                    current_time_years = j * dt  # Time in years from start
                    current_value = process_values[-1]
                    
                    # Time-varying long-run mean: θ(t) = θ₀ + θ₁*t
                    theta_t = self.theta + self.theta_1 * current_time_years
                
                    
                    # OU process: dX = κ[θ(t) - X]dt + σdW
                    drift = self.kappa * (theta_t - current_value) * dt
                    diffusion = self.sigma * np.sqrt(dt) * self.rng.normal(0, 1)
                    
                    next_value = current_value + drift + diffusion
                    
                    # Ensure non-negative prices (reflecting barrier at 0)
                    next_value = max(next_value, 0)
                    
                    process_values.append(next_value)
                
                # Remove the initial value and keep only the simulated path
                all_simulations.append(process_values[1:])
            
            # Convert to array with shape (n_steps, sims)
            all_simulations = np.array(all_simulations).T
            
            return all_simulations
        
        else:
            # Lognormal distribution sampling
            n_months = 12 
            return stats.lognorm.rvs(s=self.s, loc=self.loc, scale=self.scale, 
                size=(n_months * years, sims), random_state=self.rng
            )
                                    


# ──────────────────── production model (GEV) ─────────────────

@dataclass
class ProductionModel:
    c: float
    loc: float
    scale: float
    cap_gwh: Optional[float]
    rng: np.random.Generator

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        capacity_mw: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "ProductionModel":
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        monthly = df.set_index("time")["electricity"].resample("ME").sum().to_numpy(float)
        c, loc, scale = stats.dweibull.fit(monthly)
        cap_gwh = capacity_mw * 8760 / 1000 if capacity_mw else None
        return cls(c, loc, scale, cap_gwh, np.random.default_rng(seed))

    def simulate(self, years: int, sims: int) -> np.ndarray:
        n_months = 12
        draws = stats.dweibull.rvs(
            c=self.c, loc=self.loc, scale=self.scale,
            size=(n_months * years, sims), random_state=self.rng
        ) / 1000  # MWh → GWh
        if self.cap_gwh is not None:
            np.clip(draws, 0, self.cap_gwh, out=draws)
        return draws


# ───────────────────── capture‑rate model ───────────────────

@dataclass
class CaptureRateModel:
    price_mu: float
    price_std: float
    prod_mu: float
    prod_std:float
    corr_agg:float
    z_year_corr:np.ndarray
    mu_z_corr:float
    std_z_corr:float
    rng: np.random.Generator

    @classmethod
    def from_csv(cls, csv_path: str, seed: Optional[int] = None) -> "CaptureRateModel":
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        df["HourUTC"] = pd.to_datetime(df["HourUTC"])
        df = df.set_index("HourUTC")

        # try to locate a production column; fall back to flat 1.0 rate
        prod_col = next((c for c in df.columns if "MWhDK2" in c or "prod" in c.lower()), None)
        if prod_col is None:
            return cls(1.0, 0.05, np.random.default_rng(seed))

        # Mean and std for price
        price_mu, price_std = df['DK2_EUR/MWh'].mean(), df['DK2_EUR/MWh'].std()  # EUR/MWh

        # Mean and std for production
        prod_mu, prod_std = df['OnshoreWindGe50kW_MWhDK2'].mean(), df['OnshoreWindGe50kW_MWhDK2'].std()  # MWh

        # Correlation across price and production [hour] to maintain correlation
        # Add a 'year' column
        df['year'] = df.index.year

        # Compute hourly correlation between Onshore Wind and Price for each year
        corr_by_year = (
            df.groupby('year')[['OnshoreWindGe50kW_MWhDK2', 'DK2_EUR/MWh']]
            .corr()
            .iloc[0::2, 1]  # Select cross-correlations only
            .reset_index()
            .rename(columns={'DK2_EUR/MWh': 'hourly_corr'})
            .drop('level_1', axis=1))
        corr_by_year_arr = corr_by_year["hourly_corr"].to_numpy()[1:]

        corr_agg = df["DK2_EUR/MWh"].corr(df["OnshoreWindGe50kW_MWhDK2"])

        z_year_corr   = np.arctanh(corr_by_year_arr)   # Fisher-z
        std_z_corr  = z_year_corr.std(ddof=1)      # unbiased SD
        mu_z_corr     = z_year_corr.mean()    

        

        return cls(price_mu,price_std, prod_mu,prod_std,corr_agg,z_year_corr,mu_z_corr,std_z_corr, np.random.default_rng(seed))

    def simulate(self, years: int, sims: int) -> np.ndarray:

        noise    = self.rng.normal(0, self.std_z_corr, size=(years,sims))
        rho_samples = self.mu_z_corr + noise
        rho_samp  = np.tanh(rho_samples)                                # ρ ∈ (-1,1)

        CR = 1 + rho_samp * (self.price_std/self.price_mu) *(self.prod_std/self.prod_mu)
        return CR
    

# ───────────────────── Load-Capture model ───────────────────

@dataclass
class LoadRateModel:
    price_mu: float
    price_std: float
    consump_mu: float
    consump_std:float
    corr_agg:float
    z_year_corr:np.ndarray
    mu_z_corr:float
    std_z_corr:float
    rng: np.random.Generator

    @classmethod
    def from_csv(cls, csv_path_price: str,csv_path_consumption:str, seed: Optional[int] = None) -> "LoadRateModel":
        df_price = pd.read_csv(csv_path_price, sep=";", decimal=",")
        df_price["HourUTC"] = pd.to_datetime(df_price["HourUTC"])
        df_price = df_price.set_index("HourUTC")

        df_cons = pd.read_csv(csv_path_consumption, sep=";", decimal=",")
        df_cons["HourUTC"] = pd.to_datetime(df_cons["HourUTC"])
        df_cons = df_cons.set_index("HourUTC")
        df_cons.drop(columns=['HourDK','MunicipalityNo'], inplace=True)
        # Only keep rows where 'Branche' == 'Erhverv'
        df_cons = df_cons[df_cons['Branche'] == 'Erhverv']
        df_cons["ConsumptionMWh"] = df_cons["ConsumptionkWh"]/1000
        df_combined = pd.concat([df_price, df_cons], axis=1).dropna()

            
                # Mean and std for price
        price_mu, price_std = df_combined['DK2_EUR/MWh'].mean(), df_combined['DK2_EUR/MWh'].std()  # EUR/MWh

        # Mean and std for production
        consump_mu, consump_std = df_combined['ConsumptionMWh'].mean(), df_combined['ConsumptionMWh'].std()  # MWh

        # Correlation across price and production [hour] to maintain correlation
        # Add a 'year' column
        df_combined['year'] = df_combined.index.year

        # Compute hourly correlation between Consumption  and Price for each year
        corr_by_year = (
            df_combined.groupby('year')[['ConsumptionMWh', 'DK2_EUR/MWh']]
            .corr()
            .iloc[0::2, 1]  # Select cross-correlations only
            .reset_index()
            .rename(columns={'DK2_EUR/MWh': 'hourly_corr'})
            .drop('level_1', axis=1))
        corr_by_year_arr = corr_by_year["hourly_corr"].to_numpy()[1:]

        corr_agg = df_combined["DK2_EUR/MWh"].corr(df_combined["ConsumptionMWh"])

        z_year_corr   = np.arctanh(corr_by_year_arr)   # Fisher-z
        std_z_corr  = z_year_corr.std(ddof=1)      # unbiased SD
        mu_z_corr     = z_year_corr.mean()    

        return cls(price_mu,price_std, consump_mu,consump_std,corr_agg,z_year_corr,mu_z_corr,std_z_corr, np.random.default_rng(seed))

    def simulate(self, years: int, sims: int) -> np.ndarray:
        
        noise    = self.rng.normal(0, self.std_z_corr, size=(years,sims))
        rho_samples = self.mu_z_corr + noise
        rho_samp  = np.tanh(rho_samples)                                # ρ ∈ (-1,1)

        CR = 1 + rho_samp * (self.price_std/self.price_mu) *(self.consump_std/self.consump_mu)
 
        return CR


# ───────────────────────── load model ───────────────────────


@dataclass
class LoadModel:
    a: float
    b: float
    loc: float
    scale: float
    rng: np.random.Generator

    @classmethod
    def from_csv(cls,csv_path_consumption:str, seed: Optional[int] = None) -> "LoadModel":
       

        df = pd.read_csv(csv_path_consumption, sep=";", decimal=",")
        df["HourUTC"] = pd.to_datetime(df["HourUTC"])
        df = df.set_index("HourUTC")
        df.drop(columns=['HourDK','MunicipalityNo'], inplace=True)
        # Only keep rows where 'Branche' == 'Erhverv'
        df = df[df['Branche'] == 'Erhverv']
        df["ConsumptionGWh"] = df["ConsumptionkWh"] * 1e-6 # Convert kWh to GWh


        monthly = df["ConsumptionGWh"][1:].resample("ME").sum().to_numpy(float) 
        a,b, loc, scale = stats.beta.fit(monthly)
        return cls(a, b, loc, scale, np.random.default_rng(seed))

    def simulate(self, years: int, sims: int) -> np.ndarray:
        n_months = 12
        return stats.beta.rvs(a=self.a, b=self.b, loc=self.loc, scale=self.scale, 
                                size=(n_months* years, sims), random_state=self.rng)



# ──────────── one‑shot scenario generator ───────────

def run_scenarios(
    *,
    years: int,
    num_scenarios: int,
    start_time: str | pd.Timestamp,
    price_csv_path: str,
    prod_csv_path: str,
    consumption_csv_path: str,
    capacity_mw: Optional[float],
    output_dir: str = "outputs",
    seed: int = 42,
):
    """Generate and save **price, production, capture‑rate and load** scenarios."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 1) price (OU or lognormal)
    #sampling_type = "OU_Process"  # or "Lognormal"
    sampling_type = "OU_Process"  # or "Lognormal"
    price_mdl = PriceModel.from_csv(sampling_type,  price_csv_path, seed)
    price_mat = price_mdl.simulate(sampling_type, years, num_scenarios)
    _save_matrix(out, "price", price_mat, start_time,resample=True)

    # 2) production
    prod_mdl = ProductionModel.from_csv(prod_csv_path, capacity_mw, seed)
    prod_mat = prod_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "production", prod_mat, start_time,resample=True)

    # 3) capture‑rate (just needs price file that already has production col)
    cr_mdl = CaptureRateModel.from_csv(price_csv_path, seed)
    cr_mat = cr_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "capture_rate", cr_mat, start_time,resample=False)

    # 4) load
    load_mdl = LoadModel.from_csv(consumption_csv_path, seed)
    load_mat = load_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "load", load_mat, start_time,resample=True)
    # 5) load capture rate
    load_mdl = LoadRateModel.from_csv(price_csv_path,consumption_csv_path, seed)
    load_cr_mat = load_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "load_capture_rate", load_cr_mat, start_time,resample=False)

    print(f"Wrote scenarios to {out.resolve()}")

if __name__ == "__main__":
    # Example usage
    years = 5  # 5 years
    num_scenarios = 50000  # Reduced from 50000 for faster computation

    # Wind Profile 
    csv_wind = "Code/Data/Wind/combined_wind_data.csv"  # adjust path if needed
    csv_solar = "Code/Data/Solar/combined_solar_data.csv"  # adjust path if needed

    start_time = pd.Timestamp("2025-01-01")
    price_csv_path =  "Code/Data/EnergyReport.csv"
    prod_csv_path = csv_wind
    consumption_csv_path =  "Code/Data/ConsumptionIndustry.csv"  # Not used in this script, but can be added if needed
    capacity_mw = 30  # MW
    output_dir = "Code/scenarios"    
    run_scenarios(
        years=years,
        num_scenarios=num_scenarios,
        start_time=start_time,
        price_csv_path=price_csv_path,
        prod_csv_path=prod_csv_path,
        consumption_csv_path=consumption_csv_path,
        capacity_mw=capacity_mw,
        output_dir=output_dir
    )
    print("All scenarios generated successfully")
