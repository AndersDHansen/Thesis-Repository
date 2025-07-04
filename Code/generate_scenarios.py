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
    s:float
    loc: float
    scale: float
    rng: np.random.Generator

    @classmethod
    def from_csv(cls, csv_path: str, seed: Optional[int] = None) -> "PriceModel":
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        df.index = pd.to_datetime(df["HourUTC"])
        mean_price = df['DK2_EUR/MWh'].mean()
        std_price = df['DK2_EUR/MWh'].std()
        df_clean = df[(df['DK2_EUR/MWh'] > mean_price - 3*std_price) & 
                    (df['DK2_EUR/MWh'] < mean_price + 3*std_price)]

        print("Data shape after outlier removal:", df_clean.shape)

        # Remove negative prices if they exist and don't make economic sense
        #df_clean = df_clean[df_clean[price_col] > 0]
        monthly = df["DK2_EUR/MWh"][1:].resample("ME").mean().to_numpy(float) *1e-3 # Convert from EUR/MWh to Mio EUR/GWh
        s, loc, scale = stats.lognorm.fit(monthly)
        return cls(s, loc, scale, np.random.default_rng(seed))
        
    def simulate(self, years: int, sims: int) -> np.ndarray:
        n_months = 12
        return stats.lognorm.rvs(s=self.s, loc=self.loc, scale=self.scale, 
                                size=(n_months * years, sims), random_state=self.rng)


# ──────────────────── production model (GEV) ─────────────────

@dataclass
class ProductionModel:
    shape: float
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
        shape, loc, scale = stats.dweibull.fit(monthly)
        cap_gwh = capacity_mw * 8_760 / 1000 if capacity_mw else None
        return cls(shape, loc, scale, cap_gwh, np.random.default_rng(seed))

    def simulate(self, years: int, sims: int) -> np.ndarray:
        n_months = 12
        draws = stats.dweibull.rvs(
            self.shape, loc=self.loc, scale=self.scale,
            size=(n_months*years, sims), random_state=self.rng
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
        
        noise    = np.random.normal(0, self.std_z_corr, size=(years,sims))
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
        
        noise    = np.random.normal(0, self.std_z_corr, size=(years,sims))
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
    def from_csv(cls,csv_path_consumption:str, seed: Optional[int] = None) -> "LoadRateModel":
       

        df = pd.read_csv(csv_path_consumption, sep=";", decimal=",")
        df["HourUTC"] = pd.to_datetime(df["HourUTC"])
        df = df.set_index("HourUTC")
        df.drop(columns=['HourDK','MunicipalityNo'], inplace=True)
        # Only keep rows where 'Branche' == 'Erhverv'
        df = df[df['Branche'] == 'Erhverv']
        df["ConsumptionGWh"] = df["ConsumptionkWh"] * 1e-6 # Convert kWh to GWh

            
        monthly = df["ConsumptionGWh"][1:].resample("ME").sum().to_numpy(float) 
        a,b, loc, scale = stats.beta.fit(monthly)
        return cls(a,b, loc, scale, np.random.default_rng(seed))
        
    def simulate(self, years: int, sims: int) -> np.ndarray:
        n_months = 12
        return stats.beta.rvs(a=self.a, b= self.b, loc=self.loc, scale=self.scale, 
                                size=(n_months * years, sims), random_state=self.rng)



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

    # 1) price
    price_mdl = PriceModel.from_csv(price_csv_path, seed)
    price_mat = price_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "price", price_mat, start_time,resample=True)

    # 2) production
    prod_mdl = ProductionModel.from_csv(prod_csv_path, capacity_mw, seed)
    prod_mat = prod_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "production", prod_mat, start_time,resample =True)

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
    years = 5  # 10 years
    num_scenarios = 10000

    # Wind Profile 
    csv_wind = "Code/Data/Wind/combined_wind_data.csv"  # adjust path if needed
    csv_solar = "Code/Data/Solar/combined_solar_data.csv"  # adjust path if needed

    start_time = pd.Timestamp("2025-01-01")
    price_csv_path =  "Code/Data/EnergyReport.csv"
    prod_csv_path = csv_wind
    consumption_csv_path =  "Code/Data/ConsumptionIndustry.csv"  # Not used in this script, but can be added if needed
    capacity_mw = 100  # MW
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
