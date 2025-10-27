"""
Visualization module for power system contract negotiation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from utils import calculate_cvar_left, weighted_expected_value
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

cmap_red_green=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
# Create a custom colormap that transitions from white to very light gray



class Plotting_Class:
    """
    Handles plotting of results from the power system contract negotiation simulation.
    """
    def __init__(self, contract_model_data, CP_results_df, CP_earnings_df,
                 risk_sensitivity_df, risk_earnings_df, 
                 price_bias_sensitivity_df, 
                 price_sensitivity_mean_df, price_sensitivity_std_df,
                 production_bias_sensitivity_df,
                 production_sensitivity_mean_df, production_sensitivity_std_df,
                 load_sensitivity_mean_df, load_sensitivity_std_df,
                 gen_CR_sensitivity_df, load_CR_sensitivity_df,
                 boundary_results_df_price, boundary_results_df_production, negotiation_sensitivity_df,negotiation_earnings_df,
                 negotiation_vs_risk_df,elasticity_vs_risk_df , bias_risk_elasticity_df,
                 styles=None):
        
        self.cm_data = contract_model_data
        self.CP_results_df = CP_results_df
        self.CP_earnings_df = CP_earnings_df
        self.risk_sensitivity_df = risk_sensitivity_df
        self.earnings_risk_sensitivity_df = risk_earnings_df
        self.price_bias_sensitivity_df = price_bias_sensitivity_df
        self.production_bias_sensitivity_df = production_bias_sensitivity_df
        self.price_sensitivity_mean_df = price_sensitivity_mean_df
        self.price_sensitivity_std_df = price_sensitivity_std_df
        self.production_sensitivity_mean_df = production_sensitivity_mean_df
        self.production_sensitivity_std_df = production_sensitivity_std_df
        self.load_sensitivity_mean_df = load_sensitivity_mean_df
        self.load_sensitivity_std_df = load_sensitivity_std_df
        self.load_CR_sensitivity_df = load_CR_sensitivity_df
        self.gen_CR_sensitivity_results = gen_CR_sensitivity_df
        self.boundary_results_price = boundary_results_df_price
        self.boundary_results_production = boundary_results_df_production
        self.negotiation_sensitivity_df = negotiation_sensitivity_df
        self.negotiation_earnings_df = negotiation_earnings_df
        self.negotiation_vs_risk_df = negotiation_vs_risk_df
        self.elasticity_vs_risk_df = elasticity_vs_risk_df
        self.bias_risk_elasticity_df = bias_risk_elasticity_df
        # plotting styles
        self.legendsize = 12+2
        self.labelsize = 16+2
        self.titlesize = 17+2
        self.suptitlesize = 19+1
        #self.alpha_sensitivity_df = alpha_sensitivity_df
        #self.alpha_earnings_df = alpha_earnings_df

        self.plots_dir = os.path.join(os.path.dirname(__file__), 'Plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def _safe_local_elasticity_single(self, df_in: pd.DataFrame, factor_col: str, metric_col: str, baseline: float) -> float | None:
        """Single-metric local elasticity with central difference or local linear fit. Returns float or None/NaN.
        E = (dY/dX) * (X0 / Y0), never fills NaN with zeros.
        """
        if df_in is None or df_in.empty:
            return np.nan
        cols = [factor_col, metric_col]
        if any(c not in df_in.columns for c in cols):
            return np.nan
        df = df_in[cols].copy()
        df = df[np.isfinite(df[factor_col]) & np.isfinite(df[metric_col])]
        if df.empty:
            return np.nan
        df = df.sort_values(factor_col)
        # Round to 5 decimals to remove floating noise
        x = df[factor_col].astype(float).round(5).values
        y = df[metric_col].astype(float).round(5).values
        # Find neighbors around baseline
        left = np.where(x < baseline)[0]
        right = np.where(x > baseline)[0]
        y0 = np.nan
        slope = np.nan
        eq = np.where(np.isclose(x, baseline))[0]
        if eq.size > 0:
            y0 = y[eq[0]]
        if left.size > 0 and right.size > 0:
            iL = left[-1]
            iR = right[0]
            xL, yL = x[iL], y[iL]
            xR, yR = x[iR], y[iR]
            if xR != xL:
                slope = (yR - yL) / (xR - xL)
                if not np.isfinite(y0):
                    y0 = yL + (baseline - xL) * slope
        if not np.isfinite(slope):
            if x.size >= 2 and np.unique(x).size >= 2:
                k = min(5, x.size)
                order = np.argsort(np.abs(x - baseline))[:k]
                coeffs = np.polyfit(x[order], y[order], deg=1)
                slope = coeffs[0]
                y0 = np.polyval(coeffs, baseline)
        if not np.isfinite(slope) or not np.isfinite(y0) or np.isclose(y0, 0.0):
            return np.nan
        return float(slope * (baseline / y0))

    def _color_for_risk(self, value: float, kind: str = 'A_L'):
        """Return a consistent color for a given risk value (A_L or A_G),
        aligned with the Set2 palette used elsewhere.
        We snap to a canonical set of values for stable colors across plots.
        """
        try:
            v = round(float(value), 2)
        except Exception:
            v = value
        canonical = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        base_pal = sns.color_palette('Set2', n_colors=len(canonical))
        # Slightly darken to avoid too-light tones
        pal = [tuple(min(1.0, max(0.0, c*0.85)) for c in rgb) for rgb in base_pal]
        if v in canonical:
            idx = canonical.index(v)
        else:
            # Nearest bucket for unseen value
            idx = np.searchsorted(canonical, v)
            idx = max(0, min(idx, len(canonical)-1))
        return pal[idx]

    def _plot_negotiation_vs_risk(self, metric='StrikePrice', filename=None):
        """
        Plot metric vs tau_L for multiple (A_G, A_L) pairs.

        - Color encodes the (A_G, A_L) pair using a palette.
        - Lines are drawn across tau_L; optional secondary axis for Gamma if PAP and metric is ContractAmount.
        """

        df = self.negotiation_vs_risk_df
        if df is None or df.empty:
            print("No data provided for negotiation vs risk plot.")
            return

        # Prepare data
        plot_df = df.copy()
        if 'tau_L' not in plot_df.columns:
            print("Dataframe missing tau_L; cannot plot.")
            return
        # Round contract amount to 4 decimals for readability
        if 'ContractAmount' in plot_df.columns:
            try:
                plot_df['ContractAmount'] = plot_df['ContractAmount'].astype(float).round(4)
            except Exception:
                pass

        # Unique pairs and grouped color mapping: similar colors per A_G, distinct per A_L
        pairs = plot_df[['A_G', 'A_L']].drop_duplicates().sort_values(['A_G', 'A_L']).values.tolist()
        unique_al = sorted(plot_df['A_L'].unique())
        color_map = {}
        base_palette = sns.color_palette('Set2', n_colors=max(3, len(unique_al)))
        for idx_ag, al in enumerate(unique_al):
            base = base_palette[idx_ag % len(base_palette)]
            ags = sorted(plot_df.loc[plot_df['A_L'] == al, 'A_G'].unique())
            # Generate base-to-darker shades so higher A_L lines are not too light
            base_rgb = to_rgb(base)
            dark_rgb = tuple(max(0.0, c * 0.55) for c in base_rgb)  # darker companion
            shades = sns.blend_palette([base_rgb, dark_rgb], n_colors=max(3, len(ags)))
            for j, ag in enumerate(ags):
                color_map[(ag, al)] = shades[j]

        # Sort by tau_L for nice lines
        plot_df = plot_df.sort_values('tau_L')

        is_pap = self.cm_data.contract_type == "PAP"
        has_gamma = 'Gamma' in plot_df.columns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metrics = ['StrikePrice', 'ContractAmount']
        titles = ['Strike Price', 'Contract Amount']

        for ax, m, title in zip(axes, metrics, titles):
            for pair in pairs:
                a_g, a_l = pair
                sub = plot_df[(plot_df['A_G'] == a_g) & (plot_df['A_L'] == a_l)]
                if sub.empty:
                    continue
                # Emphasize the symmetric mid-risk case (A_G = A_L = 0.5)
                is_mid = np.isclose(a_g, 0.5) and np.isclose(a_l, 0.5)
                lw = 3.0 if is_mid else 2.0
                ms = 8 if is_mid else 6
                z = 4 if is_mid else 2
                edge_color = 'black' if is_mid else 'none'
                marker = 'o' if is_mid else 'none'
                ax.plot(sub['tau_L'], sub[m], linewidth=lw, markersize=ms,
                        color=color_map[(a_g, a_l)], label=f"A_G={a_g}, A_L={a_l}", marker=marker, zorder=z, markeredgecolor=edge_color)

            # Reference lines
            if m == 'StrikePrice':
                ref = (self.cm_data.Capture_price_G_avg if is_pap else self.cm_data.expected_price) * 1e3
                ax.axhline(ref, color='black', linestyle='--', label='Reference price')
                ax.set_ylabel("Price (EUR/MWh)", fontsize=self.labelsize)
            elif m == 'ContractAmount' and not is_pap:
                ref = self.cm_data.expected_production / 8760 * 1000
                ax.axhline(ref, color='black', linestyle='--', label='Expected production (MWh)')
                ax.set_ylabel("Contract Amount (MWh)", fontsize=self.labelsize)

            # Secondary axis for Gamma in PAP on ContractAmount
            if m == 'ContractAmount' and is_pap and has_gamma:
                ax.set_ylabel("Contract Amount (MW)", fontsize=self.labelsize)
                ax2 = ax.twinx()
                for pair in pairs:
                    a_g, a_l = pair
                    sub = plot_df[(plot_df['A_G'] == a_g) & (plot_df['A_L'] == a_l)]
                    if sub.empty or 'Gamma' not in sub.columns:
                        continue
                    is_mid = np.isclose(a_g, 0.5) and np.isclose(a_l, 0.5)
                    lw = 2.0 if is_mid else 1.0
                    z = 4 if is_mid else 2
                    edge_color = 'black' if is_mid else 'none'
                    marker = 'o' if is_mid else 'none'
                    ax2.plot(sub['tau_L'], sub['Gamma'] * 100, linestyle='--', linewidth=lw,
                             color=color_map[(a_g, a_l)], alpha=0.6, zorder=z, markeredgecolor=edge_color, marker=marker)
                ax2.set_ylabel('$\\gamma$ share of production capacity', color='gray', fontsize=self.labelsize-2)
                ax2.tick_params(axis='y', labelcolor='gray')


            ax.set_xlabel('Load Negotiation Power $\\tau_L$', fontsize=self.labelsize)
            ax.set_title(title, fontsize=self.titlesize)
            ax.grid(True, alpha=0.3)

            # Avoid scientific notation/offset text and clamp y-limits for Gamma on Contract Amount (PAP only)
            if m == 'ContractAmount' and is_pap and has_gamma:
                ax2.set_ylim(99, 101)

        # Build a single legend from unique pairs
        handles = [mpatches.Patch(color=color_map[(pair[0], pair[1])], label=f"A_G={pair[0]}, A_L={pair[1]}") for pair in pairs]
        # Place legend below plots to avoid overlapping the title
        #fig.legend(handles=handles, loc='lower center', ncol=min(3, len(handles)), bbox_to_anchor=(0.5, -0.02))
        fig.legend(
            handles=handles,
            loc='upper center',
            ncol=3,                    # 3 columns
            bbox_to_anchor=(0.5, 0.02),
            frameon=False
        )
        fig.suptitle(f"{self.cm_data.contract_type}: Negotiation Power vs Risk Aversion", fontsize=self.suptitlesize)
        plt.tight_layout(rect=[0, 0.0, 1, 1])
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()


    def _plot_3D_sensitivity_results(self, sensitivity_type, filename=None):
        
        """        Generalized function to plot 3D sensitivity analysis results."""

        """
        Generate individual 3D plots for each metric in sensitivity analysis.
        """
        
        if sensitivity_type == 'risk':
            df = self.risk_sensitivity_df
            x_col = 'A_L'
            y_col = 'A_G' 
            xlabel = 'Risk Aversion $A_L$'
            ylabel = 'Risk Aversion $A_G$'
            title = 'Risk Aversion Sensitivity Analysis'
        else:
            return print(f"Unknown sensitivity type: {sensitivity_type}")
        
        # Determine metrics to plot
        is_pap = self.cm_data.contract_type == "PAP"
        if is_pap and 'Gamma' in df.columns:
            metrics = ['StrikePrice', 'Gamma','Nash_Product']
            z_labels = ['Strike Price (EUR/MWh)', 'Gamma (%)', 'Nash_Product']
        else:
            metrics = ['StrikePrice', 'ContractAmount', 'Nash_Product']
            z_labels = ['Strike Price (EUR/MWh)', 'Contract Amount (MWh)', 'Nash_Product']

        for i, (metric, z_label) in enumerate(zip(metrics, z_labels)):
            # Create individual figure for each metric
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create pivot table without filling zeros
            pivot_table = df.pivot_table(
                index=x_col,
                columns=y_col,
                values=metric,
                aggfunc='mean'
            )
            
            # Interpolate missing values
            pivot_table = pivot_table.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
            
            # Create meshgrid
            X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
            Z = pivot_table.values
            
            # Create surface plot
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            
            # Adjust z-limits to focus on meaningful values
            #z_min, z_max = Z.min(), Z.max()
            #z_range = z_max - z_min
            #ax.set_zlim(z_min - 0.1*z_range, z_max + 0.1*z_range)
            
            # Add scatter points for actual data
            mask = df[metric] > 0  # Only plot non-zero values
            ax.scatter(df.loc[mask, y_col], df.loc[mask, x_col], df.loc[mask, metric], 
                    color='red', s=50, alpha=0.6)
            
            # Labels and title
            ax.set_xlabel(ylabel, fontsize=16, labelpad=12)
            ax.set_ylabel(xlabel, fontsize=16, labelpad=12)
            ax.set_zlabel(z_label, fontsize=16, labelpad=12)
            ax.set_title(f'{self.cm_data.contract_type}: {title}\n{metric}', fontsize=14)

            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
            ax.view_init(elev=30, azim=0)  # adjust as needed

            #for azim in range(0, 360, 15):
            #    ax.view_init(elev=30, azim=azim)
            #    plt.savefig(f'plot_elev30_azim{azim}.png')

            
                
            plt.tight_layout()
            
            if filename:
                filename = f"{metric}_{filename}" if not filename else filename
                filepath = os.path.join(self.plots_dir, filename)
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                print(f"Plot saved to {filepath}")
                plt.close(fig)
            else:
                plt.show()

    def _plot_sensitivity_results_heatmap(self,sensitivity_type,filename=None):

        """
        Generalized function to plot sensitivity analysis results using heatmaps.
        
        Parameters:
        -----------
        sensitivity_type : str
            Type of sensitivity analysis ('risk', 'price_bias', 'production_bias')
        filename : str, optional
            Filename to save the plot
        """
        
        # Configuration dictionary for different sensitivity types
        config = {
            'risk': {
                'df': self.risk_sensitivity_df,
                'title': 'Risk Aversion Sensitivity on Strike Price and Contract Amount',
                'index_col': 'A_L',
                'columns_col': 'A_G',
                'xlabel': 'Risk Aversion $A_S$',
                'ylabel': 'Risk Aversion $A_B$'
            },
            'price_bias': {
                'df': self.price_bias_sensitivity_df,
                'title': 'Price Bias Sensitivity on Strike Price and Contract Amount',
                'index_col': 'KL_Factor',
                'columns_col': 'KG_Factor',
                'xlabel': 'Generator Bias Factor (%)',
                'ylabel': 'Load Bias Factor (%)'
            },
            'production_bias': {
                'df': self.production_bias_sensitivity_df,
                'title': 'Production Bias Sensitivity on Strike Price and Contract Amount',
                'index_col': 'KL_Factor',
                'columns_col': 'KG_Factor',
                'xlabel': 'Generator Bias Factor (%)',
                'ylabel': 'Load Bias Factor (%)'
            }
        }
        
        # Get configuration for the specified sensitivity type
        if sensitivity_type not in config:
            raise ValueError(f"Unknown sensitivity_type: {sensitivity_type}")
        
        cfg = config[sensitivity_type]
        results_df = cfg['df']
        
        
        # Prepare data
        results = results_df.copy()
        results = results[results['A_L'].isin([0.1, 0.5, 0.9])]  # Add this line
        results = results[results['A_G'].isin([0.1, 0.5, 0.9])]  # Add this line
        results['ContractAmount'] = results['ContractAmount'].round(2)
        
        results

        is_pap = self.cm_data.contract_type == "PAP"
        has_gamma = 'Gamma' in results.columns
        
        # Metrics to plot
        metrics = ['StrikePrice', 'ContractAmount']
        if is_pap and has_gamma:
            units = ['€/MWh', 'MW']
        else:
            units = ['€/MWh', 'MWh']
            #results['ContractAmount/year'] = results['ContractAmount'].round(2)  # Convert to MWh
        titles = ['Strike Price', 'Contract Amount']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.flatten()
        if sensitivity_type == "bias":
            fig.suptitle(f'{self.cm_data.contract_type}: {cfg["title"]}, $A_S$={self.cm_data.A_G},$A_B$={self.cm_data.A_L}', fontsize=self.suptitlesize)
        else:
            fig.suptitle(f'{self.cm_data.contract_type}: {cfg["title"]}', fontsize=self.suptitlesize)

        for i, (metric, unit, title) in enumerate(zip(metrics, units, titles)):
            ax = axes[i]
            try:
                pivot_table = results.pivot(
                    index=cfg['index_col'],
                    columns=cfg['columns_col'],
                    values=metric
                )
                pivot_table = pivot_table.sort_index(ascending=False)
                
                sns.heatmap(
                    pivot_table,
                    ax=ax,
                    annot=True,
                    cmap="RdYlGn",
                    cbar=False,
                    linewidths=0.5,
                    fmt ='.2f',
                    linecolor='gray',
                    annot_kws={"size": 16}
                )
                """ 
                # Add custom annotations with units
                for i_idx, row_idx in enumerate(pivot_table.index):
                    for j_idx, col_idx in enumerate(pivot_table.columns):
                        val = pivot_table.iloc[i_idx, j_idx]
                        if not np.isnan(val):
                            # Get background color for text color determination
                            bg_color = plt.cm.get_cmap("cividis")(
                                plt.Normalize()(pivot_table.values) )[i_idx, j_idx, :3]
                            
                            # Calculate luminance
                            luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                            text_color = 'white' if luminance < 0.5 else 'black'
                            
                            # Add gamma annotation if applicable
                            if is_pap and has_gamma and metric == 'ContractAmount':
                                gamma_pivot = results.pivot(
                                    index=cfg['index_col'],
                                    columns=cfg['columns_col'],
                                    values='Gamma'
                                ).sort_index(ascending=False)
                                
                                gamma_val = gamma_pivot.iloc[i_idx, j_idx]
                                ax.text(j_idx + 0.5, i_idx + 0.5, 
                                    f"γ={gamma_val*100:.2f} %",
                                    ha='center', va='center', color=text_color, fontsize=7)
                                    
                                ax.text(j_idx + 0.5, i_idx + 0.63, 
                                    f"{self.cm_data.generator_contract_capacity*gamma_val:.2f} MW",
                                    ha='center', va='center', color=text_color, fontsize=7)
                        
                            else:

                                # Format the main value with units
                                if metric == 'ContractAmount':
                                    text = f"{val:.2f} {unit}"
                                else:
                                    text = f"{val:.2f}"
                                
                                # Add the main value annotation
                                ax.text(j_idx + 0.5, i_idx + 0.5, text,
                                    ha='center', va='center', color=text_color, fontsize=7)
                                
                                if metric == 'ContractAmount':                        
                                # Add the yearly contract value annotation just below the main value
                                    yearly_val = results.pivot(
                                        index=cfg['index_col'],
                                        columns=cfg['columns_col'],
                                        values='ContractAmount/year'
                                    ).sort_index(ascending=False).iloc[i_idx, j_idx]
                                    ax.text(j_idx + 0.5, i_idx + 0.63, 
                                        f"{yearly_val:.2f} GWh/y",
                                        ha='center', va='center', color=text_color, fontsize=7)
                                
                            
                           
                """
                ax.set_title(f"{title} ({unit})", fontsize=self.titlesize)
                ax.set_xlabel(cfg['xlabel'], fontsize=self.labelsize)
                ax.set_ylabel(cfg['ylabel'], fontsize=self.labelsize)

            except Exception as e:
                print(f"Could not plot heatmap for {metric}: {e}")
                ax.set_title(f'{metric} (Plotting Error)')
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()
    
    def _plot_sensitivity_results_line(self, sensitivity_type, filename=None):
        """
        Generalized function to plot sensitivity analysis results for 1D parameter sweeps.
        For negotiation: uses tau_L as parameter.
        For alpha: uses 'alpha' as parameter (single value, not Alpha_L/Alpha_G).
        """
        # Configuration dictionary for different sensitivity types
        config = {
            'negotiation': {
                'df': self.negotiation_sensitivity_df,
                'param_col': 'tau_L',
                'xlabel': 'Load Negotiation Power $\\tau_L$',
                'title': 'Negotiation Power Sensitivity on Strike Price and Contract Amount'
            }
        }
        
        # Get configuration for the specified sensitivity type
        cfg = config[sensitivity_type]
        results_df = cfg['df'].copy()
        
        # Calculate reference values
        
        # Determine contract type and setup units/labels
        is_pap = self.cm_data.contract_type == "PAP"
        has_gamma = 'Gamma' in results_df.columns
        expected_production = self.cm_data.expected_production /8760*1000  # IN MW

        
        if is_pap and has_gamma:
            capture_price_pap = self.cm_data.Capture_price_G_avg*1e3  # Convert to EUR/MWh
            # PAP contract with Gamma
            units = ['€/MWh', 'MW']
            production_type = "MWh"
            
            # Convert contract amount to MW for PAP
            results_df['ContractAmount'] = results_df['ContractAmount'] 
        else:
            avg_price = self.cm_data.expected_price *1e3  # Convert to EUR/MWh
            # Non-PAP contract
            units = ['€/MWh', 'MWh']
            capture_price_type = "\mathbb{E}(\lambda) (EUR/MWh)"
            production_type = "MWh"
            
            # Keep original contract amount and create yearly version
            #results_df['ContractAmount_yearly'] = results_df['ContractAmount'].round(2)
            results_df['ContractAmount'] = results_df['ContractAmount']  
        
        # Sort results by parameter
        results_sorted = results_df.sort_values(cfg['param_col'])
        param_values = results_sorted[cfg['param_col']].values
        
        # Setup subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metrics = ['StrikePrice', 'ContractAmount']
        titles = ['Strike Price', 'Contract Amount']
        
        for i, (metric, unit, title) in enumerate(zip(metrics, units, titles)):
            ax = axes[i]
            
            # Plot reference lines
            if metric == 'StrikePrice' and is_pap:
                ax.axhline(capture_price_pap, color='black', linestyle='--', 
                        label=f'Capture Price (G) ')
            elif metric == 'StrikePrice':
                ax.axhline(avg_price, color='black', linestyle='--', 
                        label=f'Expected Price')
            elif metric == 'ContractAmount' and not (is_pap):
                # Plot expected production line for non-PAP contracts
                ax.axhline(expected_production, color='black', linestyle='--', 
                        label=f'Expected Production {production_type}')
                    
            # Plot main metric
            ax.plot(param_values, np.round(results_sorted[metric].values,4),marker='o', linewidth=2, markersize=8, label=f"({title})")
            """ 
            sns.lineplot(
                x=param_values,
                y=results_sorted[metric].values,
                marker='o',
                linewidth=2,
                markersize=8,
                ax=ax,
                label=f"Contract ({title})"
            )
            """
            # Handle secondary y-axis for Contract Amount subplot
            if metric == 'ContractAmount':                
                if is_pap and has_gamma:
                    # Plot Gamma percentage for PAP contracts
                    ax2 = ax.twinx()

                    gamma_values = results_sorted['Gamma'].values * 100

                    ax2.plot(param_values, gamma_values, linestyle='--',color="red", linewidth=1, label='Gamma (%)')
                    """ 
                    sns.lineplot(
                        x=param_values,
                        y=gamma_values,
                        marker='s',
                        linewidth=1,
                        markersize=6,
                        color='red',
                        alpha=0.7,
                        linestyle='--',
                        ax=ax2,
                        label='Gamma (%)'
                    )
                    """
                    ax2.set_ylabel('Gamma (%)', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    
                    # Special handling if Gamma values are all close to 100%
                    if np.allclose(gamma_values, 100, atol=1e-2):
                        ax2.set_ylim(99, 101)
                        #Set y-limits with some padding
                    #y_values = results_sorted[metric].values
                    #y_padding = 0.1 * (y_values.max() - y_values.min())
                    #ax.set_ylim(y_values.min() - y_padding, y_values.max() + y_padding)          
                # Create combined legend for Contract Amount subplot only
                lines1, labels1 = ax.get_legend_handles_labels()
                if 'is_pap' and has_gamma:
                    # Include Gamma line in the legend
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=self.legendsize)
                else:
                    ax.legend(lines1, labels1, loc="best", fontsize=self.legendsize)

            else:
                # Simple legend for Strike Price subplot
                
                ax.legend(loc="best", fontsize=self.legendsize)

            # Configure axes
            ax.set_xlabel(cfg['xlabel'], fontsize=self.labelsize)
            ax.set_ylabel(f'{title} ({unit})', fontsize=self.labelsize)
            ax.set_title(title, fontsize=self.titlesize)
            ax.grid(True, alpha=0.3)
            #
            
         
        
        # Add main title and layout
        fig.suptitle(f'{self.cm_data.contract_type}: {cfg["title"]}', fontsize=self.suptitlesize)
        plt.tight_layout()
        
        # Save or show plot
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

    def _plot_earnings_histograms(self, fixed_A_G, A_L_to_plot, filename=None):
        """
        Plots histograms of G and L net earnings for different risk aversion levels.
        """
       
        earnings_df =  self.earnings_risk_sensitivity_df
        filtered_results = earnings_df[
        (earnings_df['A_G'] == fixed_A_G) & 
        (earnings_df['A_L'].isin(A_L_to_plot)) & 
        (~earnings_df['Revenue_G'].isna()) & 
        (~earnings_df['Revenue_L'].isna())
    ]

        if filtered_results.empty:
            print("No valid results to plot.")
            return
        
        all_G_values = np.concatenate([filtered_results['Revenue_G'].values,self.cm_data.net_earnings_no_contract_true_G.values])
        all_L_values = np.concatenate([filtered_results['Revenue_L'].values,self.cm_data.net_earnings_no_contract_true_L.values])
        

        # Create uniform bins based on global min and max
        bins = 20
        min_val_G = min(all_G_values) 
        max_val_G = max(all_G_values) 
        bin_edges_G = np.linspace(min_val_G, max_val_G, bins + 1)

        min_val_L = min(all_L_values) 
        max_val_L = max(all_L_values) 
        bin_edges_L = np.linspace(min_val_L, max_val_L, bins + 1)


        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot G Revenue Histogram
        ax_G = axes[0]
        ax_L = axes[1]

        # Get color cycle before the loops
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        # Plot G Revenue Histogram
        for idx, a_L in enumerate(A_L_to_plot):
            G_values = filtered_results[filtered_results['A_L'] == a_L]['Revenue_G'].values

            if G_values.size == 0:
                print(f"No G values for A_L={a_L}, skipping...")
                continue

            # Expected G Revenue
            G_6_expected = G_values.mean()
            # Calculate CVaR values 
            cvar_G = calculate_cvar_left(G_values,self.cm_data.PROB, self.cm_data.alpha)
            utility_G = (1-fixed_A_G)*G_6_expected + fixed_A_G*cvar_G

            if len(G_values) > 0:
                current_color = colors[idx % len(colors)]  # Cycle through colors
                print(f"\nPlotting histogram for A_L={a_L}")
                print(f"Values range: {G_values.min() } to {G_values.max() }")
                ax_G.hist(
                    G_values ,
                    bins=bin_edges_G,
                    alpha=0.6,
                    label=f'A_L={a_L}',
                    color=current_color,
                    density=False
                )
                ax_G.axvline(G_6_expected, linestyle="--", color=current_color, 
                              label=f"A_L={a_L} - Expected: {G_6_expected:.2f}")

            # Plot L Revenue Histogram with same color
            L_values = filtered_results[filtered_results['A_L'] == a_L]['Revenue_L'].values

            # Expected L Revenue
            L_expected = L_values.mean()
            # Calculate CVaR values 

            cvar_L = calculate_cvar_left(L_values,self.cm_data.PROB, self.cm_data.alpha)
            utility_L = (1-a_L)*L_expected + a_L*cvar_L
            if len(L_values) > 0:
                ax_L.hist(
                    L_values ,
                    bins=bin_edges_L,
                    alpha=0.6,
                    label=f'A_L={a_L}',
                    color=current_color,
                    density=False
                )
                ax_L.axvline(L_expected, linestyle="--", color=current_color, 
                              label=f"A_L={a_L} - Expected   : {L_expected:.2f}")
        
        ax_G.hist(self.cm_data.net_earnings_no_contract_true_G ,bins=bin_edges_G,alpha=0.4,label=f'No Contract',density=False,color ='black')    
        ax_G.axvline(self.cm_data.net_earnings_no_contract_true_G.mean() , linestyle="--",color ='black', label=f"No Contract - Expected : {self.cm_data.net_earnings_no_contract_true_G.mean() :.2f}")
        ax_G.set_title(f'Generator (G) Revenue Distribution', fontsize=self.titlesize)
        ax_G.set_xlabel('Generator Revenue (Mio EUR)', fontsize=self.labelsize)
        ax_G.set_ylabel('Frequency', fontsize=self.labelsize)
        # Modify legend to have two columns with specific ordering
        handles, labels = ax_G.get_legend_handles_labels()
        hist_handles = handles[::2]  # Get histogram handles
        line_handles = handles[1::2]  # Get vertical line handles
        hist_labels = labels[::2]    # Get histogram labels
        line_labels = labels[1::2]   # Get vertical line labels
        ax_G.legend(hist_handles + line_handles, hist_labels + line_labels, 
                    ncol=2, loc='upper right', 
                    fontsize=10, bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax_G.transAxes,
                    framealpha=0.8)
        ax_G.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax_L.hist(self.cm_data.net_earnings_no_contract_true_L,bins=bin_edges_L,alpha=0.4,label=f'No Contract',density=False, color ='black')
        ax_L.axvline(self.cm_data.net_earnings_no_contract_true_L.mean() , linestyle="--",color ='black', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_true_L.mean() :.2f}")    
        ax_L.set_title(f'Load (L) Revenue Distribution', fontsize=self.titlesize)
        ax_L.set_xlabel('Load Revenue (Mio EUR)', fontsize=self.labelsize)
        ax_L.set_ylabel('Frequency', fontsize=self.labelsize)
        # Apply same legend formatting to L plot
        handles, labels = ax_L.get_legend_handles_labels()
        hist_handles = handles[::2]
        line_handles = handles[1::2]
        hist_labels = labels[::2]
        line_labels = labels[1::2]
        ax_L.legend(hist_handles + line_handles, hist_labels + line_labels, 
                    ncol=2, loc='upper right',
                    fontsize=self.legendsize, bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax_L.transAxes,
                    framealpha=0.8)
        ax_L.grid(True, axis='y', linestyle='--', alpha=0.7)

        #Add suptitle 
        fig.suptitle(f'{self.cm_data.contract_type}: Expected Revenue ($A_G$ = {fixed_A_G})', fontsize=self.suptitlesize)

        plt.tight_layout()
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

    def _plot_earnings_histograms_alpha(self, filename=None):
            """
            Simplified: Plot earnings histograms for all unique alpha values in self.alpha_earnings_df.
            """
            df = self.alpha_earnings_df
            if df is None or df.empty:
                print("No alpha earnings results to plot.")
                return
        
            # Only keep rows with valid earnings
            df = df.dropna(subset=['Revenue_G', 'Revenue_L', 'alpha'])
        
            unique_alphas = sorted(df['alpha'].unique())
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
            fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(12, 6))
        
            # Collect all values for binning
            all_g = np.concatenate([df['Revenue_G'].values, self.cm_data.net_earnings_no_contract_true_G])
            all_l = np.concatenate([df['Revenue_L'].values, self.cm_data.net_earnings_no_contract_true_L])
            bins_g = np.linspace(all_g.min(), all_g.max(), 20)
            bins_l = np.linspace(all_l.min(), all_l.max(), 20)
        
            for idx, alpha in enumerate(unique_alphas):
                color = colors[idx % len(colors)]
                g_vals = df[df['alpha'] == alpha]['Revenue_G'].values
                l_vals = df[df['alpha'] == alpha]['Revenue_L'].values
        
                if len(g_vals) > 0:
                    ax_g.hist(g_vals, bins=bins_g, alpha=0.6, label=f'α={alpha}', color=color)
                    ax_g.axvline(g_vals.mean(), color=color, linestyle='--')
                if len(l_vals) > 0:
                    ax_l.hist(l_vals, bins=bins_l, alpha=0.6, label=f'α={alpha}', color=color)
                    ax_l.axvline(l_vals.mean(), color=color, linestyle='--')
        
            # No contract reference
            ax_g.hist(self.cm_data.net_earnings_no_contract_true_G, bins=bins_g, alpha=0.3, color='black', label='No Contract')
            ax_g.axvline(self.cm_data.net_earnings_no_contract_true_G.mean(), color='black', linestyle='--')
            ax_l.hist(self.cm_data.net_earnings_no_contract_true_L, bins=bins_l, alpha=0.3, color='black', label='No Contract')
            ax_l.axvline(self.cm_data.net_earnings_no_contract_true_L.mean(), color='black', linestyle='--')

            ax_g.set_title('Generator Revenue Distribution', fontsize=self.titlesize)
            ax_g.set_xlabel('Revenue (Mio EUR)', fontsize=self.labelsize)
            ax_g.set_ylabel('Frequency', fontsize=self.labelsize)
            ax_g.legend()
            ax_g.grid(True, axis='y', linestyle='--', alpha=0.7)

            ax_l.set_title('Load Revenue Distribution', fontsize=self.titlesize)
            ax_l.set_xlabel('Revenue (Mio EUR)', fontsize=self.labelsize)
            ax_l.set_ylabel('Frequency', fontsize=self.labelsize)
            ax_l.legend()
            ax_l.grid(True, axis='y', linestyle='--', alpha=0.7)

            fig.suptitle(f"{self.cm_data.contract_type}: Earnings Distribution by Alpha, $A_G$ = {self.cm_data.A_G}, $A_L$ = {self.cm_data.A_L}", fontsize=self.suptitlesize)
            plt.tight_layout()
        
            if filename:
                plt.savefig(os.path.join(self.plots_dir, filename), bbox_inches='tight', dpi=300)
                print(f"Plot saved to {filename}")
                plt.close(fig)
            else:
                plt.show()

    def _plot_expected_versus_threatpoint(self,fixed_A_G, A_L_to_plot, filename=None):

        """
        Plots histograms of G and L net earnings for different risk aversion levels.
        """
        earnings_risk_sensitivity_df = self.earnings_risk_sensitivity_df
        
        filtered_results = pd.concat([
            df[(df['A_G'] == fixed_A_G) & (df['A_L'].isin(A_L_to_plot)) & 
               (~df['Revenue_G'].isna()) & (~df['Revenue_L'].isna())]
            for df in earnings_risk_sensitivity_df
            if isinstance(df, pd.DataFrame) and not df.empty
        ], ignore_index=True)

        if filtered_results.empty:
            print("No valid results to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot G Revenue Histogram
        ax_G = axes[0]
        ax_L = axes[1]

        # Get color cycle before the loops
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        zeta_L_values = []


        # Calculate CVaR values (constant for all risk aversion values)
        cvar_G_no_contract = calculate_cvar_left(self.cm_data.net_earnings_no_contract_G,self.cm_data.PROB, self.cm_data.alpha)
        cvar_L_no_contract = calculate_cvar_left(self.cm_data.net_earnings_no_contract_L,self.cm_data.PROB, self.cm_data.alpha)
        mean_G_contract = self.cm_data.net_earnings_no_contract_G.mean()
        mean_L_no_contract = self.cm_data.net_earnings_no_contract_L.mean()

        # Calculate threat points for different risk aversion values
        for A in A_L_to_plot:
            zeta_L = (1-A)*mean_L_no_contract + A*cvar_L_no_contract
            zeta_L_values.append(zeta_L)
        zeta_G = ((1-fixed_A_G)*mean_G_contract + fixed_A_G*cvar_G_no_contract)
        # Plot G Revenue Histogram
        for idx, a_L in enumerate(A_L_to_plot):
            G_values = filtered_results[filtered_results['A_L'] == a_L]['Revenue_G'].values
            current_color = colors[idx % len(colors)]  # Cycle through colors

            if len(G_values) > 0:
                expected_G = G_values.mean()
                cvar_G = calculate_cvar_left(G_values,self.cm_data.PROB, self.cm_data.alpha)
                utility_G = (1-fixed_A_G)*expected_G + fixed_A_G*cvar_G
                ax_G.axvline(utility_G/1e5, linestyle="-", color=current_color, 
                              label=f"A_L={a_L} - Utility: {utility_G/1e5:.2f}")
            # Plot L Revenue Histogram with same color
            L_values = filtered_results[filtered_results['A_L'] == a_L]['Revenue_L'].values
            if len(L_values) > 0:
                expected_L = L_values.mean()
                cvar_L = calculate_cvar_left(L_values,self.cm_data.PROB, self.cm_data.alpha)
                utility_L = (1-a_L)*expected_L + a_L*cvar_L
                ax_L.axvline(utility_L/1e5, linestyle="-", color=current_color, 
                              label=f"A_L={a_L} - Utility: {utility_L/1e5:.2f}")
                ax_L.axvline(zeta_L_values[idx]/1e5, linestyle="--", color=current_color, label=f"A_L={a_L:.2f} - Threat= {zeta_L_values[idx]/1e5:.2f} ")

        
        #G Subplot configuration
        ax_G.axvline(zeta_G/1e5, linestyle="--", color='black', label=f"Threat Point: {zeta_G/1e5:.2f}")
        ax_G.set_title(f'Generator (G) Threatpoint\n(A_G = {fixed_A_G}) vs. L Risk Aversion', fontsize=self.titlesize)
        ax_G.set_xlabel('Generator Revenue ($ x 10^5)', fontsize=self.labelsize)
        # Modify legend to have two columns with specific ordering
        handles, labels = ax_G.get_legend_handles_labels()
        hist_handles = handles[::2]  # Get histogram handles
        line_handles = handles[1::2]  # Get vertical line handles
        hist_labels = labels[::2]    # Get histogram labels
        line_labels = labels[1::2]   # Get vertical line labels
        ax_G.legend(hist_handles + line_handles, hist_labels + line_labels, 
                    ncol=2, loc='upper right', 
                    fontsize=10, bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax_G.transAxes,
                    framealpha=0.8)
        ax_G.grid(True, axis='y', linestyle='--', alpha=0.7)

        #ax_L.axvline(self.cm_data.net_earnings_no_contract_L_df.sum().mean() / 1e5, linestyle="--",color ='black', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_L_df.sum().mean() / 1e5:.2f}")   
        #L Subplot configuration
        ax_L.set_title(f'Load (L) Threatpoints vs \n(A_G = {fixed_A_G})', fontsize=self.titlesize)
        ax_L.set_xlabel('Load Revenue ($ x 10^5)', fontsize=self.labelsize)
        # Apply same legend formatting to L plot
        handles, labels = ax_L.get_legend_handles_labels()
        hist_handles = handles[::2]
        line_handles = handles[1::2]
        hist_labels = labels[::2]
        line_labels = labels[1::2]
        ax_L.legend(hist_handles + line_handles, hist_labels + line_labels, 
                    ncol=2, loc='upper right',
                    fontsize=10, bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax_L.transAxes,
                    framealpha=0.8)
        ax_L.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()
        print(f"Plot saved t bla bla")     

    def _plot_no_contract(self, filename=None):
        """Plots histograms of no-contract revenues and threat point evolution."""
        # Create first figure for histograms
        fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot histograms of no-contract revenues
        ax_G = axes[0]
        ax_L = axes[1]

        # Scale values to 10^5 for better readability
        G_values = self.cm_data.net_earnings_no_contract_G_df.sum().values / 1e5
        L_values = self.cm_data.net_earnings_no_contract_L.values / 1e5

        # Create histogram bins
        bins_G = np.linspace(min(G_values), max(G_values), 19)
        bins_L = np.linspace(min(L_values), max(L_values), 19)

        # Plot histograms
   


        # Create second figure for threat point evolution

        # Calculate threat points for different risk aversion values
        risk_aversion_values = np.linspace(0, 1, 4)
        zeta_G_values = []
        zeta_L_values = []
        
        # Calculate CVaR values (constant for all risk aversion values)
        cvar_G = calculate_cvar_left(self.cm_data.net_earnings_no_contract_G,self.cm_data.PROB, self.cm_data.alpha)
        cvar_L = calculate_cvar_left(self.cm_data.net_earnings_no_contract_L,self.cm_data.PROB, self.cm_data.alpha)
        mean_G = self.cm_data.net_earnings_no_contract_G.mean()
        mean_L = self.cm_data.net_earnings_no_contract_L.mean()

        # Calculate threat points for different risk aversion values
        for A in risk_aversion_values:
            zeta_G = (1-A)*mean_G + A*cvar_G
            zeta_L = (1-A)*mean_L + A*cvar_L
            zeta_G_values.append(zeta_G/1e5)
            zeta_L_values.append(zeta_L/1e5)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        ax_G.hist(G_values, bins=bins_G, alpha=0.6, color='blue', density=False)
        for i,zeta in enumerate(zeta_G_values):
            current_color = colors[i % len(colors)]  # Cycle through colors
            ax_G.axvline(zeta, linestyle="--", color=current_color, label=f'A_G={risk_aversion_values[i]:.2f} - Threat Point: {zeta:.2f}')
        ax_G.set_title('Generator (G) No-Contract Revenue Distribution', fontsize=self.titlesize)
        ax_G.set_xlabel('Generator Revenue ($ x 10^5)', fontsize=self.labelsize)
        ax_G.set_ylabel('Frequency', fontsize=self.labelsize)
        ax_G.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax_G.legend()

        ax_L.hist(L_values, bins=bins_L, alpha=0.6, color='green', density=False)
        for i,zeta in enumerate(zeta_L_values):
            current_color = colors[i % len(colors)]
            ax_L.axvline(zeta, linestyle="--", color=current_color, label=f'A_L={risk_aversion_values[i]:.2f} - Threat Point: {zeta:.2f}')
        ax_L.set_title('Load (L) No-Contract Revenue Distribution', fontsize=self.titlesize)
        ax_L.set_xlabel('Load Revenue ($ x 10^5)', fontsize=self.labelsize)
        ax_L.set_ylabel('Frequency', fontsize=self.labelsize)
        ax_L.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax_L.legend()

        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            print(f"Plot saved to {filepath}")
            fig1.savefig(filepath.replace('.', '_hist.'))
            print(f"Plots saved to {filepath}")
            plt.close(fig1)
        else:
            plt.show()

    def _plot_sensitivity(self, x_column, sensitivity_name="Sensitivity", filename=None):
        """
        Generalized sensitivity plot for contract parameters.
        df: DataFrame with sensitivity results
        x_column: str, column name for x-axis (e.g. 'Production_Change', 'CaptureRate_Change', etc.)
        sensitivity_name: str, for plot titles (e.g. 'Production', 'Capture Rate', 'Price', etc.)
        filename: optional, for saving the plot
        """

        if sensitivity_name == "Production":
            df = self.production_sensitivity_df
        
        elif sensitivity_name == "Capture Rate":
            df = self.gen_CR_sensitivity_results

        elif sensitivity_name == "Load Capture Rate":
            df = self.load_CR_sensitivity_df

        elif sensitivity_name == "Load":
            df = self.load_sensitivity_df
        
        elif sensitivity_name == "Price":
            df = self.price_sensitivity_df

        else:
            return print("No Sensitivity Dataframe Found")
        # Create figure with subplots - use a 2x3 grid for detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(
            f'{self.cm_data.contract_type}: Contract Parameters Sensitivity to {sensitivity_name} $A_G$ = {self.cm_data.A_G}, $A_L$ = {self.cm_data.A_L}',
            fontsize=self.suptitlesize
        )
        axes = axes.flatten()

        # Plot Strike Price
        axes[0].plot(df[x_column], df['StrikePrice'], marker='o', linestyle='-')
        axes[0].set_xlabel(f'{x_column} (%)', fontsize=self.labelsize)
        axes[0].set_ylabel('Strike Price (EUR/MWh)', fontsize=self.labelsize)
        axes[0].set_title(f'Strike Price vs {sensitivity_name}', fontsize=self.titlesize)
        axes[0].grid(True)

        # Plot Contract Amount
        if self.cm_data.contract_type == "PAP":
            axes[1].set_ylim(25, 40)
        axes[1].plot(df[x_column], df['ContractAmount'], marker='o', linestyle='-')
        axes[1].set_xlabel(f'{x_column} (%)', fontsize=self.labelsize)
        axes[1].set_ylabel('Contract Amount (MWh)', fontsize=self.labelsize)
        axes[1].set_title(f'Contract Amount vs {sensitivity_name}', fontsize=self.titlesize)
        axes[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        axes[1].grid(True)

        # Plot Utility G
        axes[2].plot(df[x_column], df['Utility_G'], marker='o', linestyle='-', label='Utility Generator')
        axes[2].plot(df[x_column], df['ThreatPoint_G'], marker='o', linestyle='-', label='Threat Point Generator')
        axes[2].set_xlabel(f'{x_column} (%)', fontsize=self.labelsize)
        axes[2].set_ylabel('Utility Generator', fontsize=self.labelsize)
        axes[2].set_title(f'Generator Utility vs {sensitivity_name}', fontsize=self.titlesize)
        axes[2].grid(True)
        axes[2].legend()

        # Plot Utility L
        axes[3].plot(df[x_column], df['Utility_L'], marker='o', linestyle='-', label='Utility Load')
        axes[3].plot(df[x_column], df['ThreatPoint_L'], marker='o', linestyle='-', label='Threat Point Load')
        axes[3].set_xlabel(f'{x_column} (%)', fontsize=self.labelsize)
        axes[3].set_ylabel('Utility Load', fontsize=self.labelsize)
        axes[3].set_title(f'Load Utility & Threatpoint vs {sensitivity_name}', fontsize=self.titlesize)
        axes[3].grid(True)
        axes[3].legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if filename:
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved {sensitivity_name.lower()} sensitivity plot to {filename}")
        else:
            plt.show()

    def _plot_no_contract_boundaries(self, sensitivity_type, filename=None):
        """
        Plot the no-contract boundaries for different risk aversion scenarios.
        
        Parameters:
        -----------
        type:  str
            Type of boundary to plot. price or production
        filename : str, optional
            Path to save the plot. If None, the plot will be displayed.
        """
        plt.figure(figsize=(12, 8))
        xlim = (-31, 31)
        ylim = (-31, 31)

        if sensitivity_type == "price":
            boundary_results = self.boundary_results_price
        elif sensitivity_type == "production":
            boundary_results = self.boundary_results_production
        else:
            boundary_results = []

        # Prepare axis
        ax = plt.gca()

        # Plot each scenario's feasible shading and boundary contour; else fallback
        for result in boundary_results:
            scenario = result['scenario']

            if 'feas_mask' in result and 'KL_grid' in result and 'KG_grid' in result:
                KL_grid = np.array(result['KL_grid'])
                KG_grid = np.array(result['KG_grid'])
                feas_mask = np.array(result['feas_mask'])
                try:
                    # Shading: fill feasible region for this scenario (no legend entry)
                    ax.contourf(KL_grid * 100, KG_grid * 100, feas_mask,
                                levels=[0.5, 1.1], colors=[scenario['color']],
                                alpha=0.15, antialiased=True, zorder=1)
                    # Boundary curve on top
                    ax.contour(KL_grid * 100, KG_grid * 100, feas_mask,
                               levels=[0.5], colors=[scenario['color']],
                               linestyles=[scenario['linestyle']], linewidths=[scenario['linewidth']], zorder=3)
                    # Legend proxy for the line
                    ax.plot([], [], color=scenario['color'], linestyle=scenario['linestyle'],
                            linewidth=scenario['linewidth'], label=scenario['label'])
                except Exception as e:
                    print(f"Contour plotting failed for {scenario['label']}: {e}. Falling back to points/regression.")
                    # Fallback to regression through boundary points if present
                    boundary_points = np.array(result.get('boundary_points', []))
                    if len(boundary_points) >= 2:
                        lowest_boundary = self._extract_lowest_boundary(boundary_points)
                        if lowest_boundary is not None and len(lowest_boundary) >= 2:
                            n_space = np.linspace(xlim[0]*1e-2, xlim[1]*1e-2, 100)
                            X = lowest_boundary[:, 0].reshape(-1, 1)
                            y = lowest_boundary[:, 1]
                            model = LinearRegression().fit(X, y)
                            X_pred = n_space.reshape(-1, 1)
                            boundary = model.predict(X_pred)
                            sns.lineplot(x=n_space*100, y=boundary*100, label=scenario['label'],
                                         linestyle=scenario['linestyle'], linewidth=scenario['linewidth'], color=scenario['color'], zorder=3)
                            sns.scatterplot(x=lowest_boundary[:, 0]*100, y=lowest_boundary[:, 1]*100, s=90, alpha=0.5,
                                            color=scenario['color'], edgecolor='black')
            else:
                # Legacy behavior: use boundary_points with regression
                boundary_points = np.array(result.get('boundary_points', []))
                if len(boundary_points) < 2:
                    print(f"Skipping scenario {scenario['label']} due to insufficient boundary points.")
                    continue
                lowest_boundary = self._extract_lowest_boundary(boundary_points)
                if lowest_boundary is None or len(lowest_boundary) < 2:
                    print(f"Skipping scenario {scenario['label']} due to insufficient boundary points after filtering.")
                    continue
                n_space = np.linspace(xlim[0]*1e-2, xlim[1]*1e-2, 100)
                X = lowest_boundary[:, 0].reshape(-1, 1)
                y = lowest_boundary[:, 1]
                model = LinearRegression().fit(X, y)
                X_pred = n_space.reshape(-1, 1)
                boundary = model.predict(X_pred)
                sns.lineplot(x=n_space*100, y=boundary*100, label=scenario['label'],
                             linestyle=scenario['linestyle'], linewidth=scenario['linewidth'], color=scenario['color'])
                sns.scatterplot(x=lowest_boundary[:, 0]*100, y=lowest_boundary[:, 1]*100, s=90, alpha=0.5,
                                color=scenario['color'], edgecolor='black')

        # Add labels and formatting
        if sensitivity_type == "price":
            plt.xlabel(r'Load price bias $K^L$ (% of $\mathbb{E}[\mathrm{price}]$)', fontsize=self.labelsize)
            plt.ylabel(r'Generator price bias $K^G$ (% of $\mathbb{E}[\mathrm{price}]$)', fontsize=self.labelsize)
        elif sensitivity_type == "production":
            plt.xlabel(r'Load production bias $K^L$ (% of $\mathbb{E}[\mathcal{P}^G]$)', fontsize=self.labelsize)
            plt.ylabel(r'Generator production bias $K^G$ (% of $\mathbb{E}[\mathcal{P}^G]$)', fontsize=self.labelsize)
        plt.title(f'{self.cm_data.contract_type}-{sensitivity_type}: Contract Boundaries for Different Risk Aversion Levels', fontsize=self.titlesize)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.axhline(y=0, color='k', linewidth=2)
        plt.axvline(x=0, color='k', linewidth=2)

        # Set the x and y axis limits similar to the figure
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        else:
            plt.show()

    def _plot_no_contract_boundaries_all(self,sensitivity_type, filename=None):
            # Helper method for the comprehensive visualization

        if sensitivity_type == "price":
            self.boundary_results = self.boundary_results_price
        elif sensitivity_type == "production":
            self.boundary_results = self.boundary_results_production


        def _plot_boundary_on_axis(ax, result):
            """Plot a single boundary on a given axis (use contour if feas_mask available)."""
            scenario = result['scenario']
            if 'feas_mask' in result and 'KL_grid' in result and 'KG_grid' in result:
                KL_grid = np.array(result['KL_grid'])
                KG_grid = np.array(result['KG_grid'])
                feas_mask = np.array(result['feas_mask'])
                # Shading behind the curve: fill feasible region
                ax.contourf(KL_grid*100, KG_grid*100, feas_mask,
                            levels=[0.5, 1.1], colors=[scenario['color']], alpha=0.15, antialiased=True)
                # Boundary curve
                cs = ax.contour(KL_grid*100, KG_grid*100, feas_mask,
                                levels=[0.5], colors=[scenario['color']],
                                linestyles=[scenario['linestyle']], linewidths=[scenario['linewidth']])
                # Add a proxy handle for legend instead of accessing cs.collections
                ax.plot([], [], color=scenario['color'], linestyle=scenario['linestyle'],
                        linewidth=scenario['linewidth'], label=f"A_G={scenario['A_G']}, A_L={scenario['A_L']}")
            else:
                lowest_boundary = self._extract_lowest_boundary(result.get('boundary_points', []))
                if lowest_boundary is None or len(lowest_boundary) < 2:
                    print(f"Skipping scenario {scenario['label']} due to insufficient boundary points after filtering.")
                    return
                X = lowest_boundary[:, 0].reshape(-1, 1)
                y = lowest_boundary[:, 1]
                model = LinearRegression().fit(X, y)
                xlim = [-31, 31]
                n_space = np.linspace(xlim[0]/100, xlim[1]/100, 100)
                X_pred = n_space.reshape(-1, 1)
                boundary = model.predict(X_pred)
                sns.lineplot(x=n_space*100, y=boundary*100,
                             label=f"A_G={scenario['A_G']}, A_L={scenario['A_L']}",
                             linestyle=scenario['linestyle'], linewidth=scenario['linewidth'],
                             color=scenario['color'], ax=ax)
                
        
        
        """
        Create a visualization specifically showing how asymmetry in risk aversion
        affects the no-contract boundaries.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Filter scenarios by type
        sym_scenarios = [r for r in self.boundary_results 
                        if abs(r['scenario']['A_G'] - r['scenario']['A_L']) < 1e-6]
        asym_scenarios = [r for r in self.boundary_results 
                        if abs(r['scenario']['A_G'] - r['scenario']['A_L']) >= 1e-6]
        
        # Plot 1: All symmetrical cases
        for result in sym_scenarios:
            _plot_boundary_on_axis(ax1, result)
        ax1.set_title("Symmetrical Risk Aversion (A_G = A_L)", fontsize=self.titlesize)
        ax1.legend()
        
        # Plot 2: All asymmetrical cases
        for result in asym_scenarios:
            _plot_boundary_on_axis(ax2, result)
        ax2.set_title("Asymmetrical Risk Aversion (A_G ≠ A_L)", fontsize=self.titlesize)
        ax2.legend()
        
        # Plot 3: Fixed A_G, varying A_L
        fixed_ag_scenarios = [r for r in self.boundary_results if r['scenario']['A_G'] == 0.5]
        for result in fixed_ag_scenarios:
            _plot_boundary_on_axis(ax3, result)
        ax3.set_title("Fixed Generator Risk Aversion (A_G = 0.5)", fontsize=self.titlesize)
        ax3.legend()
        
        # Plot 4: Fixed A_L, varying A_G
        fixed_al_scenarios = [r for r in self.boundary_results if r['scenario']['A_L'] == 0.5]
        for result in fixed_al_scenarios:
            _plot_boundary_on_axis(ax4, result)
        ax4.set_title("Fixed Load Risk Aversion (A_L = 0.5)", fontsize=self.titlesize)
        ax4.legend()
        
        # Common formatting
        for ax in [ax1, ax2, ax3, ax4]:
            if sensitivity_type == "price":
                ax.set_xlabel('Load price bias KL (% of E[price])', fontsize=self.labelsize)
                ax.set_ylabel('Generator price bias KG (% of E[price])', fontsize=self.labelsize)
            elif sensitivity_type == "production":
                ax.set_xlabel('Load production bias KL (% of E[production])', fontsize=self.labelsize)
                ax.set_ylabel('Generator production bias KG (% of E[production])', fontsize=self.labelsize)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.axhline(y=0, color='k',linewidth=2)
            ax.axvline(x=0, color='k',linewidth=2)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        fig.suptitle(f'{self.cm_data.contract_type}-{sensitivity_type}: Contract Boundaries', fontsize=self.suptitlesize)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def _extract_lowest_boundary(self,boundary_points):
                """
                Extract the lowest boundary line by selecting the minimum KG value for each KL.
                
                Parameters:
                -----------
                boundary_points : list of tuples
                    List of (KL, KG) points representing the boundary.
                
                Returns:
                --------
                lowest_boundary : ndarray
                    Array of (KL, KG) points representing the lowest boundary line.
                """
                boundary_points = np.array(boundary_points)
                # Round KL values slightly to avoid floating-point precision issues
                if len(boundary_points) < 2:
                    print(f"Skipping scenario due to insufficient boundary points.")
                    return 
                boundary_points[:, 0] = np.round(boundary_points[:, 0], 6)
                
                
                # Create a dictionary to store the minimum KG for each KL
                kl_to_min_kg = {}
                
                for kl, kg in boundary_points:
                    if kl not in kl_to_min_kg or kg < kl_to_min_kg[kl]:
                        kl_to_min_kg[kl] = kg
                
                
                # Convert the dictionary back to an array
                lowest_boundary = np.array([[kl, kg] for kl, kg in kl_to_min_kg.items()])
                # Sort by KL for plotting
                lowest_boundary = lowest_boundary[np.argsort(lowest_boundary[:, 0])]

                # tst 

                    # Group by KG values
                kg_values = lowest_boundary[:, 1]
                unique_kg = np.unique(kg_values)
                
                filtered_rows = []
                
                # For each unique KG value
                for kg in unique_kg:
                    # Get all rows with this KG value
                    mask = kg_values == kg
                    matching_rows = lowest_boundary[mask]
                    
                    if len(matching_rows) > 1:
                        # Multiple rows with the same KG value, keep the one with highest KL
                        best_row_idx = np.argmin(matching_rows[:, 0])
                        filtered_rows.append(matching_rows[best_row_idx])
                    else:
                        # Only one row with this KG value
                        filtered_rows.append(matching_rows[0])
                filtered_boundary = np.array(filtered_rows)
                
                return filtered_boundary

    def _risk_plot_earnings_boxplot(self, fixed_A_G, A_L_to_plot, filename=None):

        
      
        """
        Plots boxplots of earnings for different risk aversion levels.
        
        Parameters:
        -----------
        fixed_A_G : float
            Fixed risk aversion level for the generator.
        A_L_to_plot : list
            List of risk aversion levels for the load to plot.
        filename : str, optional
            Path to save the plot. If None, the plot will be displayed.
        """
        
        earnings_df = self.earnings_risk_sensitivity_df
        filtered_results = earnings_df[
            (earnings_df['A_G'] == fixed_A_G) & 
            (earnings_df['A_L'].isin(A_L_to_plot))  
        ]

       # Prepare the no-contract data
        no_contract_g = self.cm_data.net_earnings_no_contract_true_G
        no_contract_l = self.cm_data.net_earnings_no_contract_true_L
        no_contract_df = pd.DataFrame({
            'Revenue_G': no_contract_g,
            'Revenue_L': no_contract_l,
            'A_L': 'No \n Contract'  # Assign a string label for the category
        })

        #Prepare Capture Price Data
        CP_df = pd.DataFrame({
            'Revenue_G': self.CP_earnings_df["Revenue_G_CP"],
            'Revenue_L': self.CP_earnings_df["Revenue_L_CP"],
            'A_L': 'Capture \n Price'  # Assign a string label for the category
        })


        # Combine the contract and no-contract data
        # Convert A_L to object type to allow mixing numbers and strings
        filtered_results['A_L'] = filtered_results['A_L'].astype(str)
        filtered_results = filtered_results.dropna()
        #plot_data = pd.concat([no_contract_df,CP_df,filtered_results], ignore_index=True)
        plot_data = pd.concat([no_contract_df,filtered_results], ignore_index=True)

            
        # Define the order for the x-axis categories
        A_L_to_plot = filtered_results['A_L'].unique()

        A_L_to_plot = sorted(A_L_to_plot)
        plot_order = A_L_to_plot.insert(0,'No Contract')  # Add 'No Contract' at the beginning
        contract_mask = plot_data['A_L'] != 'No Contract'

        # Create figure with more space at the bottom for the table
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_G = axes[0]
        ax_L = axes[1]
        
        # 1. Add violin plots behind boxplots for Generator Revenue
        sns.violinplot(
            data=plot_data,
            x='A_L',
            y='Revenue_G',
            order=plot_order,
            ax=ax_G,
            hue="A_L", 
            hue_order=plot_order,
            legend=False,
            alpha=0.3,
            width=0.8,
            inner=None,
            palette="Set2"
        )
        
        # 2. Add boxplots on top for Generator Revenue
        box_G = sns.boxplot(
            data=plot_data,
            x='A_L',
            y='Revenue_G',
            order=plot_order,
            ax=ax_G,
            hue="A_L",
            hue_order=plot_order,
            legend=False,
            width=0.5,
            showfliers=True,
            palette="Set2"
        )
        ### add capture price
        """
        sns.violinplot(
            data=plot_data,
            x='A_L',
            y='CP_G_Revenue',
            order=plot_order,
            ax=ax_G,
            legend=False,
            alpha=0.3,
            width=0.8,
            inner=None,
            color= capture_color,
        )
        
        # 2. Add boxplots on top for Generator Revenue
        box_G = sns.boxplot(
            data=plot_data,
            x='A_L',
            y='CP_G_Revenue',
            order=plot_order,
            ax=ax_G,
            legend=False,
            width=0.3,
            showfliers=True,
            color = capture_color,
        )
        """



        #########
       
        # 3. Add violin plots behind boxplots for Load Revenue
        sns.violinplot(
            data=plot_data,
            x='A_L',
            y='Revenue_L',
            order=plot_order,
            ax=ax_L,
            hue = 'A_L',
            hue_order=plot_order,
            legend=False,
            alpha=0.3,
            width=0.5,
            cut=0,
            inner=None,
            palette="Set2"
        )
        
        # 4. Add boxplots on top for Load Revenue
        box_L = sns.boxplot(
            data=plot_data,
            x='A_L',
            y='Revenue_L',
            order=plot_order,
            ax=ax_L,
            hue= 'A_L',
            hue_order=plot_order,
            legend=False,
            width=0.3,
            showfliers=True,
            palette="Set2"
        )

        # Add Capture Price 
        """
        sns.violinplot(
            data=plot_data[contract_mask],
            x='A_L',
            y='CP_L_Revenue',
            order=plot_order,
            ax=ax_L,
            legend=False,
            alpha=0.3,
            width=0.6,
            cut=0,
            inner=None,
            color = capture_color,
        )
        
        # 2. Add boxplots on top for Generator Revenue
        box_L_CP = sns.boxplot(
            data=plot_data[contract_mask],
            x='A_L',
            y='CP_L_Revenue',
            order=plot_order,
            ax=ax_L,
            legend=False,
            width=0.3,
            showfliers=True,
            color = capture_color,
        )
        """ 
        
        """
        # 6. Calculate and display percentage changes between risk aversion levels
        if len(A_L_to_plot) > 1:
            g_means = [filtered_results[filtered_results['A_L'] == a_l]['Revenue_G'].mean() for a_l in A_L_to_plot]
            l_means = [filtered_results[filtered_results['A_L'] == a_l]['Revenue_L'].mean() for a_l in A_L_to_plot]
            
            for i in range(len(A_L_to_plot) - 1):
                # Generator percentage change
                pct_change_g = ((g_means[i+1] - g_means[i]) / g_means[i]) * 100
                ax_G.annotate(f'Average Earnings Increase{pct_change_g:.1f}%', 
                            xy=(i + 0.5, (g_means[i] + g_means[i+1])/2), 
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='center',
                            va='center',
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                # Load percentage change
                pct_change_l = ((l_means[i+1] - l_means[i]) / l_means[i]) * 100
                ax_L.annotate(f'Average Earnings Decrease {-pct_change_l:.1f}%', 
                            xy=(i + 0.5, (l_means[i] + l_means[i+1])/2), 
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='center',
                            va='center',
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        """

        # Add threatpoint lins 
        # Add Expectation of no contract earnings
        #ax_G.axhline(self.cm_data.net_earnings_no_contract_true_G.mean(), linestyle="--", color='grey', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_true_G.mean() :.2f} ")
        #ax_L.axhline(self.cm_data.net_earnings_no_contract_true_L.mean(), linestyle="--", color='grey', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_true_L.mean() :.2f} ")

        # Set titles and labels
        ax_G.set_title(f'Seller Revenue',fontsize=self.titlesize)
        ax_G.tick_params(axis='both', labelsize= self.legendsize)
        ax_G.set_xlabel('Risk Aversion $(A_B)$',fontsize=self.labelsize)
        ax_G.set_ylabel('Seller Revenue (Mio EUR)',fontsize=self.labelsize)
        ax_G.grid(True, linestyle='--', alpha=0.9, axis='y')

        ax_L.set_title(f'Buyer Revenue',fontsize=self.titlesize)
        ax_L.set_xlabel('Risk Aversion $(A_B)$', fontsize=self.labelsize)
        ax_L.set_ylabel('Buyer Revenue (Mio EUR)',fontsize=self.labelsize)
        ax_L.tick_params(axis='both', labelsize= self.legendsize)
        ax_L.grid(True, linestyle='--', alpha=0.9, axis='y')

        plt.suptitle(f"{self.cm_data.contract_type}: Earnings Distribution by Risk Aversion, $A_S$ = {fixed_A_G}", fontsize=self.suptitlesize)
        
        plt.tight_layout()

        # Save or show the figure
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

    def _nego_plot_earnings_boxplot(self, filename=None):

        """
        Plots boxplots of earnings for different negotiation levels.
        
        Parameters:
        -----------
            filename : str, optional
            Path to save the plot. If None, the plot will be displayed.
        """
        earnings_df = self.negotiation_earnings_df
        earnings_df["tau_G"] = earnings_df["tau_G"].round(2)  # Round to 2 decimal places for better readability
        earnings_df["tau_L"] = earnings_df["tau_L"].round(2)  # Round to 2 decimal places for better readability
        unique_tau_g = earnings_df["tau_G"].unique()

        # Get three evenly spaced positions
       # positions = np.linspace(0, len(unique_tau_g)-1, 7, dtype=int)
        #selected_tau_g = np.round(unique_tau_g[positions], 2)

        # Filter the DataFrame for these tau_G values
        selected_tau_g = unique_tau_g,   # Round to 2 decimal places for better readability
        #df_filtered = earnings_df[earnings_df["tau_G"].isin(selected_tau_g)]
        df_filtered = earnings_df
        AL_used = df_filtered['A_L'].unique()[0]  # Assuming A_L is constant for the filtered data
        AG_used = df_filtered['A_G'].unique()[0]  # Assuming A_G is constant for the filtered data
                

       # Prepare the no-contract data
        no_contract_g = self.cm_data.net_earnings_no_contract_true_G
        no_contract_l = self.cm_data.net_earnings_no_contract_true_L
        no_contract_df = pd.DataFrame({
            'Revenue_G': no_contract_g,
            'Revenue_L': no_contract_l,
            'tau_G': 'No Contract'  # Assign a string label for the category
        })

        CP_df = pd.DataFrame({
            'Revenue_G': self.CP_earnings_df["Revenue_G_CP"],
            'Revenue_L': self.CP_earnings_df["Revenue_L_CP"],
            'tau_G': 'Capture \n Price'  # Assign a string label for the category
        })

        # Combine the contract and no-contract data
        # Convert A_L to object type to allow mixing numbers and strings
        df_filtered= df_filtered.astype(object)
        plot_data = pd.concat([no_contract_df,CP_df,df_filtered], ignore_index=True)
            
        # Define the order for the x-axis categories
        nego_to_plot = sorted(selected_tau_g)
        plot_order = nego_to_plot.insert(0,'No Contract')  # Add 'No Contract' at the beginning
        
        
        # Create figure with more space at the bottom for the table
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        ax_G = axes[0]
        ax_L = axes[1]
        
        # 1. Add violin plots behind boxplots for Generator Revenue
        sns.violinplot(
            data=plot_data,
            x='tau_G',
            y='Revenue_G',
            order=plot_order,
            ax=ax_G,
            hue="tau_G", 
            hue_order=plot_order,
            legend=False,
            alpha=0.3,
            cut=0,
            inner=None,
            palette="Set2"
        )
        
        # 2. Add boxplots on top for Generator Revenue
        box_G = sns.boxplot(
            data=plot_data,
            x='tau_G',
            y='Revenue_G',
            order=plot_order,
            ax=ax_G,
            hue="tau_G",
            hue_order=plot_order,
            legend=False,
            width=0.5,
            showfliers=True,
            palette="Set2"
        )
       
        # 3. Add violin plots behind boxplots for Load Revenue
        sns.violinplot(
            data=plot_data,
            x='tau_G',
            y='Revenue_L',
            order=plot_order,
            ax=ax_L,
            hue = 'tau_G',
            hue_order=plot_order,
            legend=False,
            alpha=0.3,
            cut=0,
            inner=None,
            palette="Set2"
        )
        
        # 4. Add boxplots on top for Load Revenue
        box_L = sns.boxplot(
            data=plot_data,
            x='tau_G',
            y='Revenue_L',
            order=plot_order,
            ax=ax_L,
            hue= 'tau_G',
            hue_order=plot_order,
            legend=False,
            width=0.5,
            showfliers=True,
            palette="Set2"
        )
        
        """
        # 6. Calculate and display percentage changes between risk aversion levels
        if len(A_L_to_plot) > 1:
            g_means = [filtered_results[filtered_results['A_L'] == a_l]['Revenue_G'].mean() for a_l in A_L_to_plot]
            l_means = [filtered_results[filtered_results['A_L'] == a_l]['Revenue_L'].mean() for a_l in A_L_to_plot]
            
            for i in range(len(A_L_to_plot) - 1):
                # Generator percentage change
                pct_change_g = ((g_means[i+1] - g_means[i]) / g_means[i]) * 100
                ax_G.annotate(f'Average Earnings Increase{pct_change_g:.1f}%', 
                            xy=(i + 0.5, (g_means[i] + g_means[i+1])/2), 
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='center',
                            va='center',
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                # Load percentage change
                pct_change_l = ((l_means[i+1] - l_means[i]) / l_means[i]) * 100
                ax_L.annotate(f'Average Earnings Decrease {-pct_change_l:.1f}%', 
                            xy=(i + 0.5, (l_means[i] + l_means[i+1])/2), 
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='center',
                            va='center',
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        """
        # Set titles and labels
        ax_G.set_title(f'Generator Revenue',fontsize=self.titlesize)
        ax_G.set_xlabel('Negotiation Power $(\\tau_G)$',fontsize=self.labelsize)
        ax_G.set_ylabel('Generator Revenue (Mio EUR)',fontsize=self.labelsize)
        ax_G.tick_params(axis='both', labelsize= self.legendsize-1)
        ax_G.grid(True, linestyle='--', alpha=0.7, axis='y')

        ax_L.set_title(f'Load Revenue',fontsize=self.titlesize)
        ax_L.set_xlabel('Negotiation Power $(\\tau_L)$', fontsize=self.labelsize)
        ax_L.set_ylabel('Load Revenue (Mio EUR)',fontsize=self.labelsize)
        ax_L.tick_params(axis='both', labelsize= self.legendsize-1)
        ax_L.grid(True, linestyle='--', alpha=0.7, axis='y')

        plt.suptitle(f"{self.cm_data.contract_type}: Earnings Distribution by Negotiation Power with Risk Aversion $A_G$={AG_used}, $A_L$={AL_used}", fontsize=self.suptitlesize)

        plt.tight_layout()

        # Save or show the figure
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

    def _plot_parameter_sensitivity_spider(self, bias = False, filename=None):
        """
        Create a spider/radar plot showing how different parameters affect contract outcomes.
        Compares the sensitivity of contract parameters to different input variables.
        """
        plt.figure(figsize=(12, 10))
        
       

        # Define the metrics we want to compare
        if self.cm_data.contract_type == "PAP":
            metrics = ['StrikePrice', 'Gamma', 'Utility_G', 'Utility_L', 'ThreatPoint_G', 'ThreatPoint_L']
        else:
            metrics = ['StrikePrice', 'ContractAmount', 'Utility_G', 'Utility_L', 'ThreatPoint_G', 'ThreatPoint_L']
        
        
      # Prepare bias data as in spider plot
        price_bias_sensitivity_df = self.price_bias_sensitivity_df.copy()
        price_bias_sensitivity_df['KG_Factor'] = 1.0 + self.price_bias_sensitivity_df['KG_Factor']
        price_bias_sensitivity_df['KL_Factor'] = 1.0 + self.price_bias_sensitivity_df['KL_Factor']
        price_bias_KG = price_bias_sensitivity_df[price_bias_sensitivity_df['KL_Factor'] == 1.00]
        price_bias_KL = price_bias_sensitivity_df[price_bias_sensitivity_df['KG_Factor'] == 1.00]

        # Bias Production 

        production_bias_sensitivity_df = self.production_bias_sensitivity_df.copy()
        production_bias_sensitivity_df['KG_Factor'] = 1.0 + self.production_bias_sensitivity_df['KG_Factor']
        production_bias_sensitivity_df['KL_Factor'] = 1.0 + self.production_bias_sensitivity_df['KL_Factor']
        production_bias_KG = production_bias_sensitivity_df[production_bias_sensitivity_df['KL_Factor'] == 1.00]
        production_bias_KL = production_bias_sensitivity_df[production_bias_sensitivity_df['KG_Factor'] == 1.00]
        # Define the sensitivity analyses to process

        elasticities = {}

        def _compute_local_elasticity(df_in: pd.DataFrame, factor_col: str, metric_cols: list[str], baseline: float) -> dict:
            """Compute local elasticities at baseline using central diff if possible, else local linear fit.
            E = (dY/dX) * (X0 / Y0). Handles NaNs by skipping invalid rows; returns NaN when insufficient data.
            """
            if df_in is None or df_in.empty:
                return {m: np.nan for m in metric_cols}

            df = df_in[[factor_col] + metric_cols].copy()
            # Keep rows where factor is finite
            df = df[np.isfinite(df[factor_col])]
            if df.empty:
                return {m: np.nan for m in metric_cols}

            # Sort by factor and drop duplicate X keeping closest to baseline
            df = df.sort_values(factor_col)
            # Identify indices around baseline
            x_vals = df[factor_col].values.astype(float)

            # Find bracket points around baseline
            left_mask = x_vals < baseline
            right_mask = x_vals > baseline

            result = {}
            for m in metric_cols:
                y_series = df[m].astype(float)
                # Valid rows for this metric
                valid = np.isfinite(y_series.values)
                if valid.sum() < 2:
                    result[m] = np.nan
                    continue

                x = x_vals[valid]
                y = y_series.values[valid]

                # Recompute masks on valid-only arrays
                left_idx = np.where(x < baseline)[0]
                right_idx = np.where(x > baseline)[0]

                slope = np.nan
                y0 = np.nan
                # Try exact baseline first
                exact_idx = np.where(np.isclose(x, baseline))[0]
                if exact_idx.size > 0:
                    y0 = y[exact_idx[0]]
                # Central difference if we have neighbors on both sides
                if left_idx.size > 0 and right_idx.size > 0:
                    iL = left_idx[-1]
                    iR = right_idx[0]
                    xL, yL = x[iL], y[iL]
                    xR, yR = x[iR], y[iR]
                    if np.isfinite(yL) and np.isfinite(yR) and xR != xL:
                        slope = (yR - yL) / (xR - xL)
                        if not np.isfinite(y0):
                            # Linear interpolate y0
                            y0 = yL + (baseline - xL) * slope
                # Fallback: local linear fit using up to 5 nearest points
                if not np.isfinite(slope):
                    if x.size >= 2:
                        # Select nearest k points
                        k = min(5, x.size)
                        order = np.argsort(np.abs(x - baseline))[:k]
                        x_fit = x[order]
                        y_fit = y[order]
                        if np.unique(x_fit).size >= 2:
                            coeffs = np.polyfit(x_fit, y_fit, deg=1)
                            slope = coeffs[0]
                            y0 = np.polyval(coeffs, baseline)
                # Compute elasticity
                if not np.isfinite(slope) or not np.isfinite(y0) or np.isclose(y0, 0.0):
                    result[m] = np.nan
                else:
                    result[m] = float(slope * (baseline / y0))
            return result
        if bias == False:
            sensitivity_analyses = [
                {
                'name': 'Production (Mean)',
                'df': self.production_sensitivity_mean_df,
                'factor_col': 'Production_Change',
            },
            {
                'name': 'Prod. Capture Rate (Mean)',
                'df': self.gen_CR_sensitivity_results,
                'factor_col': 'CaptureRate_Change',
            },
            {
                'name': 'Load. Capture Rate (Mean)',
                'df': self.load_CR_sensitivity_df,
                'factor_col': 'Load_CaptureRate_Change'
            },
            {
                'name': 'Load Sensitivity (Mean)',
                'df': self.load_sensitivity_mean_df,
                'factor_col': 'Load_Change',
            },

            {
                'name': 'Price Sensitivity (Mean)',
                'df': self.price_sensitivity_mean_df,
                'factor_col': 'Price_Change',

            },
            {
                'name': 'Price Sensitivity (Std)',
                'df': self.price_sensitivity_std_df,
                'factor_col': 'Price_Change',

            },
            {
                'name': 'Production (Std)',
                'df': self.production_sensitivity_std_df,
                'factor_col': 'Production_Change',

            },
            {
                'name': 'Load Sensitivity (Std)',
                'df': self.load_sensitivity_std_df,
                'factor_col': 'Load_Change',
                
            }
          
        ]
        else:
            sensitivity_analyses = [
             {
                'name': 'Price Bias (G)',
                    'df': price_bias_KG,
                'factor_col': 'KG_Factor',
            },
            {
                'name': 'Price Bias (L)',
                'df': price_bias_KL,
                'factor_col': 'KL_Factor',
            },
            {
                'name': 'Production Bias (G)',
                'df': production_bias_KG,
                'factor_col': 'KG_Factor'
            },
            {
                'name': 'Production Bias (L)',
                'df': production_bias_KL,
                'factor_col': 'KL_Factor',
            },
                  ]

        
        for analysis in sensitivity_analyses:
            df = analysis['df']
            factor_col = analysis['factor_col']
            # All sensitivity factors are modeled as multiplicative changes around 1.0
            baseline = 1.0
            vals = _compute_local_elasticity(df, factor_col=factor_col, metric_cols=metrics, baseline=baseline)
            elasticities[analysis['name']] = {k: (None if (v is None) else (np.nan if not np.isfinite(v) else round(v, 3))) for k, v in vals.items()}
        
        # You could add more parameter elasticities here (risk aversion, bias, etc.)
        
        # Create the spider plot
        # Set up the radar chart
        categories = metrics
        N = len(categories)
        
        # Create angles for each metric (evenly spaced around the circle)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection (for radar chart)
        ax = plt.subplot(111, polar=True)
        
        # Set the first axis to be on top
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Add a bold circle at zero elasticity
        zero_circle = np.zeros(100)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, zero_circle, 'k-', linewidth=2.5)  # Bold black line for zero

        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], [f"${cat}$" for cat in categories], size=self.labelsize)        
        # Set y limits
        ax.set_ylim(-1, 1.25)
        
        # Add parameter elasticities
        
        for i, (param, values) in enumerate(elasticities.items()):
            values_ordered = [values[metric] for metric in categories]
            values_ordered += values_ordered[:1]  # Close the loop
            
            ax.plot(angles, values_ordered, linewidth=2, linestyle='solid', 
                    label=f"${param}$")
            ax.fill(angles, values_ordered, alpha=0.1,)
        
        # Add legend
        plt.legend(loc='upper left')
        
        # Add title
        plt.title(f'{self.cm_data.contract_type}: Parameter Sensitivity Comparison Elaticities by Factor', 
                size=self.titlesize, y=1.1)

        # Add reference circles and lines
        plt.grid(True)
        
        # Add annotations for key insights
        
        
        # Save or show the plot
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Spider plot saved to {filepath}")
            plt.close()
        else:
            plt.show()

    def _plot_elasticity_tornado(self, bias = False, metrics='StrikePrice', filename=None):

        """
        Create a tornado plot showing the elasticity of a single metric
        with respect to different input factors.
        """

        if isinstance(metrics, str):
            metrics = [metrics]


        # Prepare the same elasticities as in the spider plot
        #if self.cm_data.contract_type == "PAP":
        #    metrics = ['StrikePrice', 'Gamma', 'Utility_G', 'Utility_L', 'ThreatPoint_G', 'ThreatPoint_L']
        #else:
        #    metrics = ['StrikePrice', 'ContractAmount', 'Utility_G', 'Utility_L', 'ThreatPoint_G', 'ThreatPoint_L']

        # Prepare bias data as in spider plot
        price_bias_sensitivity_df = self.price_bias_sensitivity_df.copy()
        price_bias_sensitivity_df['KG_Factor'] = 1.0 + self.price_bias_sensitivity_df['KG_Factor']
        price_bias_sensitivity_df['KL_Factor'] = 1.0 + self.price_bias_sensitivity_df['KL_Factor']
        price_bias_KG = price_bias_sensitivity_df[price_bias_sensitivity_df['KL_Factor'] == 1.00]
        price_bias_KL = price_bias_sensitivity_df[price_bias_sensitivity_df['KG_Factor'] == 1.00]

        # Bias Production 

        production_bias_sensitivity_df = self.production_bias_sensitivity_df.copy()
        production_bias_sensitivity_df['KG_Factor'] = 1.0 + self.production_bias_sensitivity_df['KG_Factor']
        production_bias_sensitivity_df['KL_Factor'] = 1.0 + self.production_bias_sensitivity_df['KL_Factor']
        production_bias_KG = production_bias_sensitivity_df[production_bias_sensitivity_df['KL_Factor'] == 1.00]
        production_bias_KL = production_bias_sensitivity_df[production_bias_sensitivity_df['KG_Factor'] == 1.00]
        # Define the sensitivity analyses to process

        if bias == False:
            sensitivity_analyses = [
                {
                'name': 'Production (Mean)',
                'df': self.production_sensitivity_mean_df,
                'factor_col': 'Production_Change',
            },
            {
                'name': 'Prod. Capture Rate (Mean)',
                'df': self.gen_CR_sensitivity_results,
                'factor_col': 'CaptureRate_Change',
            },
            {
                'name': 'Load. Capture Rate (Mean)',
                'df': self.load_CR_sensitivity_df,
                'factor_col': 'Load_CaptureRate_Change'
            },
            {
                'name': 'Load Sensitivity (Mean)',
                'df': self.load_sensitivity_mean_df,
                'factor_col': 'Load_Change',
            },

            {
                'name': 'Price Sensitivity (Mean)',
                'df': self.price_sensitivity_mean_df,
                'factor_col': 'Price_Change',

            },
            {
                'name': 'Price Sensitivity (Std)',
                'df': self.price_sensitivity_std_df,
                'factor_col': 'Price_Change',

            },
            {
                'name': 'Production (Std)',
                'df': self.production_sensitivity_std_df,
                'factor_col': 'Production_Change',

            },
            {
                'name': 'Load Sensitivity (Std)',
                'df': self.load_sensitivity_std_df,
                'factor_col': 'Load_Change',
                
            }
          
        ]
        else:
            sensitivity_analyses = [
             {
                'name': 'Price Bias (G)',
                    'df': price_bias_KG,
                'factor_col': 'KG_Factor',
            },
            {
                'name': 'Price Bias (L)',
                'df': price_bias_KL,
                'factor_col': 'KL_Factor',
            },
            {
                'name': 'Production Bias (G)',
                'df': production_bias_KG,
                'factor_col': 'KG_Factor'
            },
            {
                'name': 'Production Bias (L)',
                'df': production_bias_KL,
                'factor_col': 'KL_Factor',
            },
                  ]

        n_metrics = len(metrics)
        ncols = int(np.ceil(np.sqrt(n_metrics)))
        nrows = int(np.ceil(n_metrics / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, 6 * nrows))
        axes = axes.flatten()  # Flatten for easy indexing

        for idx, metric in enumerate(metrics):
            elasticities = {}
            for analysis in sensitivity_analyses:
                df = analysis['df']
                if df is None or df.empty:
                    print(f"Warning: DataFrame for {analysis['name']} is empty. Skipping.")
                    continue
                factor_col = analysis['factor_col']
                # All sensitivity factors are modeled as multiplicative changes around 1.0
                baseline = 1.0
                vals = self._safe_local_elasticity_single(df, factor_col=factor_col, metric_col=metric, baseline=baseline)
                if vals is not None and np.isfinite(vals):
                    # Keep full precision for plotting; format only in labels
                    elasticities[analysis['name']] = float(vals)
                else:
                    elasticities[analysis['name']] = np.nan

            # Prepare data for plotting
            factors = list(elasticities.keys())
            values = [elasticities[f] for f in factors]
            sorted_indices = np.argsort(np.abs(values))[::-1]
            sorted_factors = [factors[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]

            tornado_df = pd.DataFrame({
                'Factor': sorted_factors,
                'Elasticity': sorted_values
            })

            ax = axes[idx]
            sns.barplot(
                data=tornado_df,
                y='Factor',
                x='Elasticity',
                orient='h',
                color='skyblue',
                ax=ax
            )
            ax.axvline(0, color='k', linewidth=1)
            ax.set_xlabel(f'Elasticity of ${metric}$',fontsize=self.labelsize)

            ax.set_title(f'Sensitivity of ${metric}$',fontsize=self.titlesize)
            ax.grid(axis='x', linestyle=':', alpha=0.7)


            for bar, value in zip(ax.patches, sorted_values):
                width = bar.get_width()
                y     = bar.get_y() + bar.get_height()/2

                offset = -15            # points   (change to taste)
                ha     = 'left'
                if width < 0:         # put the label on the other side of negative bars
                    offset = -3
                    ha     = 'right'

                txt = ax.annotate(f'{value:.3f}',
                                xy=(width, y),                 # end of the bar
                                xytext=(offset, 0),            # shift *offset* points
                                textcoords='offset points',
                                va='center',
                                ha=ha,
                                fontsize=self.legendsize,
                                zorder=3)

       
            xmin, xmax = ax.get_xlim()
            span = xmax - xmin
            ax.set_xlim(xmin - 0.05*span, xmax + 0.05*span)  
        # Delete unused

    def _plot_elasticity_tornado_vs_risk(self,
                                     bias = False,
                                     fixed_A_G_values=None,
                                     fixed_A_L_values=None,
                                     metrics=['StrikePrice'],
                                     filename=None,
                                     fix: str = 'A_G'):
        """
        Plot grouped tornado bars (elasticities) for each metric across risk aversion combinations,
        computing elasticities with a log-log regression:

            log(metric) = a + b * log(factor)  ->  elasticity = b

        Parameters
        ----------
        bias : bool
            If True, analyzes bias factors (KG_Factor, KL_Factor) instead of production/price factors
        fixed_A_G_values : list[float] | None
            Values of A_G to fix when fix='A_G'
        fixed_A_L_values : list[float] | None
            Values of A_L to fix when fix='A_L'
        metrics : list[str]
            Metrics for which elasticities are plotted.
        filename : str | None
            If provided, each figure is saved as <filename>_<metric>.png
        fix : {'A_G','A_L'}
            Which party's risk aversion to hold fixed while grouping bars by the other party.
        """

        if bias == False:
            df = self.elasticity_vs_risk_df.copy()
            
            if df is None or df.empty:
                print("No data for elasticity_vs_risk plotting.")
                return

            # Map displayed factor label -> underlying factor change column
            factor_xcol = {
                'Production (Expected)': 'Production_Change',
                'Production (Std)': 'Production_Change',
                'Price Sensitivity (Expected)': 'Price_Change',
                'Price Sensitivity (Std)': 'Price_Change',
                'Load Sensitivity (Expected)': 'Load_Change',
                'Load Sensitivity (Std)': 'Load_Change',
                'Prod. Capture Rate (Expected)': 'CaptureRate_Change',
                'Load. Capture Rate (Expected)': 'Load_CaptureRate_Change',
            }

            factor_order = [
                'Prod. Capture Rate (Expected)',
                'Price Sensitivity (Expected)',
                'Production (Expected)',
                'Production (Std)',
                'Price Sensitivity (Std)',
                'Load Sensitivity (Expected)',
                'Load. Capture Rate (Expected)',
                'Load Sensitivity (Std)',
            ]
            
        else:  # bias == True
            df = self.bias_risk_elasticity_df.copy()
            
            if df is None or df.empty:
                print("No data for bias_risk_elasticity plotting.")
                return

            # Convert bias factors to multiplicative form (from additive)
            df['KG_Factor_mult'] = 1.0 + df['KG_Factor']
            df['KL_Factor_mult'] = 1.0 + df['KL_Factor']

            # Create separate datasets for each bias scenario
            scenarios = []
            
            # Scenario 1: KG_Factor = 0 (no bias for G), varying KL_Factor
            # This gives us elasticity w.r.t. L's bias
            kg_zero = df[df['KG_Factor'] == 0.0].copy()
            if not kg_zero.empty:
                # For Price Bias
                price_bias_kg_zero = kg_zero[kg_zero['Factor'] == 'Price Bias'].copy()
                price_bias_kg_zero['Factor'] = 'Price Bias (L)'
                price_bias_kg_zero['varying_factor'] = 'KL_Factor_mult'
                scenarios.append(price_bias_kg_zero)
                
                # For Production Bias  
                prod_bias_kg_zero = kg_zero[kg_zero['Factor'] == 'Production Bias'].copy()
                prod_bias_kg_zero['Factor'] = 'Production Bias (L)'
                prod_bias_kg_zero['varying_factor'] = 'KL_Factor_mult'
                scenarios.append(prod_bias_kg_zero)

            # Scenario 2: KL_Factor = 0 (no bias for L), varying KG_Factor
            # This gives us elasticity w.r.t. G's bias
            kl_zero = df[df['KL_Factor'] == 0.0].copy()
            if not kl_zero.empty:
                # For Price Bias
                price_bias_kl_zero = kl_zero[kl_zero['Factor'] == 'Price Bias'].copy()
                price_bias_kl_zero['Factor'] = 'Price Bias (G)'
                price_bias_kl_zero['varying_factor'] = 'KG_Factor_mult'
                scenarios.append(price_bias_kl_zero)
                
                # For Production Bias
                prod_bias_kl_zero = kl_zero[kl_zero['Factor'] == 'Production Bias'].copy()
                prod_bias_kl_zero['Factor'] = 'Production Bias (G)'
                prod_bias_kl_zero['varying_factor'] = 'KG_Factor_mult'
                scenarios.append(prod_bias_kl_zero)

            # Combine all scenarios
            df = pd.concat(scenarios, ignore_index=True)
            
            if df.empty:
                print("No valid bias scenarios found (need KG_Factor=0 or KL_Factor=0 rows).")
                return

            # Factor mapping for bias analysis
            factor_xcol = {
                'Price Bias (G)': 'KG_Factor_mult',
                'Price Bias (L)': 'KL_Factor_mult', 
                'Production Bias (G)': 'KG_Factor_mult',
                'Production Bias (L)': 'KL_Factor_mult',
            }

            factor_order = [
                'Price Bias (G)',
                'Price Bias (L)',
                'Production Bias (G)', 
                'Production Bias (L)',
            ]

        def _loglog_elasticity(block: pd.DataFrame, factor_col: str, metric_col: str) -> float | None:
            """
            Return slope of log(metric) vs log(factor) for the block (elasticity).
            Requires >= 2 positive finite observations for both.
            """
            if block is None or block.empty or factor_col not in block.columns or metric_col not in block.columns:
                return np.nan
            x_raw = pd.to_numeric(block[factor_col], errors='coerce')
            y_raw = pd.to_numeric(block[metric_col], errors='coerce')
            mask = (x_raw > 0) & (y_raw > 0) & np.isfinite(x_raw) & np.isfinite(y_raw)
            if mask.sum() < 2:
                return np.nan
            lx = np.log(x_raw[mask].values)
            ly = np.log(y_raw[mask].values)
            # Guard against zero variance
            if np.allclose(lx, lx[0]) or np.allclose(ly, ly[0]):
                return np.nan
            slope = np.polyfit(lx, ly, 1)[0]
            # Clean near-zero numerical noise
            if np.isfinite(slope) and abs(slope) < 1e-9:
                slope = 0.0
            return float(slope) if np.isfinite(slope) else np.nan

        # Main plotting logic
        for metric in metrics:
            if fix == 'A_G':
                if not fixed_A_G_values:
                    print("No fixed_A_G_values provided for fix='A_G'.")
                    return
                for ag in fixed_A_G_values:
                    sub = df[df['A_G'].round(3) == round(ag, 3)].copy()
                    if sub.empty:
                        print(f"Warning: No rows for A_G={ag} in df.")
                        continue
                    var_values = sorted(sub['A_L'].dropna().unique().tolist())
                    present_factors = [f for f in factor_order if f in sub['Factor'].unique()]
                    if not present_factors:
                        print(f"No recognized factors for A_G={ag}")
                        continue

                    # rows = factors, cols = varying A_L
                    data = {al: [] for al in var_values}
                    for factor in present_factors:
                        fcol = factor_xcol.get(factor)
                        for al in var_values:
                            block = sub[(sub['Factor'] == factor) &
                                        (sub['A_L'].round(3) == round(al, 3))]
                            val = _loglog_elasticity(block, fcol, metric) if fcol else np.nan
                            data[al].append(val if np.isfinite(val) else np.nan)

                    plot_df = pd.DataFrame(data, index=present_factors)
                    valid_cols = [c for c in plot_df.columns if np.isfinite(plot_df[c].values).any()]
                    if not valid_cols:
                        print(f"No valid elasticities (log-log) for A_G={ag} across any A_L.")
                        continue
                    plot_df = plot_df[valid_cols]

                    # Plot
                    n_factors = len(present_factors)
                    n_groups = len(valid_cols)
                    bar_h = 0.8 / max(1, n_groups)
                    y_positions = np.arange(n_factors)

                    fig, ax = plt.subplots(figsize=(8.5, 0.5 * n_factors + 2))
                    for i, al in enumerate(valid_cols):
                        vals = plot_df[al].values
                        color = self._color_for_risk(al, kind='A_L')
                        ax.barh(y_positions + (i - (n_groups - 1)/2) * bar_h,
                                vals, height=bar_h, color=color, label=f"A_L={al}")

                    ax.axvline(0.0, color='k', linewidth=1)
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels([f"{f}" for f in present_factors], fontsize=self.legendsize)
                    ax.set_xlabel(f"Elasticity of ${metric}$", fontsize=self.labelsize)
                    
                    bias_suffix = " (Bias Analysis)" if bias else ""
                    ax.set_title(f"Parameter Sensitivity {self.cm_data.contract_type}, A_G={ag}{bias_suffix}",
                                fontsize=self.titlesize)
                    ax.grid(axis='x', linestyle=':', alpha=0.6)
                    ax.legend(fontsize=self.legendsize-1, title_fontsize=self.legendsize-1,loc="upper right",
                            ncol=min(3, n_groups))

                    plt.tight_layout()
                    if filename:
                        bias_suffix = "_bias" if bias else ""
                        fname = f"{filename}{bias_suffix}_{metric}_AG_{ag}.png"
                        plt.savefig(fname, bbox_inches='tight', dpi=300)
                        print(f"Saved log-log elasticity-vs-risk tornado: {fname}")
                        plt.close(fig)
                    else:
                        plt.show()

            elif fix == 'A_L':
                if not fixed_A_L_values:
                    print("No fixed_A_L_values provided for fix='A_L'.")
                    return
                for al in fixed_A_L_values:
                    sub = df[df['A_L'].round(3) == round(al, 3)].copy()
                    if sub.empty:
                        print(f"Warning: No rows for A_L={al} in df.")
                        continue
                    var_values = sorted(sub['A_G'].dropna().unique().tolist())
                    present_factors = [f for f in factor_order if f in sub['Factor'].unique()]
                    if not present_factors:
                        print(f"No recognized factors for A_L={al}")
                        continue

                    data = {ag: [] for ag in var_values}
                    for factor in present_factors:
                        fcol = factor_xcol.get(factor)
                        for ag in var_values:
                            block = sub[(sub['Factor'] == factor) &
                                        (sub['A_G'].round(3) == round(ag, 3))]
                            val = _loglog_elasticity(block, fcol, metric) if fcol else np.nan
                            data[ag].append(val if np.isfinite(val) else np.nan)

                    plot_df = pd.DataFrame(data, index=present_factors)
                    valid_cols = [c for c in plot_df.columns if np.isfinite(plot_df[c].values).any()]
                    if not valid_cols:
                        print(f"No valid elasticities (log-log) for A_L={al} across any A_G.")
                        continue
                    plot_df = plot_df[valid_cols]

                    n_factors = len(present_factors)
                    n_groups = len(valid_cols)
                    bar_h = 0.8 / max(1, n_groups)
                    y_positions = np.arange(n_factors)

                    fig, ax = plt.subplots(figsize=(8.5, 0.5 * n_factors + 2))
                    for i, ag in enumerate(valid_cols):
                        vals = plot_df[ag].values
                        color = self._color_for_risk(ag, kind='A_G')
                        ax.barh(y_positions + (i - (n_groups - 1)/2) * bar_h,
                                vals, height=bar_h, color=color, label=f"A_G={ag}")

                    ax.axvline(0.0, color='k', linewidth=1)
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels([f"{f}" for f in present_factors], fontsize=self.legendsize)
                    ax.set_xlabel(f"Elasticity of ${metric}$", fontsize=self.labelsize)

                    bias_suffix = " (Bias Analysis)" if bias else ""
                    ax.set_title(f"Parameter Sensitivity {self.cm_data.contract_type}, A_L={al}{bias_suffix}",
                                fontsize=self.titlesize)
                    ax.grid(axis='x', linestyle=':', alpha=0.6)
                    ax.legend(fontsize=self.legendsize-1, title_fontsize=self.legendsize-1,loc="upper right",
                            ncol=min(3, n_groups))

                    plt.tight_layout()
                    if filename:
                        bias_suffix = "_bias" if bias else ""
                        fname = f"{filename}{bias_suffix}_{metric}_AL_{al}.png"
                        plt.savefig(fname, bbox_inches='tight', dpi=300)
                        print(f"Saved log-log elasticity-vs-risk tornado: {fname}")
                        plt.close(fig)
                    else:
                        plt.show()
            else:
                print("Invalid fix mode. Use fix='A_G' or fix='A_L'.")
        
    def _plot_nash_product_evolution(self, filename=None):
        """
        Plot how Nash Product changes across different sensitivity analyses.
        This is crucial for understanding the efficiency of negotiation outcomes.
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        # 1. Risk Aversion Impact on Nash Product
        risk_df = self.risk_sensitivity_df.copy()
        #risk_df['Nash_Product'] = risk_df['Nash_Product']**2 # Since it is using the 0.5 utlity function from gurobi, manually would give the same results. 
        
        # Create pivot table for heatmap
        pivot = risk_df.pivot_table(
            index='A_L',
            columns='A_G',
            values='Nash_Product',
            aggfunc='mean'
        )

        pivot = pivot.sort_index(ascending=False)
        
        sns.heatmap(
            pivot,
            ax=ax,
            cmap='RdYlGn',
            center=pivot.mean().mean(),
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Nash Product'}
        )
        ax.set_title('Nash Product: Risk Aversion Impact', fontsize=self.titlesize)
        ax.set_xlabel('Generator Risk Aversion ($A_G$)', fontsize=self.labelsize)
        ax.set_ylabel('Load Risk Aversion ($A_L$)', fontsize=self.labelsize)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Nash Product evolution plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

    def _plot_utility_space(self, filename=None):
        """
        Plot the utility space showing feasible region, threat points, and Nash bargaining solution.
        This is fundamental for understanding the negotiation dynamics.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 1. Utility Space with Different Risk Scenarios
        risk_df = self.risk_sensitivity_df.copy()
        
        # Plot threat points and contract utilities for different scenarios
        scenarios = [
            {'A_G': 0, 'A_L': 0, 'label': 'Risk Neutral', 'color': 'green', 'marker': 'o'},
            {'A_G': 0.5, 'A_L': 0.5, 'label': 'Moderate Risk', 'color': 'orange', 'marker': 's'},
            {'A_G': 1, 'A_L': 1, 'label': 'High Risk', 'color': 'red', 'marker': '^'},
            {'A_G': 0, 'A_L': 1, 'label': 'Asymmetric (G:0, L:1)', 'color': 'purple', 'marker': 'D'},
            {'A_G': 1, 'A_L': 0, 'label': 'Asymmetric (G:1, L:0)', 'color': 'blue', 'marker': 'v'}
        ]
        
        for scenario in scenarios:
            row = risk_df[(risk_df['A_G'] == scenario['A_G']) & (risk_df['A_L'] == scenario['A_L'])]
            if not row.empty:
                # Plot threat point
                ax1.scatter(row['ThreatPoint_G'], row['ThreatPoint_L'], 
                        color=scenario['color'], marker=scenario['marker'], 
                        s=200, alpha=0.5, edgecolor='black', linewidth=2,
                        label=f"{scenario['label']} (Threat)")
                
                # Plot contract utility
                ax1.scatter(row['Utility_G'], row['Utility_L'], 
                        color=scenario['color'], marker=scenario['marker'], 
                        s=200, alpha=1.0, edgecolor='black', linewidth=2)
                
                # Draw line from threat point to contract
                ax1.plot([row['ThreatPoint_G'].iloc[0], row['Utility_G'].iloc[0]], 
                        [row['ThreatPoint_L'].iloc[0], row['Utility_L'].iloc[0]], 
                        color=scenario['color'], linestyle='--', alpha=0.5, linewidth=2)
        
        # Add Nash bargaining curve (approximate)
        # This would be the Pareto frontier in the utility space
        utility_g_range = np.linspace(risk_df['ThreatPoint_G'].min(), risk_df['Utility_G'].max(), 100)
        
        ax1.set_xlabel('Generator Utility', fontsize=self.labelsize)
        ax1.set_ylabel('Load Utility', fontsize=self.labelsize)
        ax1.set_title('Utility Space: Contract vs Threat Points', fontsize=self.titlesize)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add annotations
        ax1.annotate('Feasible Region', xy=(risk_df['Utility_G'].mean(), risk_df['Utility_L'].mean()),
                    xytext=(risk_df['Utility_G'].mean() + 20, risk_df['Utility_L'].mean() + 20),
                    fontsize=12, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        # 2. Nash Product Contours
        # Create a grid for contour plot
        util_g_min = min(risk_df['ThreatPoint_G'].min(), risk_df['Utility_G'].min()) - 10
        util_g_max = max(risk_df['ThreatPoint_G'].max(), risk_df['Utility_G'].max()) + 10
        util_l_min = min(risk_df['ThreatPoint_L'].min(), risk_df['Utility_L'].min()) - 10
        util_l_max = max(risk_df['ThreatPoint_L'].max(), risk_df['Utility_L'].max()) + 10
        
        g_range = np.linspace(util_g_min, util_g_max, 100)
        l_range = np.linspace(util_l_min, util_l_max, 100)
        G, L = np.meshgrid(g_range, l_range)
        
        # Calculate Nash product for each point (using average threat points)
        threat_g_avg = risk_df['ThreatPoint_G'].mean()
        threat_l_avg = risk_df['ThreatPoint_L'].mean()
        
        # Nash product = (U_G - T_G) * (U_L - T_L)
        nash_product = np.maximum(G - threat_g_avg, 0) * np.maximum(L - threat_l_avg, 0)
        
        # Plot contours
        contour = ax2.contour(G, L, nash_product, levels=20, cmap='viridis', alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=10)
        
        # Overlay actual points
        scatter = ax2.scatter(risk_df['Utility_G'], risk_df['Utility_L'], 
                            c=risk_df['Nash_Product'], s=200, cmap='viridis',
                            edgecolor='black', linewidth=2)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Nash Product', fontsize=self.legendsize)
        
        ax2.set_xlabel('Generator Utility', fontsize=self.labelsize)
        ax2.set_ylabel('Load Utility', fontsize=self.labelsize)
        ax2.set_title('Nash Product Contours in Utility Space', fontsize=self.titlesize)
        ax2.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'{self.cm_data.contract_type}: Utility Space Analysis', fontsize=self.suptitlesize)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Utility space plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

    def _plot_disagreement_points(self,filename=None):

        """
        Plot the disagreement points for different risk aversion levels.
        This is crucial for understanding the negotiation dynamics and potential outcomes.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        A_values = np.linspace(0, 1, 100)
        d_G_vals = np.zeros(len(A_values))
        d_L_vals = np.zeros(len(A_values))

        for i, A in enumerate(A_values):
            d_G_vals[i] = ((1 - A) * (self.cm_data.PROB * self.cm_data.net_earnings_no_contract_priceG_G).sum() +
                        A * self.cm_data.CVaR_no_contract_priceG_G)
            d_L_vals[i] = ((1 - A) * (self.cm_data.PROB * self.cm_data.net_earnings_no_contract_priceL_L).sum() +
                        A * self.cm_data.CVaR_no_contract_priceL_L)
        ax2 = ax.twinx()  # Create a second y-axis for Load disagreement points

        # Plot disagreement points
        line2, = ax2.plot(A_values, d_L_vals, label='Load Disagreement Point', color='orange', linewidth=2,  marker='o', markevery=10)
        line1, = ax.plot(A_values, d_G_vals, label='Generator Disagreement Point', color='blue',  marker='o', markevery=5)


        # Add labels and title
        ax.set_xlabel('Risk Aversion', fontsize=self.labelsize)
        ax.set_ylabel('Generator Disagreement Point ($d_G$)', fontsize=self.labelsize)
        ax2.set_ylabel('Load Disagreement Point ($d_L$)', fontsize=self.labelsize)
        ax.set_title(f'{self.cm_data.contract_type}: Disagreement Points', fontsize=self.titlesize)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Combine legends from both axes
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=self.legendsize)


        plt.tight_layout()
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Disagreement points plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

