"""
Visualization module for power system contract negotiation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from utils import calculate_cvar_left
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

cmap_red_green=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
# Create a custom colormap that transitions from white to very light gray



class Plotting_Class:
    """
    Handles plotting of results from the power system contract negotiation simulation.
    """
    def __init__(self, contract_model_data, risk_sensitivity_df, sensitivity_earnings_df, 
                 bias_sensitivity_df,capture_rate_sensitivity_df, production_sensitivity_df,
                 boundary_results_df, negotiation_sensitivity_df,negotiation_earnings_df,
                 alpha_sensitivity_df,alpha_earnings_df,
                 styles=None):
        self.cm_data = contract_model_data
        self.risk_sensitivity_df = risk_sensitivity_df
        self.earnings_risk_sensitivity_df = sensitivity_earnings_df
        self.bias_sensitivity_df = bias_sensitivity_df
        self.production_sensitivity_df = production_sensitivity_df
        self.capture_rate_sensitivity_results = capture_rate_sensitivity_df
        self.boundary_results = boundary_results_df
        self.negotiation_sensitivity_df = negotiation_sensitivity_df
        self.negotiation_earnings_df = negotiation_earnings_df
        self.alpha_sensitivity_df = alpha_sensitivity_df
        self.alpha_earnings_df = alpha_earnings_df
        self.styles = styles if styles else {}
        self.white_cmap = LinearSegmentedColormap.from_list("white_cmap", ["white", "#F8F8F8"])


        self.plots_dir = os.path.join(os.path.dirname(__file__), 'Plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def _plot_sensitivity_results_heatmap(self,sensitivity_type,filename=None):

        """
        Generalized function to plot sensitivity analysis results using heatmaps.
        
        Parameters:
        -----------
        sensitivity_type : str
            Type of sensitivity analysis ('risk', 'bias', 'negotiation', 'alpha')
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
                'xlabel': 'Risk Aversion $A_G$',
                'ylabel': 'Risk Aversion $A_L$'
            },
            'bias': {
                'df': self.bias_sensitivity_df,
                'title': 'Bias Sensitivity on Strike Price and Contract Amount',
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
        results['ContractAmount'] = results['ContractAmount'].round(2)
        
        is_pap = self.cm_data.contract_type == "PAP"
        has_gamma = 'Gamma' in results.columns
        
        # Metrics to plot
        metrics = ['StrikePrice', 'ContractAmount']
        if is_pap and has_gamma:
            units = ['€/MWh', '%,MW']
        else:
            units = ['€/MWh', 'MWh']
        titles = ['Strike Price', 'Contract Amount']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.flatten()
        
        fig.suptitle(f'{self.cm_data.contract_type}: {cfg["title"]}', fontsize=16)
        
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
                    annot=False,
                    cmap="cividis",
                    cbar=False,
                    linewidths=0.5,
                    linecolor='gray'
                )
                
                # Add custom annotations with units
                for i_idx, row_idx in enumerate(pivot_table.index):
                    for j_idx, col_idx in enumerate(pivot_table.columns):
                        val = pivot_table.iloc[i_idx, j_idx]
                        if not np.isnan(val):
                            # Get background color for text color determination
                            bg_color = plt.cm.get_cmap("cividis")(
                                plt.Normalize()(pivot_table.values)
                            )[i_idx, j_idx, :3]
                            
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
                                    ha='center', va='center', color=text_color, fontsize=10)
                                    
                                ax.text(j_idx + 0.5, i_idx + 0.63, 
                                    f"{self.cm_data.generator_contract_capacity*gamma_val:.2f} MW",
                                    ha='center', va='center', color=text_color, fontsize=10)
                        
                            else:
                                # Format the main value with units
                                text = f"{val:.2f} {unit}"
                                
                                # Add the main value annotation
                                ax.text(j_idx + 0.5, i_idx + 0.5, text,
                                    ha='center', va='center', color=text_color, fontsize=10)
                            
                            
                           
                
                ax.set_title(f"{title} ({unit})")
                ax.set_xlabel(cfg['xlabel'])
                ax.set_ylabel(cfg['ylabel'])
                
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
        For negotiation: uses Beta_L as parameter.
        For alpha: uses 'alpha' as parameter (single value, not Alpha_L/Alpha_G).
        """
        # Configuration dictionary for different sensitivity types
        config = {
            'negotiation': {
                'df': self.negotiation_sensitivity_df,
                'param_col': 'Beta_L',
                'xlabel': 'Load Negotiation Power $\\beta_L$',
                'title': 'Negotiation Power Sensitivity on Strike Price and Contract Amount'
            },
            'alpha': {
                'df': self.alpha_sensitivity_df,
                'param_col': 'alpha',  # Use the single alpha column
                'xlabel': 'CVaR Alpha $\\alpha$',
                'title': 'CVaR Alpha Sensitivity on Strike Price and Contract Amount'
            }
        }

        cfg = config[sensitivity_type]
        results_df = cfg['df']

        results = results_df.copy()
        results['ContractAmount'] = results['ContractAmount'].round(2)

        is_pap = self.cm_data.contract_type == "PAP"
        has_gamma = 'Gamma' in results.columns

        # Sort by the parameter column
        results_sorted = results.sort_values(cfg['param_col'])
        param_values = results_sorted[cfg['param_col']].values

        # Metrics to plot
        metrics = ['StrikePrice', 'ContractAmount']
        if is_pap and has_gamma:
            units = ['€/MWh', 'MW']
        else:
            units = ['€/MWh', 'MWh']
        titles = ['Strike Price', 'Contract Amount']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i, (metric, unit, title) in enumerate(zip(metrics, units, titles)):
            ax = axes[i]
            sns.lineplot(
            x=param_values,
            y=results_sorted[metric].values,
            marker='o',
            linewidth=2,
            markersize=8,
            ax=ax
            )

            last_val = None
            for j, (x, y) in enumerate(zip(param_values, results_sorted[metric].values)):
                if not np.isnan(y) and (last_val is None or not np.isclose(y, last_val)):
                    ax.annotate(f'{y:.2f} {unit}',
                                (x, y),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center',
                                fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    last_val = y
                # Add gamma annotations if applicable (PAP contract, ContractAmount)
                
            if is_pap and has_gamma and metric == 'ContractAmount':
                ax2 = ax.twinx()
                gamma_values = results_sorted['Gamma'].values * 100  # Convert to percentage
                sns.lineplot(
                x=param_values,
                y=gamma_values,
                marker='s',
                linewidth=1,
                markersize=6,
                color='red',
                alpha=0.7,
                linestyle='--',
                ax=ax2
                )
                ax2.set_ylabel('Gamma (%)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                """
                last_gamma = None
                for j, (x, gamma) in enumerate(zip(param_values, gamma_values)):
                    if not np.isnan(gamma):
                        if j == 0 or j == len(param_values) - 1 or (last_gamma is None or not np.isclose(gamma, last_gamma)):
                            ax2.annotate(f'γ={gamma:.1f}%',
                                        (x, gamma),
                                        textcoords="offset points",
                                        xytext=(0, -15),
                                        ha='center',
                                        fontsize=8,
                                        color='red',
                                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                        last_gamma = gamma
                """
                # Fix y-axis if all gamma values are (almost) 100%
                if np.allclose(gamma_values, 100, atol=1e-2):
                    ax2.set_ylim(99, 101)
          
        
            ax.set_xlabel(cfg['xlabel'])
            ax.set_ylabel(f'{title} ({unit})')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{self.cm_data.contract_type}: {cfg["title"]}', fontsize=16)
        plt.tight_layout()

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
            cvar_G = calculate_cvar_left(G_values, self.cm_data.alpha)
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
            
            cvar_L = calculate_cvar_left(L_values, self.cm_data.alpha)
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
        ax_G.set_title(f'Generator (G) Revenue Distribution')
        ax_G.set_xlabel('Generator Revenue (Mio EUR)')
        ax_G.set_ylabel('Frequency')
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
        ax_L.set_title(f'Load (L) Revenue Distribution')
        ax_L.set_xlabel('Load Revenue (Mio EUR)')
        ax_L.set_ylabel('Frequency')
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

        #Add suptitle 
        fig.suptitle(f'{self.cm_data.contract_type}: Expected Revenue ($A_G$ = {fixed_A_G})', fontsize=16)

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
        
            ax_g.set_title('Generator Revenue Distribution')
            ax_g.set_xlabel('Revenue (Mio EUR)')
            ax_g.set_ylabel('Frequency')
            ax_g.legend()
            ax_g.grid(True, axis='y', linestyle='--', alpha=0.7)
        
            ax_l.set_title('Load Revenue Distribution')
            ax_l.set_xlabel('Revenue (Mio EUR)')
            ax_l.set_ylabel('Frequency')
            ax_l.legend()
            ax_l.grid(True, axis='y', linestyle='--', alpha=0.7)
        
            fig.suptitle(f"{self.cm_data.contract_type}: Earnings Distribution by Alpha, $A_G$ = {self.cm_data.A_G}, $A_L$ = {self.cm_data.A_L}", fontsize=15)
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
        cvar_G_no_contract = calculate_cvar_left(self.cm_data.net_earnings_no_contract_G, self.cm_data.alpha)
        cvar_L_no_contract = calculate_cvar_left(self.cm_data.net_earnings_no_contract_L, self.cm_data.alpha)
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
                cvar_G = calculate_cvar_left(G_values, self.cm_data.alpha)
                utility_G = (1-fixed_A_G)*expected_G + fixed_A_G*cvar_G
                ax_G.axvline(utility_G/1e5, linestyle="-", color=current_color, 
                              label=f"A_L={a_L} - Utility: {utility_G/1e5:.2f}")
            # Plot L Revenue Histogram with same color
            L_values = filtered_results[filtered_results['A_L'] == a_L]['Revenue_L'].values
            if len(L_values) > 0:
                expected_L = L_values.mean()
                cvar_L = calculate_cvar_left(L_values, self.cm_data.alpha)
                utility_L = (1-a_L)*expected_L + a_L*cvar_L
                ax_L.axvline(utility_L/1e5, linestyle="-", color=current_color, 
                              label=f"A_L={a_L} - Utility: {utility_L/1e5:.2f}")
                ax_L.axvline(zeta_L_values[idx]/1e5, linestyle="--", color=current_color, label=f"A_L={a_L:.2f} - Threat= {zeta_L_values[idx]/1e5:.2f} ")

        
        #G Subplot configuration
        ax_G.axvline(zeta_G/1e5, linestyle="--", color='black', label=f"Threat Point: {zeta_G/1e5:.2f}")
        ax_G.set_title(f'Generator (G) Threatpoint\n(A_G = {fixed_A_G}) vs. L Risk Aversion')
        ax_G.set_xlabel('Generator Revenue ($ x 10^5)')
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
        ax_L.set_title(f'Load (L) Threatpoints vs \n(A_G = {fixed_A_G})')
        ax_L.set_xlabel('Load Revenue ($ x 10^5)')
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
        cvar_G = calculate_cvar_left(self.cm_data.net_earnings_no_contract_G, self.cm_data.alpha)
        cvar_L = calculate_cvar_left(self.cm_data.net_earnings_no_contract_L, self.cm_data.alpha)
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
        ax_G.set_title('Generator (G) No-Contract Revenue Distribution')
        ax_G.set_xlabel('Generator Revenue ($ x 10^5)')
        ax_G.set_ylabel('Frequency')
        ax_G.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax_G.legend()

        ax_L.hist(L_values, bins=bins_L, alpha=0.6, color='green', density=False)
        for i,zeta in enumerate(zeta_L_values):
            current_color = colors[i % len(colors)]
            ax_L.axvline(zeta, linestyle="--", color=current_color, label=f'A_L={risk_aversion_values[i]:.2f} - Threat Point: {zeta:.2f}')
        ax_L.set_title('Load (L) No-Contract Revenue Distribution')
        ax_L.set_xlabel('Load Revenue ($ x 10^5)')
        ax_L.set_ylabel('Frequency')
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
    def _plot_capture_rate_sensitivity(self, filename=None):
        """Plot capture rate sensitivity analysis results with improved visualizations."""
        # Check if capture rate sensitivity results are available
        if not hasattr(self, 'capture_rate_sensitivity_results') or self.capture_rate_sensitivity_results is None:
            print("No capture rate sensitivity results available to plot.")
            return
        
        df = self.capture_rate_sensitivity_results
        
        # Create figure with subplots - use a 2x3 grid for more detailed analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.cm_data.contract_type}: Contract Parameters Sensitivity to Capture Rate $A_G$ = {self.cm_data.A_G}, $A_L$  = {self.cm_data.A_L}', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        # Plot strike price vs capture rate factor
        x_column = 'CaptureRate'
        
        axes[0].plot(df[x_column], df['StrikePrice'], marker='o', linestyle='-')
        axes[0].set_xlabel('Capture Rate (%)')
        axes[0].set_ylabel('Strike Price (EUR/MWh)')
        axes[0].set_title('Strike Price vs Capture Rate')
        axes[0].grid(True)
        
        # Plot contract amount vs capture rate
        axes[1].plot(df[x_column], df['ContractAmount'], marker='o', linestyle='-')
        axes[1].set_xlabel('Capture Rate (%)')
        axes[1].set_ylabel('Contract Amount (MWh)')
        axes[1].set_title('Contract Amount vs Capture Rate')
        axes[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # Format to 1 decimal place
        if self.cm_data.contract_type == "PAP":
            axes[1].set_ylim(25, 40)  # Adjust these values based on your specific data range
        axes[1].grid(True)
        
        # Plot utility G vs capture rate
        axes[2].plot(df[x_column], df['Utility_G'], marker='o', linestyle='-')
        axes[2].set_xlabel('Capture Rate (%)')
        axes[2].set_ylabel('Utility Generator')
        axes[2].set_title('Generator Utility vs Capture Rate')
        axes[2].grid(True)
        
        # Plot utility L vs capture rate
        axes[3].plot(df[x_column], df['Utility_L'], marker='o', linestyle='-')
        axes[3].set_xlabel('Capture Rate (%)')
        axes[3].set_ylabel('Utility Load')
        axes[3].set_title('Load Utility vs Capture Rate')
        axes[3].grid(True)
        
        # Plot threat points vs capture rate
        axes[4].plot(df[x_column], df['ThreatPoint_G'], marker='o', linestyle='-', label='Generator')
        axes[4].set_xlabel('Capture Rate (%)')
        axes[4].set_ylabel('Threat Point Value')
        axes[4].set_title('Threat Points vs Capture Rate')
        axes[4].grid(True)

        # Plot Nash Product vs capture rate
        axes[5].plot(df[x_column], df['Nash_Product'], marker='o', linestyle='-')
        axes[5].set_xlabel('Production Factor (%)')
        axes[5].set_ylabel('Nash Product')
        axes[5].set_title('Nash Product vs Production Factor')
        axes[5].grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if filename:
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved capture rate sensitivity plot to {filename}")
        else:
            plt.show()
   
    def _plot_production_sensitivity(self, filename=None):
        """Plot production sensitivity analysis results with improved visualizations."""
        # Check if production sensitivity results are available
        if not hasattr(self, 'production_sensitivity_df') or self.production_sensitivity_df is None:
            print("No production sensitivity results available to plot.")
            return
        
        df = self.production_sensitivity_df
        
        # Create figure with subplots - use a 2x3 grid for more detailed analysis
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))
        fig.suptitle(f'{self.cm_data.contract_type}: Contract Parameters Sensitivity to Production $A_G$ = {self.cm_data.A_G}, $A_L$  = {self.cm_data.A_L}', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot strike price vs production factor
        x_column = 'ProductionFactor' 
        
        axes[0].plot(df[x_column], df['StrikePrice'], marker='o', linestyle='-')
        axes[0].set_xlabel('Production Factor (%)')
        axes[0].set_ylabel('Strike Price (EUR/MWh)')
        axes[0].set_title('Strike Price vs Production Factor')
        axes[0].grid(True)
        
        # Plot contract amount vs production factor
        if self.cm_data.contract_type == "PAP":
            axes[1].set_ylim(25, 40)  # Adjust these values based on your specific data range
        axes[1].plot(df[x_column], df['ContractAmount'], marker='o', linestyle='-')
        axes[1].set_xlabel('Production Factor (%)')
        axes[1].set_ylabel('Contract Amount (MWh)')
        axes[1].set_title('Contract Amount vs Production Factor')
        axes[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # Format to 1 decimal place
        axes[1].grid(True)
        
        # Plot utility G vs production factor
        axes[2].plot(df[x_column], df['Utility_G'], marker='o', linestyle='-')
        axes[2].set_xlabel('Production Factor (%)')
        axes[2].set_ylabel('Utility Generator')
        axes[2].set_title('Generator Utility vs Production Factor')
        axes[2].grid(True)

        axes[3].plot(df[x_column], df['Utility_L'], marker='o', linestyle='-')
        axes[3].set_xlabel('Production Factor (%)')
        axes[3].set_ylabel('Utility Load')
        axes[3].set_title('Load Utility vs Production Factor')
        axes[3].grid(True)
        # Plot threat points vs production factor
        axes[4].plot(df[x_column], df['ThreatPoint_G'], marker='o', linestyle='-', label='Generator')
        axes[4].set_xlabel('Production Factor (%)')
        axes[4].set_ylabel('Threat Point Value')
        axes[4].set_title('Threat Points vs Production Factor')
        axes[4].grid(True)
        
        # Plot Nash Product vs production factor
        axes[5].plot(df[x_column], df['Nash_Product'], marker='o', linestyle='-')
        axes[5].set_xlabel('Production Factor (%)')
        axes[5].set_ylabel('Nash Product')
        axes[5].set_title('Nash Product vs Production Factor')
        axes[5].grid(True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if filename:
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved capture rate sensitivity plot to {filename}")
        else:
            plt.show()

    def _plot_no_contract_boundaries(self, filename=None):
        """
        Plot the no-contract boundaries for different risk aversion scenarios.
        
        Parameters:
        -----------
        filename : str, optional
            Path to save the plot. If None, the plot will be displayed.
        """
        
        
        plt.figure(figsize=(10, 8))
        xlim = (-31, 31)
        ylim = (-31, 31)

        
        
        # Plot each scenario's boundary
        for result in self.boundary_results:
            scenario = result['scenario']

            boundary_points = np.array(result['boundary_points'])

            lowest_boundary = self._extract_lowest_boundary(boundary_points)
            n_space = np.linspace(xlim[0]*1e-2, xlim[1]*1e-2, 100)
            # Create and fit the regression model
            X = lowest_boundary[:, 0].reshape(-1, 1)  # X needs to be 2D for sklearn
            y = lowest_boundary[:, 1]
            model = LinearRegression().fit(X, y)

            # Generate points along the regression line
            X_pred = n_space.reshape(-1, 1)  # Reshape for prediction
            boundary = model.predict(X_pred)

            
            # Plot the boundary points
            sns.lineplot(x = n_space*100, y = boundary * 100, 
                     label=scenario['label'],
                     linestyle=scenario['linestyle'], 
                     linewidth=scenario['linewidth'],
                     color=scenario['color'])
            sns.scatterplot( x = lowest_boundary[:, 0] * 100, y = lowest_boundary[:, 1] * 100, s=90, alpha=0.5)
        # Add labels and formatting
        plt.xlabel('$K_L/E^P(\lambda_\Sigma)$ (%)')
        plt.ylabel('$K_G/E^P(\lambda_\Sigma)$ (%)')
        plt.title(f'{self.cm_data.contract_type}: No-Contract Boundaries for Different Risk Aversion Levels')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=0, color='k',linewidth=2)
        plt.axvline(x=0, color='k',linewidth=2)
        
        # Set the x and y axis limits similar to the figure
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0],  ylim[1])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        else:
            plt.show()

    def _plot_no_contract_boundaries_all(self, filename=None):
            # Helper method for the comprehensive visualization
        def _plot_boundary_on_axis( ax, result):
            """Plot a single boundary on a given axis."""
            scenario = result['scenario']
            lowest_boundary = self._extract_lowest_boundary(result['boundary_points'])
            
            if len(lowest_boundary) < 2:
                return
                
            # Fit linear regression
            X = lowest_boundary[:, 0].reshape(-1, 1)
            y = lowest_boundary[:, 1]
            model = LinearRegression().fit(X, y)
            
            # Generate points along the regression line
            xlim = [-31, 31]
            n_space = np.linspace(xlim[0]/100, xlim[1]/100, 100)
            X_pred = n_space.reshape(-1, 1)
            boundary = model.predict(X_pred)
            
            # Plot with styling from scenario
            sns.lineplot(x = n_space * 100, y =  boundary * 100, 
                    label=f"A_G={scenario['A_G']}, A_L={scenario['A_L']}",
                    linestyle=scenario['linestyle'], 
                    linewidth=scenario['linewidth'],
                    color=scenario['color'],
                    ax=ax,)
                
        
        
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
        ax1.set_title("Symmetrical Risk Aversion (A_G = A_L)")
        ax1.legend()
        
        # Plot 2: All asymmetrical cases
        for result in asym_scenarios:
            _plot_boundary_on_axis(ax2, result)
        ax2.set_title("Asymmetrical Risk Aversion (A_G ≠ A_L)")
        ax2.legend()
        
        # Plot 3: Fixed A_G, varying A_L
        fixed_ag_scenarios = [r for r in self.boundary_results if r['scenario']['A_G'] == 0.5]
        for result in fixed_ag_scenarios:
            _plot_boundary_on_axis(ax3, result)
        ax3.set_title("Fixed Generator Risk Aversion (A_G = 0.5)")
        ax3.legend()
        
        # Plot 4: Fixed A_L, varying A_G
        fixed_al_scenarios = [r for r in self.boundary_results if r['scenario']['A_L'] == 0.5]
        for result in fixed_al_scenarios:
            _plot_boundary_on_axis(ax4, result)
        ax4.set_title("Fixed Load Risk Aversion (A_L = 0.5)")
        ax4.legend()
        
        # Common formatting
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('$K_L/E^P(\lambda_\Sigma)$ (%)')
            ax.set_ylabel('$K_G/E^P(\lambda_\Sigma)$ (%)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.axhline(y=0, color='k',linewidth=2)
            ax.axvline(x=0, color='k',linewidth=2)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        fig.suptitle(f'{self.cm_data.contract_type}: No-Contract Boundaries', fontsize=16)
        
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
            'A_L': 'No Contract'  # Assign a string label for the category
        })

        capture_color = "#5E9AA2BC"  # Color for capture price

        # Combine the contract and no-contract data
        # Convert A_L to object type to allow mixing numbers and strings
        filtered_results['A_L'] = filtered_results['A_L'].astype(object)
        plot_data = pd.concat([no_contract_df,filtered_results], ignore_index=True)
            
        # Define the order for the x-axis categories
        A_L_to_plot = sorted(A_L_to_plot)
        plot_order = A_L_to_plot.insert(0,'No Contract')  # Add 'No Contract' at the beginning
        contract_mask = plot_data['A_L'] != 'No Contract'

        
        # Create figure with more space at the bottom for the table
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
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
            width=0.3,
            showfliers=True,
            palette="Set2"
        )
        ### add capture price

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
        ax_G.axhline(self.cm_data.Zeta_G , linestyle="--", color='black', label=f"Threat Point: {self.cm_data.Zeta_G :.2f}")
        ax_L.axhline(self.cm_data.Zeta_L , linestyle="--", color='black', label=f"Threat Point: {self.cm_data.Zeta_L :.2f}")
        # Add Expectation of no contract earnings
        ax_G.axhline(self.cm_data.net_earnings_no_contract_true_G.mean(), linestyle="--", color='grey', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_true_G.mean() :.2f} ")
        ax_L.axhline(self.cm_data.net_earnings_no_contract_true_L.mean(), linestyle="--", color='grey', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_true_L.mean() :.2f} ")

        # Set titles and labels
        ax_G.set_title(f'Generator Revenue',fontsize=14)
        ax_G.set_xlabel('Risk Aversion Level $(A_L)$',fontsize=13)
        ax_G.set_ylabel('Generator Revenue (Mio EUR)',fontsize=13)
        ax_G.grid(True, linestyle='--', alpha=0.9, axis='y')
        
        ax_L.set_title(f'Load Revenue',fontsize=14)
        ax_L.set_xlabel('Risk Aversion Level $(A_L)$', fontsize=13)
        ax_L.set_ylabel('Load Revenue (Mio EUR)',fontsize=13)
        ax_L.grid(True, linestyle='--', alpha=0.9, axis='y')


        cp_color_patch = mpatches.Patch(color=capture_color, label="Capture Price")
        ax_G.legend(handles=[cp_color_patch], loc='upper left', fontsize=11, frameon=True)
        ax_L.legend(handles=[cp_color_patch], loc='upper left', fontsize=11, frameon=True)

        plt.suptitle(f"{self.cm_data.contract_type}: Earnings Distribution by Risk Aversion, $A_G$ = {self.cm_data.A_G}", fontsize=16)
        
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
        unique_beta_g = earnings_df["Beta_G"].unique()

        # Get three evenly spaced positions
        positions = np.linspace(1, len(unique_beta_g)-1,2,dtype=int)
        selected_beta_g = np.round(unique_beta_g[positions],2)

        # Filter the DataFrame for these Beta_G values
        df_filtered = earnings_df[earnings_df["Beta_G"].isin(selected_beta_g)]

        AL_used = df_filtered['A_L'].unique()[0]  # Assuming A_L is constant for the filtered data
        AG_used = df_filtered['A_G'].unique()[0]  # Assuming A_G is constant for the filtered data
                

       # Prepare the no-contract data
        no_contract_g = self.cm_data.net_earnings_no_contract_true_G
        no_contract_l = self.cm_data.net_earnings_no_contract_true_L
        no_contract_df = pd.DataFrame({
            'Revenue_G': no_contract_g,
            'Revenue_L': no_contract_l,
            'Beta_G': 'No Contract'  # Assign a string label for the category
        })

        # Combine the contract and no-contract data
        # Convert A_L to object type to allow mixing numbers and strings
        df_filtered= df_filtered.astype(object)
        plot_data = pd.concat([no_contract_df,df_filtered], ignore_index=True)
            
        # Define the order for the x-axis categories
        nego_to_plot = sorted(selected_beta_g)
        plot_order = nego_to_plot.insert(0,'No Contract')  # Add 'No Contract' at the beginning
        
        
        # Create figure with more space at the bottom for the table
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        ax_G = axes[0]
        ax_L = axes[1]
        
        # 1. Add violin plots behind boxplots for Generator Revenue
        sns.violinplot(
            data=plot_data,
            x='Beta_G',
            y='Revenue_G',
            order=plot_order,
            ax=ax_G,
            hue="Beta_G", 
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
            x='Beta_G',
            y='Revenue_G',
            order=plot_order,
            ax=ax_G,
            hue="Beta_G",
            hue_order=plot_order,
            legend=False,
            width=0.5,
            showfliers=True,
            palette="Set2"
        )
       
        # 3. Add violin plots behind boxplots for Load Revenue
        sns.violinplot(
            data=plot_data,
            x='Beta_G',
            y='Revenue_L',
            order=plot_order,
            ax=ax_L,
            hue = 'Beta_G',
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
            x='Beta_G',
            y='Revenue_L',
            order=plot_order,
            ax=ax_L,
            hue= 'Beta_G',
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
        ax_G.set_title(f'Generator Revenue',fontsize=14)
        ax_G.set_xlabel('Negotiation Power $(Beta_G)$',fontsize=13)
        ax_G.set_ylabel('Generator Revenue (Mio EUR)',fontsize=13)
        ax_G.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        ax_L.set_title(f'Load Revenue',fontsize=14)
        ax_L.set_xlabel('Negotiation Power $(Beta_G)$', fontsize=13)
        ax_L.set_ylabel('Load Revenue (Mio EUR)',fontsize=13)
        ax_L.grid(True, linestyle='--', alpha=0.7, axis='y')

        plt.suptitle(f"{self.cm_data.contract_type}: Earnings Distribution by Negotiation Power with Risk Aversion $A_G$={AG_used}, $A_L$={AL_used}", fontsize=16)
        
        plt.tight_layout()

        # Save or show the figure
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()


    def _plot_parameter_sensitivity_spider(self, filename=None):
        """
        Create a spider/radar plot showing how different parameters affect contract outcomes.
        Compares the sensitivity of contract parameters to different input variables.
        """
        plt.figure(figsize=(12, 10))
        
       

        # Define the metrics we want to compare
        if self.cm_data.contract_type == "PAP":
            metrics = ['StrikePrice', 'Gamma', 'Utility_G', 'Utility_L', 'ThreatPoint_G', 'ThreatPoint_L', 'Nash_Product']
        else:
            metrics = ['StrikePrice', 'ContractAmount', 'Utility_G', 'Utility_L', 'ThreatPoint_G', 'ThreatPoint_L', 'Nash_Product']
        
        
        
        # Modify Price Sensitivity 
        bias_sensitivity_df = self.bias_sensitivity_df.copy()
        bias_sensitivity_df['KG_Factor'] = 1.0 +  self.bias_sensitivity_df['KG_Factor']          # 0.99, 1.00, 1.01
        bias_sensitivity_df['KL_Factor'] = 1.0 +  self.bias_sensitivity_df['KL_Factor']          # 0.99, 1.00, 1.01

        bias_KG = bias_sensitivity_df[bias_sensitivity_df['KL_Factor'] == 1.00]
        bias_KL = bias_sensitivity_df[bias_sensitivity_df['KG_Factor'] == 1.00]


        # Initialize elasticities dictionary
        elasticities = {}
        
        # Define the sensitivity analyses to process
        sensitivity_analyses = [
            {
                'name': 'Production',
                'df': self.production_sensitivity_df,
                'factor_col': 'ProductionFactor'
            },
            {
                'name': 'Capture Rate',
                'df': self.capture_rate_sensitivity_results,
                'factor_col': 'CaptureRate'
            },
            {
                'name': 'Negotation Power',
                'df': self.negotiation_sensitivity_df,
                'factor_col': 'Beta_G'
            },
            {
                'name': 'Price Sensitivity (KG Factor)',
                'df': bias_KG,
                'factor_col': 'KG_Factor'

            },
            {
                'name': 'Price Sensitivity (KL Factor)',
                'df': bias_KL,
                'factor_col': 'KL_Factor'

            },
          
        ]
        
        for analysis in sensitivity_analyses:
            df = analysis['df']        
            # Sort by factor column to ensure proper pair-wise comparison
            #df = df.sort_values(analysis['factor_col'])
            
             # Sort by factor column to ensure correct order for pct_change
            df = df.sort_values(analysis['factor_col']).reset_index(drop=True)
            
            # Calculate percentage change for metrics and the factor column
            pct_change_metrics  =df[metrics].pct_change()
            
            pct_change_factor = df[analysis['factor_col']].pct_change()
            
            # Calculate elasticity: (% change in metric) / (% change in factor)
            elasticity_df = pct_change_metrics.div(pct_change_factor, axis=0)
         
            # Store the results for this analysis
            elasticity_df.dropna(inplace=True)  # Drop rows with NaN values
            elasticity_df = elasticity_df.mean().round(4)#.to_frame().T  # Convert to DataFrame
            elasticities[analysis['name']] = elasticity_df.to_dict()
        
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
        plt.xticks(angles[:-1], categories, size=12)
        
        # Set y limits
        ax.set_ylim(-1, 1.5)
        
        # Add parameter elasticities
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (param, values) in enumerate(elasticities.items()):
            values_ordered = [values[metric] for metric in categories]
            values_ordered += values_ordered[:1]  # Close the loop
            
            ax.plot(angles, values_ordered, linewidth=2, linestyle='solid', 
                    label=param, color=colors[i % len(colors)])
            ax.fill(angles, values_ordered, alpha=0.1, color=colors[i % len(colors)])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(f'{self.cm_data.contract_type}: Parameter Sensitivity Comparison Elaticities by Factor', 
                size=15, y=1.1)
        
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

    def _plot_elasticity_tornado(self, metric='StrikePrice', filename=None):
        """
        Create a tornado plot showing the elasticity of a single metric
        with respect to different input factors.
        """
        # Prepare the same elasticities as in the spider plot
        if self.cm_data.contract_type == "PAP":
            metrics = ['StrikePrice', 'Gamma', 'Utility_G', 'Utility_L', 'ThreatPoint_G', 'ThreatPoint_L']
        else:
            metrics = ['StrikePrice', 'ContractAmount', 'Utility_G', 'Utility_L', 'ThreatPoint_G', 'ThreatPoint_L']

        # Prepare bias data as in spider plot
        bias_sensitivity_df = self.bias_sensitivity_df.copy()
        bias_sensitivity_df['KG_Factor'] = 1.0 + self.bias_sensitivity_df['KG_Factor']
        bias_sensitivity_df['KL_Factor'] = 1.0 + self.bias_sensitivity_df['KL_Factor']
        bias_KG = bias_sensitivity_df[bias_sensitivity_df['KL_Factor'] == 1.00]
        bias_KL = bias_sensitivity_df[bias_sensitivity_df['KG_Factor'] == 1.00]

        # Define the sensitivity analyses to process
        sensitivity_analyses = [
            {
                'name': 'Production',
                'df': self.production_sensitivity_df,
                'factor_col': 'ProductionFactor'
            },
            {
                'name': 'Capture Rate',
                'df': self.capture_rate_sensitivity_results,
                'factor_col': 'CaptureRate'
            },
            {
                'name': 'Negotiation Power',
                'df': self.negotiation_sensitivity_df,
                'factor_col': 'Beta_G'
            },
            {
                'name': 'Price Sensitivity (KG Factor)',
                'df': bias_KG,
                'factor_col': 'KG_Factor'
            },
            {
                'name': 'Price Sensitivity (KL Factor)',
                'df': bias_KL,
                'factor_col': 'KL_Factor'
            },
        ]

        elasticities = {}
        for analysis in sensitivity_analyses:
            df = analysis['df']
            df = df.sort_values(analysis['factor_col']).reset_index(drop=True)
            pct_change_metric = df[metric].pct_change()
            pct_change_factor = df[analysis['factor_col']].pct_change()
            elasticity = pct_change_metric.div(pct_change_factor, axis=0)
            elasticities[analysis['name']] = elasticity

        # Prepare data for plotting
        factors = list(elasticities.keys())
        values = [elasticities[f] for f in factors]
        sorted_indices = np.argsort(np.abs(values))[::-1]
        sorted_factors = [factors[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        # Plot tornado
        plt.figure(figsize=(8, 5))
        bars = plt.barh(sorted_factors, sorted_values, color='skyblue')
        plt.axvline(0, color='k', linewidth=1)
        plt.xlabel(f'Elasticity of {metric}')
        plt.title(f'Tornado Plot: Sensitivity of {metric}')
        plt.grid(axis='x', linestyle=':', alpha=0.7)

        # Annotate bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center', ha='left' if width > 0 else 'right', fontsize=10)

        plt.tight_layout()
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Tornado plot saved to {filepath}")
            plt.close()
        else:
            plt.show()