"""
Visualization module for power system contract negotiation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from utils import calculate_cvar_left

class Plotting_Class:
    """
    Handles plotting of results from the power system contract negotiation simulation.
    """
    def __init__(self, contract_model_data, risk_sensitivity_df, sensitivity_earnings_df, bias_sensitivity_df, styles=None):
        self.cm_data = contract_model_data
        self.risk_sensitivity_df = risk_sensitivity_df
        self.earnings_risk_sensitivity_df = sensitivity_earnings_df
        self.bias_sensitivity_df = bias_sensitivity_df
        self.styles = styles if styles else {}

        self.plots_dir = os.path.join(os.path.dirname(__file__), 'Plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def _plot_sensitivity_results(self,filename=None,new_data = None):
        """
        Plots the sensitivity analysis results using heatmaps.
        """
        if self.risk_sensitivity_df.empty:
            print("No results to plot.")
            return
        
        if new_data is not None:
            risk_sensitivity_df = new_data
        else:
            risk_sensitivity_df = self.risk_sensitivity_df
            
        # Metrics to plot
        #metrics = ['StrikePrice', 'Utility_G', 'Utility_L', 'NashProduct']
        metrics = ['StrikePrice', 'ContractAmount']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.flatten()

        titles = ['Strike Price S', 'Contract Amount M']
        fig.suptitle('Risk New Objective Analysis', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            try:
                pivot_table = risk_sensitivity_df.pivot(
                    index='A_L',
                    columns='A_G',
                    values=metric
                )
                fmt = ".2f" if metric == "StrikePrice" else ".1f"
                sns.heatmap(
                    pivot_table,
                    ax=ax,
                    annot=True,
                    fmt=fmt,
                    cmap="viridis",
                    cbar=True
                )
                ax.set_title(titles[i])
                ax.set_xlabel('Risk Aversion Generator (A_G)')
                ax.set_ylabel('Risk Aversion Load (A_L)')
                ax.invert_yaxis()
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

    def _plot_earnings_histograms(self, fixed_A_G, A_L_to_plot, filename=None,new_data = None):
        """
        Plots histograms of G and L net earnings for different risk aversion levels.
        """
        if new_data is not None:
            earnings_risk_sensitivity_df = new_data
        else:
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
        
        all_G_values = np.concatenate([filtered_results['Revenue_G'].values,self.cm_data.net_earnings_no_contract_true_G.values])
        all_L_values = np.concatenate([filtered_results['Revenue_L'].values,self.cm_data.net_earnings_no_contract_true_L.values])
        

        # Create uniform bins based on global min and max
        bins = 19
        min_val_G = min(all_G_values) / 1e5
        max_val_G = max(all_G_values) / 1e5
        bin_edges_G = np.linspace(min_val_G, max_val_G, bins + 1)

        min_val_L = min(all_L_values) / 1e5
        max_val_L = max(all_L_values) / 1e5
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

            # Expected G Revenue
            G_6_expected = G_values.mean()
            # Calculate CVaR values 
            cvar_G = calculate_cvar_left(G_values, self.cm_data.alpha)
            utility_G = (1-a_L)*G_6_expected + a_L*cvar_G

            if len(G_values) > 0:
                current_color = colors[idx % len(colors)]  # Cycle through colors
                print(f"\nPlotting histogram for A_L={a_L}")
                print(f"Values range: {G_values.min() / 1e5} to {G_values.max() / 1e5}")
                ax_G.hist(
                    G_values / 1e5,
                    bins=bin_edges_G,
                    alpha=0.6,
                    label=f'A_L={a_L}',
                    color=current_color,
                    density=False
                )
                ax_G.axvline(utility_G/1e5, linestyle="--", color=current_color, 
                              label=f"A_L={a_L} - Utility: {utility_G/1e5:.2f}")

            # Plot L Revenue Histogram with same color
            L_values = filtered_results[filtered_results['A_L'] == a_L]['Revenue_L'].values

            # Expected L Revenue
            L_expected = L_values.mean()
            # Calculate CVaR values 
            
            cvar_L = calculate_cvar_left(L_values, self.cm_data.alpha)
            utility_L = (1-a_L)*L_expected + a_L*cvar_L
            if len(L_values) > 0:
                ax_L.hist(
                    L_values / 1e5,
                    bins=bin_edges_L,
                    alpha=0.6,
                    label=f'A_L={a_L}',
                    color=current_color,
                    density=False
                )
                ax_L.axvline(utility_L/1e5, linestyle="--", color=current_color, 
                              label=f"A_L={a_L} - Utility : {utility_L/1e5:.2f}")
        
        ax_G.hist(self.cm_data.net_earnings_no_contract_G_df.sum() / 1e5,bins=bin_edges_G,alpha=0.4,label=f'No Contract',density=False,color ='black')    
        #ax_G.axvline(self.cm_data.net_earnings_no_contract_G_df.sum().mean() / 1e5, linestyle="--",color ='black', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_G_df.sum().mean() / 1e5:.2f}")
        ax_G.set_title(f'Generator (G) Revenue Distribution\n(A_G = {fixed_A_G})')
        ax_G.set_xlabel('Generator Revenue ($ x 10^5)')
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

        ax_L.hist(self.cm_data.net_earnings_no_contract_true_L/ 1e5,bins=bin_edges_L,alpha=0.4,label=f'No Contract',density=False, color ='black')
        #ax_L.axvline(self.cm_data.net_earnings_no_contract_L_df.sum().mean() / 1e5, linestyle="--",color ='black', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_L_df.sum().mean() / 1e5:.2f}")    
        ax_L.set_title(f'Load (L) Revenue Distribution\n(A_G = {fixed_A_G})')
        ax_L.set_xlabel('Load Revenue ($ x 10^5)')
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

        plt.tight_layout()
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
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


    def _plot_bias_sensitivity(self, filename=None):
        
        """Plots the price bias sensitivity analysis results using heatmaps."""
        if self.bias_sensitivity_df.empty:
            print("No results to plot.")
            return
            
        metrics = ['StrikePrice', 'ContractAmount']
        titles = ['Strike Price S', 'Contract Amount M']
        formats = ['.3f', '.1f']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i, (metric, title, fmt) in enumerate(zip(metrics, titles, formats)):
            ax = axes[i]
            try:
                pivot_table = self.bias_sensitivity_df.pivot(
                    index='KG_Factor',
                    columns='KL_Factor',
                    values=metric
                )
                sns.heatmap(
                    pivot_table,
                    ax=ax,
                    annot=True,
                    fmt=fmt,
                    cmap="viridis",
                    cbar=True
                )
                ax.set_title(title)
                ax.set_xlabel('Load Price Bias Factor (KL)')
                ax.set_ylabel('Generator Price Bias Factor (KG)')
                ax.invert_yaxis()
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

