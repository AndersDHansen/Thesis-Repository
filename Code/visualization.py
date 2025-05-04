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
        #metrics = ['StrikePrice', 'Utility_G6', 'Utility_L2', 'NashProduct']
        metrics = ['StrikePrice', 'ContractAmount']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.flatten()

        titles = ['Strike Price S', 'Contract Amount M']
        fig.suptitle('Risk New Objective Analysis', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            try:
                pivot_table = risk_sensitivity_df.pivot(
                    index='A_L2',
                    columns='A_G6',
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
                ax.set_xlabel('Risk Aversion Generator (A_G6)')
                ax.set_ylabel('Risk Aversion Load (A_L2)')
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

    def _plot_earnings_histograms(self, fixed_A_G6, A_L2_to_plot, filename=None,new_data = None):
        """
        Plots histograms of G6 and L2 net earnings for different risk aversion levels.
        """
        if new_data is not None:
            earnings_risk_sensitivity_df = new_data
        else:
            earnings_risk_sensitivity_df = self.earnings_risk_sensitivity_df
        filtered_results = pd.concat([
            df[(df['A_G6'] == fixed_A_G6) & (df['A_L2'].isin(A_L2_to_plot)) & 
               (~df['Revenue_G6'].isna()) & (~df['Revenue_L2'].isna())]
            for df in earnings_risk_sensitivity_df
            if isinstance(df, pd.DataFrame) and not df.empty
        ], ignore_index=True)

        if filtered_results.empty:
            print("No valid results to plot.")
            return
        
        all_g6_values = np.concatenate([filtered_results['Revenue_G6'].values,self.cm_data.net_earnings_no_contract_G6_df.sum().values])
        all_l2_values = np.concatenate([filtered_results['Revenue_L2'].values,self.cm_data.net_earnings_no_contract_L2])
        

        # Create uniform bins based on global min and max
        bins = 19
        min_val_g6 = min(all_g6_values) / 1e5
        max_val_g6 = max(all_g6_values) / 1e5
        bin_edges_g6 = np.linspace(min_val_g6, max_val_g6, bins + 1)

        min_val_l2 = min(all_l2_values) / 1e5
        max_val_l2 = max(all_l2_values) / 1e5
        bin_edges_l2 = np.linspace(min_val_l2, max_val_l2, bins + 1)


        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot G6 Revenue Histogram
        ax_g6 = axes[0]
        ax_l2 = axes[1]

        # Get color cycle before the loops
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        # Plot G6 Revenue Histogram
        for idx, a_l2 in enumerate(A_L2_to_plot):
            g6_values = filtered_results[filtered_results['A_L2'] == a_l2]['Revenue_G6'].values

            # Expected G6 Revenue
            G_6_expected = g6_values.mean()
            # Calculate CVaR values 
            cvar_g6 = calculate_cvar_left(g6_values, self.cm_data.alpha)
            utility_g6 = (1-a_l2)*G_6_expected + a_l2*cvar_g6

            if len(g6_values) > 0:
                current_color = colors[idx % len(colors)]  # Cycle through colors
                print(f"\nPlotting histogram for A_L2={a_l2}")
                print(f"Values range: {g6_values.min() / 1e5} to {g6_values.max() / 1e5}")
                ax_g6.hist(
                    g6_values / 1e5,
                    bins=bin_edges_g6,
                    alpha=0.6,
                    label=f'A_L={a_l2}',
                    color=current_color,
                    density=False
                )
                ax_g6.axvline(utility_g6/1e5, linestyle="--", color=current_color, 
                              label=f"A_L={a_l2} - Utility: {utility_g6/1e5:.2f}")

            # Plot L2 Revenue Histogram with same color
            l2_values = filtered_results[filtered_results['A_L2'] == a_l2]['Revenue_L2'].values

            # Expected L2 Revenue
            l2_expected = g6_values.mean()
            # Calculate CVaR values 
            
            cvar_l2 = calculate_cvar_left(l2_values, self.cm_data.alpha)
            utility_l2 = (1-a_l2)*l2_expected + a_l2*cvar_l2
            if len(l2_values) > 0:
                ax_l2.hist(
                    l2_values / 1e5,
                    bins=bin_edges_l2,
                    alpha=0.6,
                    label=f'A_L={a_l2}',
                    color=current_color,
                    density=False
                )
                ax_l2.axvline(utility_l2/1e5, linestyle="--", color=current_color, 
                              label=f"A_L={a_l2} - Utility : {utility_l2/1e5:.2f}")
        
        ax_g6.hist(self.cm_data.net_earnings_no_contract_G6_df.sum() / 1e5,bins=bin_edges_g6,alpha=0.4,label=f'No Contract',density=False,color ='black')    
        #ax_g6.axvline(self.cm_data.net_earnings_no_contract_G6_df.sum().mean() / 1e5, linestyle="--",color ='black', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_G6_df.sum().mean() / 1e5:.2f}")
        ax_g6.set_title(f'Generator (G6) Revenue Distribution\n(A_G6 = {fixed_A_G6})')
        ax_g6.set_xlabel('Generator Revenue ($ x 10^5)')
        ax_g6.set_ylabel('Frequency')
        # Modify legend to have two columns with specific ordering
        handles, labels = ax_g6.get_legend_handles_labels()
        hist_handles = handles[::2]  # Get histogram handles
        line_handles = handles[1::2]  # Get vertical line handles
        hist_labels = labels[::2]    # Get histogram labels
        line_labels = labels[1::2]   # Get vertical line labels
        ax_g6.legend(hist_handles + line_handles, hist_labels + line_labels, 
                    ncol=2, loc='upper right', 
                    fontsize=10, bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax_g6.transAxes,
                    framealpha=0.8)
        ax_g6.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax_l2.hist(self.cm_data.net_earnings_no_contract_L2/ 1e5,bins=bin_edges_l2,alpha=0.4,label=f'No Contract',density=False, color ='black')
        #ax_l2.axvline(self.cm_data.net_earnings_no_contract_L2_df.sum().mean() / 1e5, linestyle="--",color ='black', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_L2_df.sum().mean() / 1e5:.2f}")    
        ax_l2.set_title(f'Load (L2) Revenue Distribution\n(A_G6 = {fixed_A_G6})')
        ax_l2.set_xlabel('Load Revenue ($ x 10^5)')
        ax_l2.set_ylabel('Frequency')
        # Apply same legend formatting to L2 plot
        handles, labels = ax_l2.get_legend_handles_labels()
        hist_handles = handles[::2]
        line_handles = handles[1::2]
        hist_labels = labels[::2]
        line_labels = labels[1::2]
        ax_l2.legend(hist_handles + line_handles, hist_labels + line_labels, 
                    ncol=2, loc='upper right',
                    fontsize=10, bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax_l2.transAxes,
                    framealpha=0.8)
        ax_l2.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

    def _plot_expected_versus_threatpoint(self,fixed_A_G6, A_L2_to_plot, filename=None):

        """
        Plots histograms of G6 and L2 net earnings for different risk aversion levels.
        """
        earnings_risk_sensitivity_df = self.earnings_risk_sensitivity_df
        
        filtered_results = pd.concat([
            df[(df['A_G6'] == fixed_A_G6) & (df['A_L2'].isin(A_L2_to_plot)) & 
               (~df['Revenue_G6'].isna()) & (~df['Revenue_L2'].isna())]
            for df in earnings_risk_sensitivity_df
            if isinstance(df, pd.DataFrame) and not df.empty
        ], ignore_index=True)

        if filtered_results.empty:
            print("No valid results to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot G6 Revenue Histogram
        ax_g6 = axes[0]
        ax_l2 = axes[1]

        # Get color cycle before the loops
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        zeta_l2_values = []


        # Calculate CVaR values (constant for all risk aversion values)
        cvar_g6_no_contract = calculate_cvar_left(self.cm_data.net_earnings_no_contract_G6, self.cm_data.alpha)
        cvar_l2_no_contract = calculate_cvar_left(self.cm_data.net_earnings_no_contract_L2, self.cm_data.alpha)
        mean_g6_contract = self.cm_data.net_earnings_no_contract_G6.mean()
        mean_l2_no_contract = self.cm_data.net_earnings_no_contract_L2.mean()

        # Calculate threat points for different risk aversion values
        for A in A_L2_to_plot:
            zeta_l2 = (1-A)*mean_l2_no_contract + A*cvar_l2_no_contract
            zeta_l2_values.append(zeta_l2)
        zeta_g6 = ((1-fixed_A_G6)*mean_g6_contract + fixed_A_G6*cvar_g6_no_contract)
        # Plot G6 Revenue Histogram
        for idx, a_l2 in enumerate(A_L2_to_plot):
            g6_values = filtered_results[filtered_results['A_L2'] == a_l2]['Revenue_G6'].values
            current_color = colors[idx % len(colors)]  # Cycle through colors

            if len(g6_values) > 0:
                expected_g6 = g6_values.mean()
                cvar_g6 = calculate_cvar_left(g6_values, self.cm_data.alpha)
                utility_g6 = (1-fixed_A_G6)*expected_g6 + fixed_A_G6*cvar_g6
                ax_g6.axvline(utility_g6/1e5, linestyle="-", color=current_color, 
                              label=f"A_L={a_l2} - Utility: {utility_g6/1e5:.2f}")
            # Plot L2 Revenue Histogram with same color
            l2_values = filtered_results[filtered_results['A_L2'] == a_l2]['Revenue_L2'].values
            if len(l2_values) > 0:
                expected_l2 = l2_values.mean()
                cvar_l2 = calculate_cvar_left(l2_values, self.cm_data.alpha)
                utility_l2 = (1-a_l2)*expected_l2 + a_l2*cvar_l2
                ax_l2.axvline(utility_l2/1e5, linestyle="-", color=current_color, 
                              label=f"A_L={a_l2} - Utility: {utility_l2/1e5:.2f}")
                ax_l2.axvline(zeta_l2_values[idx]/1e5, linestyle="--", color=current_color, label=f"A_L={a_l2:.2f} - Threat= {zeta_l2_values[idx]/1e5:.2f} ")

        
        #G6 Subplot configuration
        ax_g6.axvline(zeta_g6/1e5, linestyle="--", color='black', label=f"Threat Point: {zeta_g6/1e5:.2f}")
        ax_g6.set_title(f'Generator (G6) Threatpoint\n(A_G6 = {fixed_A_G6}) vs. L2 Risk Aversion')
        ax_g6.set_xlabel('Generator Revenue ($ x 10^5)')
        # Modify legend to have two columns with specific ordering
        handles, labels = ax_g6.get_legend_handles_labels()
        hist_handles = handles[::2]  # Get histogram handles
        line_handles = handles[1::2]  # Get vertical line handles
        hist_labels = labels[::2]    # Get histogram labels
        line_labels = labels[1::2]   # Get vertical line labels
        ax_g6.legend(hist_handles + line_handles, hist_labels + line_labels, 
                    ncol=2, loc='upper right', 
                    fontsize=10, bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax_g6.transAxes,
                    framealpha=0.8)
        ax_g6.grid(True, axis='y', linestyle='--', alpha=0.7)

        #ax_l2.axvline(self.cm_data.net_earnings_no_contract_L2_df.sum().mean() / 1e5, linestyle="--",color ='black', label=f"No Contract - Average Earnings: {self.cm_data.net_earnings_no_contract_L2_df.sum().mean() / 1e5:.2f}")   
        #L2 Subplot configuration
        ax_l2.set_title(f'Load (L2) Threatpoints vs \n(A_G6 = {fixed_A_G6})')
        ax_l2.set_xlabel('Load Revenue ($ x 10^5)')
        # Apply same legend formatting to L2 plot
        handles, labels = ax_l2.get_legend_handles_labels()
        hist_handles = handles[::2]
        line_handles = handles[1::2]
        hist_labels = labels[::2]
        line_labels = labels[1::2]
        ax_l2.legend(hist_handles + line_handles, hist_labels + line_labels, 
                    ncol=2, loc='upper right',
                    fontsize=10, bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax_l2.transAxes,
                    framealpha=0.8)
        ax_l2.grid(True, axis='y', linestyle='--', alpha=0.7)

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
        ax_g6 = axes[0]
        ax_l2 = axes[1]

        # Scale values to 10^5 for better readability
        g6_values = self.cm_data.net_earnings_no_contract_G6_df.sum().values / 1e5
        l2_values = self.cm_data.net_earnings_no_contract_L2.values / 1e5

        # Create histogram bins
        bins_g6 = np.linspace(min(g6_values), max(g6_values), 19)
        bins_l2 = np.linspace(min(l2_values), max(l2_values), 19)

        # Plot histograms
   


        # Create second figure for threat point evolution

        # Calculate threat points for different risk aversion values
        risk_aversion_values = np.linspace(0, 1, 4)
        zeta_g6_values = []
        zeta_l2_values = []
        
        # Calculate CVaR values (constant for all risk aversion values)
        cvar_g6 = calculate_cvar_left(self.cm_data.net_earnings_no_contract_G6, self.cm_data.alpha)
        cvar_l2 = calculate_cvar_left(self.cm_data.net_earnings_no_contract_L2, self.cm_data.alpha)
        mean_g6 = self.cm_data.net_earnings_no_contract_G6.mean()
        mean_l2 = self.cm_data.net_earnings_no_contract_L2.mean()

        # Calculate threat points for different risk aversion values
        for A in risk_aversion_values:
            zeta_g6 = (1-A)*mean_g6 + A*cvar_g6
            zeta_l2 = (1-A)*mean_l2 + A*cvar_l2
            zeta_g6_values.append(zeta_g6/1e5)
            zeta_l2_values.append(zeta_l2/1e5)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        ax_g6.hist(g6_values, bins=bins_g6, alpha=0.6, color='blue', density=False)
        for i,zeta in enumerate(zeta_g6_values):
            current_color = colors[i % len(colors)]  # Cycle through colors
            ax_g6.axvline(zeta, linestyle="--", color=current_color, label=f'A_G6={risk_aversion_values[i]:.2f} - Threat Point: {zeta:.2f}')
        ax_g6.set_title('Generator (G6) No-Contract Revenue Distribution')
        ax_g6.set_xlabel('Generator Revenue ($ x 10^5)')
        ax_g6.set_ylabel('Frequency')
        ax_g6.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax_g6.legend()

        ax_l2.hist(l2_values, bins=bins_l2, alpha=0.6, color='green', density=False)
        for i,zeta in enumerate(zeta_l2_values):
            current_color = colors[i % len(colors)]
            ax_l2.axvline(zeta, linestyle="--", color=current_color, label=f'A_L2={risk_aversion_values[i]:.2f} - Threat Point: {zeta:.2f}')
        ax_l2.set_title('Load (L2) No-Contract Revenue Distribution')
        ax_l2.set_xlabel('Load Revenue ($ x 10^5)')
        ax_l2.set_ylabel('Frequency')
        ax_l2.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax_l2.legend()

        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            print(f"Plot saved to {filepath}")
            fig1.savefig(filepath.replace('.', '_hist.'))
            print(f"Plots saved to {filepath}")
            plt.close(fig1)
        else:
            plt.show()

