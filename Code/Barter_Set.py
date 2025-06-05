import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from utils import calculate_cvar_left


class Barter_Set:
    def __init__(self, data,results,scipy_results,old_obj_func):
        self.data = data
        self.results = results
        self.scipy_results = scipy_results
        self.old_obj_func = old_obj_func
        self.n = 5000  # Number of points for plotting
        self.BS_strikeprice_min = data.strikeprice_min * self.data.hours_in_year * 1e-3 # Converting EUR/MWH to Mio EUR/GWh
        self.BS_strikeprice_max = data.strikeprice_max * self.data.hours_in_year * 1e-3 # Converting EUR/MWH to Mio EUR/GWh
        self.BS_contract_amount_min = data.contract_amount_min * self.data.hours_in_year * 1e-3 # Converting MWh to GWh to yearly GWH mix capacity
        self.BS_contract_amount_max = data.contract_amount_max * self.data.hours_in_year * 1e-3 # Converting MWh to GWh to yearly GWH max capacity

    def cvar_derivative_wrt_M_L(self,M_base, earnings_base, price_matrix, strike, alpha):
        """
        Estimate gradient of CVaR with respect to contract volume M using finite differences.
        """
        epsilon = 1e-4
        M_plus = M_base + epsilon
        M_minus = M_base - epsilon

        rev_plus = ((M_plus ) * (( price_matrix) - (strike ))).sum(axis=0)
        rev_minus =((M_minus ) * (( price_matrix) - (strike ))).sum(axis=0)

        cvar_plus = calculate_cvar_left(earnings_base + rev_plus, alpha)
        cvar_minus = calculate_cvar_left(earnings_base + rev_minus, alpha)

        return (cvar_plus - cvar_minus) / (2 * epsilon)
    
    def cvar_derivative_wrt_M_G(self,M_base, earnings_base, price_matrix, strike, alpha):
        """
        Estimate gradient of CVaR with respect to contract volume M using finite differences.
        """
        epsilon = 1e-4
        M_plus = M_base + epsilon
        M_minus = M_base - epsilon
 

        rev_plus = ( M_plus*((strike) - price_matrix)).sum(axis=0)
        rev_minus = ( M_minus* ((strike)  - price_matrix  )).sum(axis=0)

        cvar_plus = calculate_cvar_left(earnings_base + rev_plus, alpha)
        cvar_minus = calculate_cvar_left(earnings_base + rev_minus, alpha)

        return (cvar_plus - cvar_minus) / (2 * epsilon)
    
    def expectation_derivative_wrt_M_L(self, M_base, earnings_base, price_matrix, strike):
        """
        Estimate the gradient of the expected value of earnings with respect to contract volume M.
        """
        epsilon = 1e-4
        M_plus = M_base + epsilon
        M_minus = M_base - epsilon
 # shape (n_months,)

        # Calculate revenue for M_plus and M_minus
        rev_plus = (M_plus  *  (price_matrix -strike )).sum(axis=0)
        rev_minus =(M_minus  * (price_matrix - strike )).sum(axis=0)

        # Calculate expected earnings for M_plus and M_minus
        expected_plus = (earnings_base + rev_plus).mean()
        expected_minus = (earnings_base + rev_minus).mean()

        # Return the finite difference approximation of the derivative
        return (expected_plus - expected_minus) / (2 * epsilon)

    def expectation_derivative_wrt_M_G(self, M_base, earnings_base, price_matrix, strike):
        """
        Estimate the gradient of the expected value of earnings with respect to contract volume M.
        """
        epsilon = 1e-4
        M_plus = M_base + epsilon
        M_minus = M_base - epsilon
        # Calculate revenue for M_plus and M_minus

        rev_plus = (M_plus * ( strike  - price_matrix )).sum(axis=0)
        rev_minus = (M_minus * ( strike -price_matrix  )).sum(axis=0)

        # Calculate expected earnings for M_plus and M_minus
        expected_plus = (earnings_base + rev_plus).mean()
        expected_minus = (earnings_base + rev_minus).mean()

        # Return the finite difference approximation of the derivative
        return (expected_plus - expected_minus) / (2 * epsilon)

    def Utility_G(self, strike,volume):

        """
        strike : Strike Price [float]
        volume : Contract volume [float]
        """

        rev_contract = volume  * (strike - self.data.price_G)
        rev_contract_total = rev_contract.sum(axis=0)
        Expected = self.data.net_earnings_no_contract_priceG_G 
        earnings = Expected + rev_contract_total
        CVaR_G = calculate_cvar_left(earnings, self.data.alpha)
        if self.old_obj_func == True:  # Use self.old_obj_func instead of parameter
            Utility = earnings.mean() + self.data.A_G * CVaR_G
        else:
            Utility = (1-self.data.A_G)*earnings.mean() + self.data.A_G * CVaR_G
        return Utility

    def Utility_L(self, strike,volume):
        

        rev_contract = volume  * ( self.data.price_L - strike )
        rev_contract_total = rev_contract.sum(axis=0)
        Expected = self.data.net_earnings_no_contract_priceL_L
        earnings = Expected + rev_contract_total
        CVaR_L = calculate_cvar_left(earnings, self.data.alpha)
        
        if self.old_obj_func == True:  # Use self.old_obj_func instead of parameter
            Utility = earnings.mean() + self.data.A_L * CVaR_L
        else:
            Utility =(1-self.data.A_L)*earnings.mean() + self.data.A_L * CVaR_L
        return Utility

    
    def Plotting_Barter_Set_Lemma2(self,plotting=False):
        """
        Plot the utility possibility curve for Lemma 2:
        Fix contract amount M, vary strike price S from S^R to S^U.
        """
        
        M_fixed = 0.5 * (self.BS_contract_amount_min + self.BS_contract_amount_max)
        S_space = np.linspace(self.BS_strikeprice_min, self.BS_strikeprice_max, self.n)

        V_Lemma2 = np.zeros((self.n, 2))
        for i, S in enumerate(S_space):
            # Calculate contract revenues for this S       
            V_Lemma2[i, 0] = self.Utility_G(S, M_fixed)
            V_Lemma2[i, 1] = self.Utility_L(S, M_fixed)
        
        slope, intercept, r_value, p_value, std_err = linregress(V_Lemma2[:, 0], V_Lemma2[:, 1])
        if self.old_obj_func == True:
            print(f"Theoreitcal Slope of Lemma 2 should be:{-(1+self.data.A_L)/(1+self.data.A_G):.3f}")
        else:
            print(f"Theoretical Slope of Lemma 2 should be:{-1:f}")
        
        print(f"Slope of Lemma 2 curve(Calculated): {slope:.4f}")

        if plotting == True:
            plt.figure(figsize=(10, 6))
            plt.plot(V_Lemma2[:, 0], V_Lemma2[:, 1], label='Lemma 2 Curve (M fixed)', color='purple')
            plt.scatter(V_Lemma2[0, 0], V_Lemma2[0, 1], color='purple', marker='o', s=100, label='Start (S = S^R)')
            plt.scatter(V_Lemma2[-1, 0], V_Lemma2[-1, 1], color='purple', marker='*', s=150, label='End (S = S^U)')
            plt.annotate(f"Slope: {slope:.2f}",
                 xy=(V_Lemma2[self.n//2, 0], V_Lemma2[self.n//2, 1]),
                 xytext=(30, 30), textcoords='offset points',
                 color='purple', fontsize=10,
                 arrowprops=dict(arrowstyle="->", color='purple'))
            plt.xlabel('Utility G')
            plt.ylabel('Utility L')
            plt.legend()
            plt.grid()
            plt.title(f'Lemma 2: Utility Set for Fixed M={M_fixed:.2f}, S in [{self.BS_strikeprice_min}, {self.BS_strikeprice_max}]')
            plt.show()


            plt.figure(figsize=(10, 6))
            plt.plot(V_Lemma2[:, 0], V_Lemma2[:, 1], label='Lemma 2 Curve (M fixed)', color='purple')
            plt.scatter(V_Lemma2[0, 0], V_Lemma2[0, 1], color='purple', marker='o', s=100, label='Start (S = S^R)')
            plt.scatter(V_Lemma2[-1, 0], V_Lemma2[-1, 1], color='purple', marker='*', s=150, label='End (S = S^U)')
            plt.annotate(f"Slope: {slope:.2f}",
                xy=(V_Lemma2[self.n//2, 0], V_Lemma2[self.n//2, 1]),
                xytext=(30, 30), textcoords='offset points',
                color='purple', fontsize=10,
                arrowprops=dict(arrowstyle="->", color='purple'))
            plt.xlabel('Utility G')
            plt.ylabel('Utility L')
            plt.legend()
            plt.grid()
            plt.title(f'Lemma 2: Utility Set for Fixed M={M_fixed:.2f}, S in [{self.BS_strikeprice_max}, {self.BS_strikeprice_max}]')
            plt.show()

        return slope

    def calculate_utility_derivative(self, M_space, V_1_Low, V_2_High):
        # Initial slope calculations

        dx = M_space[1] - M_space[0]  # Step size
        duG_1 = np.gradient(V_1_Low[:,0],M_space,edge_order=2)  
        duL_1 = np.gradient(V_1_Low[:,1],M_space,edge_order=2)  
        slope_1 = duL_1 / duG_1  
        
        duG_2 = np.gradient(V_2_High[:,0],M_space,edge_order=2)
        duL_2 = np.gradient(V_2_High[:,1],M_space,edge_order=2)
        slope_2 = duL_2 / duG_2

        # Lemma 5 MR
        cvgradientv1_L =  self.cvar_derivative_wrt_M_L(0,self.data.net_earnings_no_contract_priceL_L, self.data.price_L, self.BS_strikeprice_min, self.data.alpha)
        cvgradientv1_G =  self.cvar_derivative_wrt_M_G(0,self.data.net_earnings_no_contract_priceG_G, self.data.price_G, self.BS_strikeprice_min, self.data.alpha)
        Egradientv1_L = self.expectation_derivative_wrt_M_L(0,self.data.net_earnings_no_contract_priceL_L, self.data.price_L, self.BS_strikeprice_min)
        Egradientv1_G = self.expectation_derivative_wrt_M_G(0,self.data.net_earnings_no_contract_priceG_G, self.data.price_G, self.BS_strikeprice_min)

        uL_duG_theoretical_MR = ((1-self.data.A_L)*Egradientv1_L + self.data.A_L * cvgradientv1_L)/((1-self.data.A_G)*Egradientv1_G + self.data.A_G * cvgradientv1_G)
        
        # Lemma 5 MU
        cvgradientv2_L =  self.cvar_derivative_wrt_M_L(M_space[-1],self.data.net_earnings_no_contract_priceL_L, self.data.price_L, self.BS_strikeprice_min, self.data.alpha)
        cvgradientv2_G =  self.cvar_derivative_wrt_M_G(M_space[-1],self.data.net_earnings_no_contract_priceG_G, self.data.price_G, self.BS_strikeprice_min, self.data.alpha)
        Egradientv2_L = self.expectation_derivative_wrt_M_L(M_space[-1],self.data.net_earnings_no_contract_priceL_L, self.data.price_L, self.BS_strikeprice_min)
        Egradientv2_G = self.expectation_derivative_wrt_M_G(M_space[-1],self.data.net_earnings_no_contract_priceG_G, self.data.price_G, self.BS_strikeprice_min)
      
        uL_duG_theoretical_MU = ((1-self.data.A_L)*Egradientv2_L + self.data.A_L * cvgradientv2_L)/((1-self.data.A_G)*Egradientv2_G + self.data.A_G * cvgradientv2_G)

        #cond_MR = self.Condition_lemma5_MR()
        #cond_MU = self.Condition_lemma5_MU()

        if uL_duG_theoretical_MR < self.dS:
            cond_MR = True
        else:
            cond_MR = False
        
        if uL_duG_theoretical_MU > self.dS:
            cond_MU = True
        else:
            cond_MU = False

        # Find first crossing points
        mask_negative_v1 = slope_1 >= self.dS
        mask_positive_v2 = slope_2 <= self.dS
        first_index_negative_v1 = np.argmax(mask_negative_v1)
        first_index_positive_v2 = np.argmax(mask_positive_v2)
        M_SR = M_space[first_index_negative_v1]
        M_SU = M_space[first_index_positive_v2]

        if cond_MR == False and cond_MU == False:
            print("No Barter Set exists, as the conditions of Lemma 5 are not satisfied.")
            M_SR, M_SU = 0,0
            return cond_MR,cond_MU,None, None, M_SR, M_SU, None, None
        elif cond_MR == True and cond_MU == False:
            print("Barter Set exists, no concave part in the utility curves")
            M_SR,M_SU = self.BS_contract_amount_min, self.BS_contract_amount_max
            return cond_MR,cond_MU,None, None, M_SR, M_SU ,None , None
        else :
            print("Barter Set exists, concave part in the utility curves")

     
        """
        # Refined search around first crossing points
        if first_index_negative_v1 > 0 and first_index_positive_v2 >0 :
            
            n_refined = self.n * 2  # Double the number of points for refinement
        
            # SR refinement
            M_refined_SR = np.linspace(M_SR - 1, M_SR + 1, n_refined)
            dx_refined = M_refined_SR[1] - M_refined_SR[0]

            # Calculate refined utilities and slopes for V1
            V_1_refined = np.zeros((len(M_refined_SR), 2))
            for i, M in enumerate(M_refined_SR):               
                V_1_refined[i,0] = self.Utility_G(self.data.strikeprice_min, M)
                V_1_refined[i,1] = self.Utility_L(self.data.strikeprice_min, M)
            
            duG_1_refined = np.gradient(V_1_refined[:,0], M_refined_SR,edge_order=2)
            duL_1_refined = np.gradient(V_1_refined[:,1], M_refined_SR,edge_order=2)
            slope_1_refined = duL_1_refined / duG_1_refined
            
            # Find refined crossing point
            mask_refined = slope_1_refined >= self.dS
            refined_index_SR = np.argmax(mask_refined)
            M_SR_refined = M_refined_SR[refined_index_SR]
            
            print(f"\nRefined M_SR: {M_SR_refined:.6f} (Original: {M_SR:.6f})")
            print(f"Refined slope: {slope_1_refined[refined_index_SR]:.6f}")

        # Same refinement for V2
            M_refined_SU = np.linspace(M_SU - 1, M_SU + 1, n_refined)
            dx_refined = M_refined_SU[1] - M_refined_SU[0]

            
            # Calculate refined utilities and slopes for V2
            V_2_refined = np.zeros((len(M_refined_SU), 2))
            for i, M in enumerate(M_refined_SU):
                V_2_refined[i,0] = self.Utility_G(self.data.strikeprice_max,M)
                V_2_refined[i,1] = self.Utility_L(self.data.strikeprice_max,M)
            
            duG_2_refined = np.gradient(V_2_refined[:,0],M_refined_SU, edge_order=2)
            duL_2_refined = np.gradient(V_2_refined[:,1], M_refined_SU,edge_order=2)
            slope_2_refined = duL_2_refined / duG_2_refined
            
            # Find refined crossing point
            mask_refined = slope_2_refined <= self.dS
            refined_index_SU = np.argmax(mask_refined)
            M_SU_refined = M_refined_SU[refined_index_SU]
            
            print(f"\nRefined M_SU: {M_SU_refined:.6f} (Original: {M_SU:.6f})")
            print(f"Refined slope: {slope_2_refined[refined_index_SU]:.6f}")

        
            return cond_MR,cond_MU, slope_1_refined, slope_2_refined, M_SR_refined, M_SU_refined ,refined_index_SU , refined_index_SR
        else:
        """
        return cond_MR,cond_MU,slope_1, slope_2, M_SR, M_SU ,first_index_negative_v1 , first_index_positive_v2
    
 
    def Plotting_Barter_Set(self):

        self.dS = self.Plotting_Barter_Set_Lemma2(plotting=False) # dS slope from lemma 2 

        # Change in Bias if modified objective function is used 
  
        V_1_Low= np.zeros((self.n,2))
        V_2_High = np.zeros((self.n,2))

        #Saving threatpoint for plotting
        threat_point_x = self.data.Zeta_G
        threat_point_y = self.data.Zeta_L
      

        M_space = np.linspace(self.BS_contract_amount_min, self.BS_contract_amount_max, self.n)
        
        # Reshape M_space for proper broadcasting
        M_space_reshaped = M_space[:, None, None]  # Shape: (n, 1, 1)

        # Calculate the utility for each contract revenue
        for i in range(len(M_space)):            #Curve 1 
            V_1_Low[i,0] = self.Utility_G(self.BS_strikeprice_min, M_space[i]) 
            V_1_Low[i,1] = self.Utility_L(self.BS_strikeprice_min, M_space[i])
            #Curve 2
            V_2_High[i,0] = self.Utility_G(self.BS_strikeprice_max, M_space[i]) 
            V_2_High[i,1] = self.Utility_L(self.BS_strikeprice_max, M_space[i])

        
        cond_MR,cond_MU,slope_1, slope_2, M_SR,M_SU, first_index_v1,first_index_v2= self.calculate_utility_derivative(M_space,V_1_Low, V_2_High)
        # Calculate the slope of the utility curves     
        print(M_SR, M_SU)
        self.plot_utility_cvar_vs_M(strike_price='max')
        
        #Calculate Utlity for the optimal contract amount        
        UG_Low_Mopt = self.Utility_G(self.BS_strikeprice_min, M_SR)
        UL_Low_Mopt = self.Utility_L(self.BS_strikeprice_min, M_SR)
        UG_High_Mopt = self.Utility_G(self.BS_strikeprice_max, M_SU)
        UL_High_Mopt = self.Utility_L(self.BS_strikeprice_max, M_SU)
        UG_Low_Mopt_SR = self.Utility_G(self.data.SR_star_new, M_SR)
        UL_Low_Mopt_SR = self.Utility_L(self.data.SR_star_new, M_SR)
        UG_High_Mopt_SU = self.Utility_G(self.data.SU_star_new, M_SU)
        UL_High_Mopt_SU = self.Utility_L(self.data.SU_star_new, M_SU)

        # Temporary test values 
        UG_term1 = self.Utility_G(self.data.term1_G_new,M_SR)
        UL_term1 = self.Utility_L(self.data.term1_G_new,M_SR)

        UG_term2 = self.Utility_G(self.data.term2_G_new,M_SR)
        UL_term2 = self.Utility_L(self.data.term2_G_new,M_SR)
        UG_term3 = self.Utility_G(self.data.term3_L_SR_new,M_SR)
        UL_term3 = self.Utility_L(self.data.term3_L_SR_new,M_SR)

        UG_term4 = self.Utility_G(self.data.term4_L_SU_new,M_SR)
        UL_term4 = self.Utility_L(self.data.term4_L_SU_new,M_SR)

        #Utility from Scipy results 




        # Find Intersection Point with vertical line from threatpoint

        if cond_MR == True:
            slope_opt = np.round((UL_High_Mopt - UL_Low_Mopt) / (UG_High_Mopt - UG_Low_Mopt),0)
            b_opt = UL_Low_Mopt - slope_opt * UG_Low_Mopt
            vertical_intersect_y = slope_opt * threat_point_x + b_opt
            
            # Horizontal line intersection (y = threat_point_y)
            horizontal_intersect_x = (threat_point_y - b_opt) / slope_opt
    
        
        #self.plot_utility_cvar_vs_M(strike_price='min')
        #self.plot_utility_cvar_vs_M(strike_price='max')

        # Keeping SR constant and plotting through MR to MU (Curve 1)
        plt.figure(figsize=(10, 6))
        plt.plot(V_1_Low[:,0], V_1_Low[:,1], label='Curve 1', color='blue')
        plt.plot(V_2_High[:,0], V_2_High[:,1], label='Curve 2', color='red')

        arrow_positions = np.linspace(0,1,10)  # Positions along the curve (as fractions)
        for pos in arrow_positions:
            point_idx = int(len(V_1_Low) * pos)
            if point_idx + 1 < len(V_1_Low):
                plt.annotate('', 
                    xy=(V_1_Low[point_idx+1,0], V_1_Low[point_idx+1,1]),
                    xytext=(V_1_Low[point_idx,0], V_1_Low[point_idx,1]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                    annotation_clip=True)

        # Add multiple direction arrows for Curve 2 (red)
        for pos in arrow_positions:
            point_idx = int(len(V_2_High) * pos)
            if point_idx + 1 < len(V_2_High):
                plt.annotate('', 
                    xy=(V_2_High[point_idx+1,0], V_2_High[point_idx+1,1]),
                    xytext=(V_2_High[point_idx,0], V_2_High[point_idx,1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    annotation_clip=True)

        # Add single "M increasing" label for each curve
        plt.annotate('M increasing', 
                    xy=(V_1_Low[len(V_1_Low)//2,0], V_1_Low[len(V_1_Low)//2,1]),
                    xytext=(30, 30), textcoords='offset points',
                    color='blue', fontsize=10)
        plt.annotate('M increasing',
                    xy=(V_2_High[len(V_2_High)//2,0], V_2_High[len(V_2_High)//2,1]),
                    xytext=(30, 30), textcoords='offset points',
                    color='red', fontsize=10)

        # Hopefully this should be at the intersection points 
        # Plot the points
        
        if cond_MR==True:
            plt.scatter(UG_Low_Mopt_SR, UL_Low_Mopt_SR, color='cyan', marker='D', s=100, 
            label=f'SR Utility (M={M_SR:.2f})')
            plt.scatter(UG_High_Mopt_SU, UL_High_Mopt_SU, color='magenta', marker='D', s=100, 
            label=f'SU Utility (M={M_SU:.2f})')

            # Plot Optimal Contract Amount Point with fixed price SR and SU
            plt.scatter(UG_Low_Mopt, UL_Low_Mopt, color='green', marker='o', s=100, label=f'V1 M* = ({M_SR:2f} MW)')
            plt.scatter(UG_High_Mopt, UL_High_Mopt, color='green', marker='*', s=150, label=f'V2 M* = ({M_SU:2f} MW)')

            # Draw straight line between optimal points
            plt.plot([UG_Low_Mopt, UG_High_Mopt], 
                    [UL_Low_Mopt, UL_High_Mopt], 
                'g--', 
                label='Line between optimal points',
                alpha=0.3)  # alpha makes line slightly transparent        # Plot Nash bargaining terms
            plt.scatter(UG_term1, UL_term1, color='brown', marker='s', s=100, label='Term 1 (G)')
            plt.scatter(UG_term2, UL_term2, color='brown', marker='v', s=100, label='Term 2 (G)')
            plt.scatter(UG_term3, UL_term3, color='purple', marker='s', s=100, label='Term 3 (L)')
            plt.scatter(UG_term4, UL_term4, color='purple', marker='v', s=100, label='Term 4 (L)')

            # Plot Utility G and L for the contract at optimal solution 
            #plt.scatter(self.results.utility_G, self.results.utility_L, color='orange', marker='o', s=100, label='Optimal Solution (G,L)')
            #Plot scipy resulting utility 
            #plt.scatter(self.scipy_results.utility_G, self.scipy_results.utility_L, color='red', marker='o', s=100, label='Scipy Result (G,L)')

        # After plotting the threat point, add horizontal and vertical lines
        # Add horizontal line from threat point
        plt.axhline(y=threat_point_y, color='black', linestyle='--', alpha=0.3)

        # Add vertical line from threat point  
        plt.axvline(x=threat_point_x, color='black', linestyle='--', alpha=0.3)

         # Plot intersection points
        plt.scatter(threat_point_x, vertical_intersect_y, 
                color='purple', marker='x', s=100,) 
                #label='Vertical Line Intersection')
        plt.scatter(horizontal_intersect_x, threat_point_y, 
                color='purple', marker='x', s=100,) 
                #label='Horizontal Line Intersection')

        # Original threat point plotting
        plt.scatter(threat_point_x, threat_point_y, color='black', marker='o', s=100, label='Threatpoint')

        vertices = np.array([
        [threat_point_x, threat_point_y],  # Start at threat point
        [threat_point_x, vertical_intersect_y],      # Go right to first optimal point
        [horizontal_intersect_x, threat_point_y],        # Go to second optimal point      # Go down vertically
        [threat_point_x, threat_point_y]    # Back to start
        ])

        # Create and add the polygon
        polygon = plt.Polygon(vertices, facecolor='gray', alpha=0.2, edgecolor=None)
        plt.gca().add_patch(polygon)


        plt.xlabel('Utility G')
        plt.ylabel('Utility L')
        if self.old_obj_func == True:
            plt.title(r'Barter Set (E($\pi$) + A*CVaR($\pi$))')
        else:
            plt.title(r'Barter Set (1-A)*E($\pi$) + A*CVaR($\pi$)')
        plt.legend()
        plt.grid()
        plt.show()

   
        print("Done")
  

    def plot_utility_cvar_vs_M(self, strike_price='min'):
        """
        Diagnostic plot: show how Expected, CVaR, and Utility evolve with contract amount M
        for fixed strike price (S = strikeprice_min or strikeprice_max).
        """
        if strike_price == 'min':
            S = self.data.strikeprice_min
            title_suffix = "S = S^R (Min)"
        elif strike_price == 'max':
            S = self.data.strikeprice_max
            title_suffix = "S = S^U (Max)"
        else:
            raise ValueError("strike_price must be 'min' or 'max'")

        M_space = np.linspace(self.BS_contract_amount_min, self.BS_contract_amount_max, self.n)

        UG_exp = []
        UG_cvar = []
        UG_total = []

        UL_exp = []
        UL_cvar = []
        UL_total = []
        hours = self.data.hours_in_year # shape (n_months,)


        for M in M_space:
            # Contract revenue per scenario

            rev_contract_G = (M * hours) * ((S * hours) - self.data.price_G)
            rev_contract_total_G = rev_contract_G.sum(axis=0)
            Expected_G = self.data.net_earnings_no_contract_priceG_G 
            earnings_G = Expected_G + rev_contract_total_G
            CVaR_G = calculate_cvar_left(earnings_G, self.data.alpha)
      
            Utility_G = (1-self.data.A_G)*earnings_G.mean() + self.data.A_G * CVaR_G
          
    
            rev_contract_L = (M * hours) * (( self.data.price_L) - (S * hours))
            rev_contract_total_L = rev_contract_L.sum(axis=0)
            Expected_L = self.data.net_earnings_no_contract_priceL_L
            earnings_L = Expected_L + rev_contract_total_L
            CVaR_L = calculate_cvar_left(earnings_L, self.data.alpha)
    
            Utility_L =(1-self.data.A_L)*earnings_L.mean() + self.data.A_L * CVaR_L

            UG_exp.append(earnings_G.mean())
            UG_cvar.append(CVaR_G)
            UG_total.append(Utility_G)

            UL_exp.append(earnings_L.mean())
            UL_cvar.append(CVaR_L)
            UL_total.append(Utility_L)

        # Calculate rate of change (derivative)
        dcvar_G = np.gradient(UG_cvar, M_space, edge_order= 1)
        dcvar_L = np.gradient(UL_cvar, M_space,edge_order= 1)

          # Second derivatives
        d2cvar_G = np.gradient(dcvar_G, M_space, edge_order=1)
        d2cvar_L = np.gradient(dcvar_L, M_space,edge_order=1)

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].plot(M_space, UG_exp, label='Expected G')
        axs[0].plot(M_space, UG_cvar, label='CVaR G')
        axs[0].plot(M_space, UG_total, label='Utility G', linewidth=2)
        axs[0].set_title(f"G Utility Components vs M ({title_suffix})")
        axs[0].set_xlabel("Contract Amount M")
        axs[0].set_ylabel("Value")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(M_space, UL_exp, label='Expected L')
        axs[1].plot(M_space, UL_cvar, label='CVaR L')
        axs[1].plot(M_space, UL_total, label='Utility L', linewidth=2)
        axs[1].set_title(f"L Utility Components vs M ({title_suffix})")
        axs[1].set_xlabel("Contract Amount M")
        axs[1].set_ylabel("Value")
        axs[1].legend()
        axs[1].grid(True)

        plt.suptitle(f"Diagnostic Utility Analysis ({title_suffix})")
        plt.tight_layout()
        plt.show()

        fig, ((ax1, ax2, ax3), (ax4, ax5,ax6)) = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot CVaR values
        ax1.plot(M_space, UG_cvar, label='CVaR G', color='blue')
        ax1.set_title(f'Generator CVaR vs M ({title_suffix})')
        ax1.set_xlabel('Contract Amount (M)')
        ax1.set_ylabel('CVaR Value')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(M_space, UL_cvar, label='CVaR L', color='red')
        ax2.set_title(f'Load CVaR vs M ({title_suffix})')
        ax2.set_xlabel('Contract Amount (M)')
        ax2.set_ylabel('CVaR Value')
        ax2.grid(True)
        ax2.legend()
        
        # Plot CVaR rate of change
        ax3.plot(M_space, dcvar_G, label='dCVaR/dM G', color='blue')
        ax3.set_title('Rate of Change of Generator CVaR')
        ax3.set_xlabel('Contract Amount (M)')
        ax3.set_ylabel('dCVaR/dM')
        ax3.grid(True)
        ax3.legend()
        
        ax4.plot(M_space, dcvar_L, label='dCVaR/dM L', color='red')
        ax4.set_title('Rate of Change of Load CVaR')
        ax4.set_xlabel('Contract Amount (M)')
        ax4.set_ylabel('dCVaR/dM')
        ax4.grid(True)
        ax4.legend()

           # Plot first derivatives
        ax5.plot(M_space, d2cvar_L, label='Second Derivative Load', color='blue')
        ax5.axhline(y=0, color='r', linestyle='--')
        ax5.set_title('Second Derivative of CVaR of Load')
        ax5.set_xlabel('Contract Amount (M)')
        ax5.set_ylabel('dCVaR/dM')
        ax5.grid(True)
        ax5.legend()
        
        # Plot second derivatives
        ax6.plot(M_space, d2cvar_G, label='Second Derivative G', color='blue')
        ax6.axhline(y=0, color='r', linestyle='--')
        ax6.set_title('Second Derivative of CVaR of G')
        ax6.set_xlabel('Contract Amount (M)')
        ax6.set_ylabel('d²CVaR/dM²')
        ax6.grid(True)
        ax6.legend()
            
        plt.suptitle(f'CVaR Sensitivity Analysis for {title_suffix}')
        plt.tight_layout()
        plt.show()

        print()
