import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from utils import calculate_cvar_left, _left_tail_mask
from scipy.signal import find_peaks



class Barter_Set:
    def __init__(self, data,results,scipy_results):
        self.data = data
        self.results = results
        self.scipy_results = scipy_results
        self.n = 1000  # Number of points for plotting
        #self.BS_strike_min = data.strikeprice_min- 20*1e-3
        #self.BS_strike_max = data.strikeprice_max+ 20*1e-3

        if self.data.contract_type == "PAP":
            self.BS_strike_min = 25 * 1e-3
            self.BS_strike_max = 120 * 1e-3
        else:
            self.BS_strike_min =  data.SR_star_new
            self.BS_strike_max =  self.data.SU_star_new -1e-5
        print(f"{self.BS_strike_min*1e3:.4f} EUR/MWh")
        print(f"{self.BS_strike_max*1e3:.4f} EUR/MWh")


    

    def cvar_derivative_wrt_M_L(self, M_base, earnings_base, price_matrix, strike, alpha):
        """
        Estimate gradient of CVaR with respect to contract share M (gamma) for the Load-side in PAP.
        """
        epsilon = dx

        if self.data.contract_type == "PAP":
            gamma_plus = M_base + epsilon
            gamma_minus = M_base - epsilon

            # Constant across gamma
            EuL = (-self.data.price_L * self.data.load_CR * self.data.load_scenarios).sum(axis=0)

            # Contracted revenue depends on gamma
            con_plus = (gamma_plus * self.data.production * (self.data.price_G * self.data.capture_rate - strike)).sum(axis=0)
            con_minus = (gamma_minus * self.data.production * (self.data.price_G * self.data.capture_rate - strike)).sum(axis=0)

            rev_plus = EuL + con_plus
            rev_minus = EuL + con_minus

            cvar_plus = calculate_cvar_left(rev_plus, alpha)
            cvar_minus = calculate_cvar_left(rev_minus, alpha)

        else:
            M_plus = M_base + epsilon
            M_minus = M_base - epsilon

            rev_plus = (M_plus * (price_matrix - strike)).sum(axis=0)
            rev_minus = (M_minus * (price_matrix - strike)).sum(axis=0)

            cvar_plus = calculate_cvar_left(earnings_base + rev_plus, alpha)
            cvar_minus = calculate_cvar_left(earnings_base + rev_minus, alpha)

        return (cvar_plus - cvar_minus) / (2 * epsilon)

    def cvar_derivative_wrt_M_G(self, M_base, earnings_base, price_matrix, strike, alpha):
        """
        Estimate gradient of CVaR with respect to contract share M (gamma) for the Generator-side in PAP.
        """
        epsilon = dx

        if self.data.contract_type == "PAP":
            gamma_plus = M_base + epsilon
            gamma_minus = M_base - epsilon

            # Generator revenue with gamma_plus
            rev_plus = (
                (1 - gamma_plus) * self.data.production * self.data.price_G * self.data.capture_rate+ gamma_plus * self.data.production * strike
            ).sum(axis=0)

            # Generator revenue with gamma_minus
            rev_minus = (
                (1 - gamma_minus) * self.data.production * self.data.price_G * self.data.capture_rate + gamma_minus * self.data.production * strike
            ).sum(axis=0)

            cvar_plus = calculate_cvar_left(rev_plus, alpha)
            cvar_minus = calculate_cvar_left(rev_minus, alpha)

        else:
            M_plus = M_base + epsilon
            M_minus = M_base - epsilon

            rev_plus = (M_plus * (strike - price_matrix)).sum(axis=0)
            rev_minus = (M_minus * (strike - price_matrix)).sum(axis=0)

            cvar_plus = calculate_cvar_left(earnings_base + rev_plus, alpha)
            cvar_minus = calculate_cvar_left(earnings_base + rev_minus, alpha)

        return (cvar_plus - cvar_minus) / (2 * epsilon)
    
    def expectation_derivative_wrt_M_L(self, M_base, earnings_base, price_matrix, strike):
        """
        Estimate the gradient of the expected value of earnings with respect to contract volume M.
        """
        epsilon = dx
        if self.data.contract_type =="PAP":
        
            gamma_plus = M_base + epsilon
            gamma_minus = M_base - epsilon

            EuL = (-self.data.price_L * self.data.load_CR * self.data.load_scenarios).sum(axis=0) # Sum across time periods for each scenario

            con_plus = (gamma_plus * self.data.production * (self.data.price_G * self.data.capture_rate - strike)).sum(axis=0)
            con_minus = (gamma_minus * self.data.production * (self.data.price_G * self.data.capture_rate - strike)).sum(axis=0)
            
            rev_plus = EuL + con_plus
            rev_minus = EuL + con_minus
            # Calculate expected earnings for M_plus and M_minus
            expected_plus = rev_plus.mean()
            expected_minus = rev_minus.mean() 
        else:
            M_plus = M_base + epsilon
            M_minus = M_base - epsilon

            rev_plus = ((M_plus ) * (( price_matrix) - (strike ))).sum(axis=0)
            rev_minus =((M_minus ) * (( price_matrix) - (strike ))).sum(axis=0)

            # Calculate expected earnings for M_plus and M_minus
            expected_plus = (earnings_base + rev_plus).mean()
            expected_minus = (earnings_base + rev_minus).mean()


        # Return the finite difference approximation of the derivative
        return (expected_plus - expected_minus) / (2 * epsilon)

    def expectation_derivative_wrt_M_G(self, M_base, earnings_base, price_matrix, strike):
        """
        Estimate the gradient of the expected value of earnings with respect to contract volume M.
        """
        epsilon = dx
        if self.data.contract_type =="PAP":
            gamma_plus = M_base + epsilon
            gamma_minus = M_base - epsilon
            
            rev_plus = ((1-gamma_plus) * self.data.production * self.data.price_G * self.data.capture_rate + gamma_plus * self.data.production * strike).sum(axis=0)
            rev_minus = ((1-gamma_minus) * self.data.production * self.data.price_G * self.data.capture_rate + gamma_plus * self.data.production * strike).sum(axis=0)

            expected_plus = rev_plus.mean()
            expected_minus = rev_minus.mean()
        
        else:
            M_plus  = M_base  + epsilon
            M_minus = M_base - epsilon

            rev_plus = ( M_plus*((strike) - price_matrix)).sum(axis=0)
            rev_minus = ( M_minus* ((strike)  - price_matrix  )).sum(axis=0)

            expected_plus = (earnings_base + rev_plus).mean()
            expected_minus = (earnings_base + rev_minus).mean()


        # Calculate expected earnings for M_plus and M_minus
        
        # Return the finite difference approximation of the derivative
        return (expected_plus - expected_minus) / (2 * epsilon)

    def Utility_G(self, strike,volume):

        """
        strike : Strike Price [float]
        volume : Contract volume [float] or percentage [float] in PAP
        """

        if self.data.contract_type == "PAP":
            # For each scenario, sum production over time, then apply contract fraction
            earnings = ((1-volume) * self.data.production * self.data.price_G * self.data.capture_rate + volume * self.data.production * strike).sum(axis=0)
            # earnings: scenario-wise
            CVaR_G = calculate_cvar_left(earnings, self.data.alpha) 
        
        else:
            rev_contract = volume  * (strike - self.data.price_G)
            rev_contract_total = rev_contract.sum(axis=0)
            no_contract = self.data.net_earnings_no_contract_priceG_G 
        
            earnings = no_contract + rev_contract_total
        
            CVaR_G = calculate_cvar_left(earnings, self.data.alpha)
        
        Utility = (1 - self.data.A_G) * earnings.mean() + self.data.A_G * CVaR_G
        return Utility

    def Utility_L(self, strike,volume):
        if self.data.contract_type =="PAP":


            # For each scenario, sum over time
            # Contracted load: γ × production at strike, rest at spot
            # Load's net position: (price_L * load_CR * (γ * production - load_scenarios)).sum(axis=0) + (−γ * production * strike).sum(axis=0)
            EuL = (-self.data.price_L * self.data.load_CR * self.data.load_scenarios).sum(axis=0) # Sum across time periods for each scenario
            SML =   (volume* self.data.production * self.data.price_L * self.data.capture_rate -  volume * strike * self.data.production).sum(axis=0) # Sum across time periods for each scenario

            earnings = EuL + SML
            CVaR_L = calculate_cvar_left(earnings, self.data.alpha)
        else:
            rev_contract = (volume * (self.data.price_L - strike)).sum(axis=0)
            no_conctract = self.data.net_earnings_no_contract_priceL_L

            earnings = no_conctract + rev_contract
        
        CVaR_L = calculate_cvar_left(earnings, self.data.alpha)
    
        Utility =(1-self.data.A_L)*earnings.mean() + self.data.A_L * CVaR_L
        
        
        return Utility

    def _Revenue_G(self, strike, volume):

        if self.data.contract_type == "PAP":
            pi_G = ((1-volume) * self.data.production * self.data.price_G * self.data.capture_rate + volume * self.data.production * strike).sum(axis=0)
        else: 
            pi_G = self.data.net_earnings_no_contract_priceG_G + volume * (strike - self.data.price_G).sum(axis=0)
        return pi_G
    
    def _Revenue_L(self, strike, volume):
        if self.data.contract_type == "PAP":
            # For each scenario, sum over time
            EuL = (-self.data.price_L * self.data.load_CR * self.data.load_scenarios).sum(axis=0) # Sum across time periods for each scenario
            SML = (volume * self.data.production * (self.data.price_L * self.data.capture_rate - strike)).sum(axis=0) # Sum across time periods for each scenario

            pi_L = EuL + SML
        else:
            pi_L = self.data.net_earnings_no_contract_priceL_L + volume * (self.data.price_L - strike).sum(axis=0)

        return pi_L
    
    def _dS_PAP(self,S,gamma):
        """
        Calculate the slope of the utility curve for PAP contracts.
        S: Strike Price
        gamma: Contract volume (percentage)
        """
        pi_G = self._Revenue_G(S, gamma)
        pi_L = self._Revenue_L(S, gamma)

        mask_G = _left_tail_mask(pi_G, self.data.alpha)
        mask_L = _left_tail_mask(pi_L, self.data.alpha)

        prod = self.data.production.sum(axis=0)

        tail_G = prod[mask_G].mean() if mask_G.any() else 0.0
        tail_L = prod[mask_L].mean() if mask_L.any() else 0.0

        num =  (1-self.data.A_L)*prod.mean() + self.data.A_L * tail_L
        den =  (1-self.data.A_G)*prod.mean() + self.data.A_G * tail_G

        return - num / den
    
    def _dgamma_PAP(self,S,gamma):
        """
        Calculate the slope of the utility curve for PAP contracts.
        S: Strike Price
        gamma: Contract volume (percentage)
        """
        pi_G = self._Revenue_G(S, gamma)
        pi_L = self._Revenue_L(S, gamma)

        mask_G = _left_tail_mask(pi_G, self.data.alpha)
        mask_L = _left_tail_mask(pi_L, self.data.alpha)
        
        rev_G = ( self.data.production*(S-self.data.capture_rate*self.data.price_G)).sum(axis=0)
        rev_L = ( self.data.production*(self.data.capture_rate*self.data.price_L-S)).sum(axis=0)


        tail_G = rev_G[mask_G].mean() if mask_G.any() else 0.0
        tail_L = rev_L[mask_L].mean() if mask_L.any() else 0.0

        num =  (1-self.data.A_L) + self.data.A_L * tail_L
        den =  (1-self.data.A_G) + self.data.A_G * tail_G

        return - num / den
    
    def Plotting_Barter_Set_Lemma2(self,plotting=False):
        """
        Plot the utility possibility curve for Lemma 2:
        Fix contract amount M, vary strike price S from S^R to S^U.
        """
        if self.data.contract_type == "PAP":
            M_fixed = 0.4 # [0,1] for PAP
        else:
            M_fixed = 0.5 * (self.data.contract_amount_min + self.data.contract_amount_max)
        
        S_space = np.linspace(self.BS_strike_min, self.BS_strike_max, self.n)

        V_Lemma2 = np.zeros((self.n, 2))
        for i, S in enumerate(S_space):
            # Calculate contract revenues for this S       
            V_Lemma2[i, 0] = self.Utility_G(S, M_fixed)
            V_Lemma2[i, 1] = self.Utility_L(S, M_fixed)
        
        slope, intercept, r_value, p_value, std_err = linregress(V_Lemma2[:, 0], V_Lemma2[:, 1])
        print(f"R Value: {r_value:.4f}")
        print(f"Numerical Slope of Lemma 2 curve: {slope:.4f}")
        
        if  self.data.contract_type == "PAP":

            pap_slope = np.empty(self.n)

            # ---- masks for the α-tails --------------------------------------

            prod = self.data.production.sum(axis=0)
            E_prod_G =prod.mean()

            for i, S in enumerate(S_space):
            
                pi_G = self._Revenue_G(S, M_fixed)
                pi_L = self._Revenue_L(S, M_fixed)

                mask_G = _left_tail_mask(pi_G, self.data.alpha)
                test = calculate_cvar_left(pi_G, self.data.alpha)
                mask_L = _left_tail_mask(pi_L, self.data.alpha)

                tail_G = prod[mask_G].mean() if mask_G.any() else 0.0
                tail_L = prod[mask_L].mean() if mask_L.any() else 0.0

                num = (1-self.data.A_L) * E_prod_G + (self.data.A_L) * tail_L
                den = (1-self.data.A_G) * E_prod_G + (self.data.A_G) * tail_G
                pap_slope[i] = - num / den
            theo_slope = pap_slope.mean()
            print(f"Theoretical Slope of Lemma 2 should be:{theo_slope:.4f}")
  
        else:
            print(f"Theoretical Slope of Lemma 2 should be:{-1:f}")
        

        if plotting == True:
            plt.figure(figsize=(10, 6))
            plt.plot(V_Lemma2[:, 0], V_Lemma2[:, 1], label='Lemma 2 Curve (M fixed)', color='purple')
            plt.scatter(V_Lemma2[0, 0], V_Lemma2[0, 1], color='purple', marker='o', s=100, label='Start (S = S^R)')
            plt.scatter(V_Lemma2[-1, 0], V_Lemma2[-1, 1], color='purple', marker='*', s=150, label='End (S = S^U)')
            plt.annotate(f"Slope: {slope:.4f}",
                 xy=(V_Lemma2[self.n//2, 0], V_Lemma2[self.n//2, 1]),
                 xytext=(30, 30), textcoords='offset points',
                 color='purple', fontsize=10,
                 arrowprops=dict(arrowstyle="->", color='purple'))
            plt.xlabel('Utility G')
            plt.ylabel('Utility L')
            plt.legend()
            plt.grid()
            plt.title(f'Lemma 2: Utility Set for Fixed M={M_fixed:.2f} MWh, S in [{self.BS_strike_min*1e3}, {self.BS_strike_max*1e3}] EUR/MWh')
            plt.show()

        return slope

    def calculate_utility_derivative(self, M_space, V_1_Low, V_2_High):
        # Initial slope calculations

        if self.data.contract_type == "PAP":
            dS_SR = np.zeros(self.n)
            dS_SU = np.zeros(self.n)

            for i in range(self.n):
                dS_SR[i] = self._dS_PAP(self.BS_strike_min, M_space[i])
                dS_SU[i] = self._dS_PAP(self.BS_strike_max, M_space[i])

        global dx
        dx = M_space[1] - M_space[0]  # Step size
        duG_1 = np.gradient(V_1_Low[:,0],dx,edge_order=1)  
        duL_1 = np.gradient(V_1_Low[:,1],dx,edge_order=1)  
        slope_1 = duL_1 / duG_1  
        
        duG_2 = np.gradient(V_2_High[:,0],dx,edge_order=1)
        duL_2 = np.gradient(V_2_High[:,1],dx,edge_order=1)
        slope_2 = duL_2 / duG_2
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        ax_1 = axes[0]
        ax_2 = axes[1]
        ax_1.plot(M_space, slope_1, label='Slope Curve 1', color='blue'  )
        if self.data.contract_type == "PAP":
            ax_1.plot(M_space, dS_SR, label='dS Curve 1', color='black', linestyle='--')
        else:
            ax_1.axhline(self.dS, color='black', linestyle='--', label='dS Threshold')
        ax_2.plot(M_space, slope_2, label='Slope Curve 2', color='red'  )
        if self.data.contract_type == "PAP":
            ax_2.plot(M_space, dS_SU, label='dS Curve 2', color='black', linestyle='--')
        else:
            ax_2.axhline(self.dS, color='black', linestyle='--', label='dS Threshold')
        plt.show()

        # Lemma 5 MR (L)
        cvgradientv1_L =  self.cvar_derivative_wrt_M_L(M_space[0],self.data.net_earnings_no_contract_priceL_L, self.data.price_L, self.BS_strike_min, self.data.alpha)
        cvgradientv1_G =  self.cvar_derivative_wrt_M_G(M_space[0],self.data.net_earnings_no_contract_priceG_G, self.data.price_G, self.BS_strike_min, self.data.alpha)
        Egradientv1_L = self.expectation_derivative_wrt_M_L(M_space[0],self.data.net_earnings_no_contract_priceL_L, self.data.price_L, self.BS_strike_min)
        Egradientv1_G = self.expectation_derivative_wrt_M_G(M_space[0],self.data.net_earnings_no_contract_priceG_G, self.data.price_G, self.BS_strike_min)

        
        # Lemma 5 MU (L )
        cvgradientv2_L =  self.cvar_derivative_wrt_M_L(M_space[-1],self.data.net_earnings_no_contract_priceL_L, self.data.price_L, self.BS_strike_min, self.data.alpha)
        cvgradientv2_G =  self.cvar_derivative_wrt_M_G(M_space[-1],self.data.net_earnings_no_contract_priceG_G, self.data.price_G, self.BS_strike_min, self.data.alpha)
        Egradientv2_L = self.expectation_derivative_wrt_M_L(M_space[-1],self.data.net_earnings_no_contract_priceL_L, self.data.price_L, self.BS_strike_min)
        Egradientv2_G = self.expectation_derivative_wrt_M_G(M_space[-1],self.data.net_earnings_no_contract_priceG_G, self.data.price_G, self.BS_strike_min)

        if self.data.contract_type == "PAP":
            uL_duG_theoretical_MR = ((1-self.data.A_L)*Egradientv1_L + self.data.A_L * cvgradientv1_L)/((1-self.data.A_G)*Egradientv1_G + self.data.A_G * cvgradientv1_G)
            uL_duG_theoretical_MU = ((1-self.data.A_L)*Egradientv2_L + self.data.A_L * cvgradientv2_L)/((1-self.data.A_G)*Egradientv2_G + self.data.A_G * cvgradientv2_G)

            test_MR = self._dgamma_PAP(self.BS_strike_min, M_space[0])
            test_MU = self._dgamma_PAP(self.BS_strike_min, M_space[-1])
        
        else:
            uL_duG_theoretical_MR = ((1-self.data.A_L)*Egradientv1_L + self.data.A_L * cvgradientv1_L)/((1-self.data.A_G)*Egradientv1_G + self.data.A_G * cvgradientv1_G)
            uL_duG_theoretical_MU = ((1-self.data.A_L)*Egradientv2_L + self.data.A_L * cvgradientv2_L)/((1-self.data.A_G)*Egradientv2_G + self.data.A_G * cvgradientv2_G)

        #cond_MR = self.Condition_lemma5_MR()
        #cond_MU = self.Condition_lemma5_MU()

        print(slope_1[0])
        print(uL_duG_theoretical_MR)
        print(slope_1[-1])
        print(uL_duG_theoretical_MU)

        print(test_MR)
        print(test_MU)

    
        if self.data.contract_type == "PAP":
            self.dS_min = self._dS_PAP(self.BS_strike_min, M_space[0])
            self.dS_max = self._dS_PAP(self.BS_strike_min, M_space[-1])
        else:
            self.dS_min = self.dS
            self.dS_max = self.dS
       
        if slope_1[0] < self.dS_min:
            cond_MR = True
        else:
            cond_MR = False
        
        if slope_1[-1] > self.dS_max:
            cond_MU = True
        else:
            cond_MU = False


        # Find first crossing points
        if self.data.contract_type == "PAP":
            mask_negative_v1 = slope_1 <= dS_SR
            mask_positive_v2 = slope_2 >= dS_SU
        else:
            mask_negative_v1 = slope_1 <= self.dS
            mask_positive_v2 = slope_2 >= self.dS
        first_index_negative_v1 = np.argmax(mask_negative_v1)
        first_index_positive_v2 = np.argmax(mask_positive_v2)
        M_SR = M_space[first_index_negative_v1]
        M_SU = M_space[first_index_positive_v2]

        if cond_MR == False and cond_MU == False:
            print("No Barter Set exists, as the conditions of Lemma 5 are not satisfied.")
            M_SR, M_SU = M_SU,M_SU
            return cond_MR,cond_MU,None, None, M_SR, M_SU, None, None
        elif cond_MR == True and cond_MU == False:
            print("Barter Set exists, no concave part in the utility curves")
            if self.data.contract_type == "PAP":
                M_SR = 1
                M_SU = 1
            else:
                M_SR = self.data.contract_amount_min
                M_SU = self.data.contract_amount_max
            return cond_MR,cond_MU,None, None, M_SR, M_SU ,None , None
        else :
            print("Barter Set exists, concave part in the utility curves")

        return cond_MR,cond_MU,slope_1, slope_2, M_SR, M_SU ,first_index_negative_v1 , first_index_positive_v2
    
    def Plotting_Barter_Set(self):

        self.dS = self.Plotting_Barter_Set_Lemma2(plotting=False) # dS slope from lemma 2 
        #self.dS =-1
        # Change in Bias if modified objective function is used 
  
        V_1_Low= np.zeros((self.n,2))
        V_2_High = np.zeros((self.n,2))

        V_CP= np.zeros((self.n,2))


        #Saving threatpoint for plotting
        threat_point_x = self.data.Zeta_G
        threat_point_y = self.data.Zeta_L

        #self.plot_utility_cvar_vs_M(self.data.SR_star_new )

        
        
      
        if self.data.contract_type == "PAP":
            # For PAP, we need to calculate the contract amount as a percentage of production
            M_space = np.linspace(0, 2, self.n)
        else:
            M_space = np.linspace(self.data.contract_amount_min, self.data.contract_amount_max, self.n)
        
        # Reshape M_space for proper broadcasting
        test = np.zeros((self.n,2))

        # Calculate the utility for each contract revenue
        for i in range(len(M_space)):            #Curve 1 
            V_1_Low[i,0] = self.Utility_G(self.BS_strike_min, M_space[i]) 
            V_1_Low[i,1] = self.Utility_L(self.BS_strike_min, M_space[i])
            #Curve 2
            V_2_High[i,0] = self.Utility_G(self.BS_strike_max , M_space[i]) 
            V_2_High[i,1] = self.Utility_L(self.BS_strike_max , M_space[i])


        #nash_product_test = (test[:,0]-self.data.Zeta_G)*(test[:,1]-self.data.Zeta_L)
        nash_product_low = (V_1_Low[:,0]-self.data.Zeta_G)*(V_1_Low[:,1]-self.data.Zeta_L)

        nash_product_high = (V_2_High[:,0]-self.data.Zeta_G)*(V_2_High[:,1]-self.data.Zeta_L)

        
        cond_MR,cond_MU,slope_1, slope_2, M_SR,M_SU, first_index_v1,first_index_v2= self.calculate_utility_derivative(M_space,V_1_Low, V_2_High)
        # Calculate the slope of the utility curves     
        #print(M_SR, M_SU)
        #self.plot_utility_cvar_vs_M(strike_price='min')
        #self.plot_utility_cvar_vs_M(strike_price='max')
        #self.plot_utility_contours()
        
        #Calculate Utlity for the optimal contract amount        
        UG_Low_Mopt = self.Utility_G(self.BS_strike_min, M_SR)
        UL_Low_Mopt = self.Utility_L(self.BS_strike_min, M_SR)
        UG_High_Mopt = self.Utility_G(self.BS_strike_max, M_SU)
        UL_High_Mopt = self.Utility_L(self.BS_strike_max, M_SU)
        

        #Utility R* and SU*

        CP = self.data.capture_rate.mean().mean()*self.data.price_true.mean().mean()

        for i in range(len(M_space)):            #Curve 1 
            V_CP[i,0] = self.Utility_G(CP, M_space[i]) 
            V_CP[i,1] = self.Utility_L(CP, M_space[i])
     
        
        #UG_nego_upper,UL_nego_upper = self.Utility_G(negotation_upper,M_SR),self.Utility_L(negotation_upper,M_SR)
        #UG_nego_lower,UL_nego_lower = self.Utility_G(negotation_lower,M_SR),self.Utility_L(negotation_lower,M_SR)

        
        # Find Intersection Point with vertical line from threatpoint

        if cond_MR == True:
            slope_opt = np.round((UL_High_Mopt - UL_Low_Mopt) / (UG_High_Mopt - UG_Low_Mopt),8)
            b_opt = UL_Low_Mopt - slope_opt * UG_Low_Mopt
            vertical_intersect_y = slope_opt * threat_point_x + b_opt
            
            # Horizontal line intersection (y = threat_point_y)
            horizontal_intersect_x = (threat_point_y - b_opt) / slope_opt
    

        # Keeping SR constant and plotting through MR to MU (Curve 1)
        plt.figure(figsize=(10, 6))
        plt.plot(V_1_Low[:,0], V_1_Low[:,1], label='Curve 1 $S^R$', color='blue')
        plt.plot(V_2_High[:,0], V_2_High[:,1], label='Curve 2 $S^U$', color='red')
        #plt.plot(V_CP[:,0], V_CP[:,1], linestyle ="--", label='Capture Price', color='orange')

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
        """
        # Add single "M increasing" label for each curve
        plt.annotate('Contract Amount increasing', 
                    xy=(V_1_Low[len(V_1_Low)//2,0], V_1_Low[len(V_1_Low)//2,1]),
                    xytext=(30, 30), textcoords='offset points',
                    color='blue', fontsize=10)
        plt.annotate('Contract Amount increasing',
                    xy=(V_2_High[len(V_2_High)//2,0], V_2_High[len(V_2_High)//2,1]),
                    xytext=(30, 30), textcoords='offset points',
                    color='red', fontsize=10)
        """
        # Hopefully this should be at the intersection points 
        # Plot the points
        
        if cond_MR==True:
            # Plot Optimal Contract Amount Point with fixed price SR and SU
            if self.data.contract_type == "PAP":
                plt.scatter(UG_Low_Mopt, UL_Low_Mopt, color='green', marker='o', s=100,    label=fr'V1 $\gamma$ = {100*M_SR:.2f}%, M* = ({self.data.generator_contract_capacity * M_SR:.2f} MW)')
                plt.scatter(UG_High_Mopt, UL_High_Mopt, color='green', marker='*', s=150,     label=fr'V1 $\gamma$ = {100*M_SU:.2f}%, M* = ({self.data.generator_contract_capacity * M_SU:.2f} MW)')


                # --- PARAMETERS ---------------------------------------------------------
                n_pts   = 80                                          # resolution

                # helper: utility pair for ANY (γ,S)
                def _U_pair(gam, S):
                    return self.Utility_G(S, gam), self.Utility_L(S, gam)

                # sample strikes from SR* to SU*
                S_span = np.linspace(self.BS_strike_min, self.BS_strike_max, n_pts)
                UG_path, UL_path = [], []

                for S_val in S_span:
                    # quick grid search for γ*(S_val) that maximises Nash product
                    γ_grid  = np.linspace(0.0, 1.0, 50)
                    best_K  = -np.inf
                    best_ug = best_ul = None
                    for γ in γ_grid:
                        ug, ul = _U_pair(γ, S_val)
                        K_val  = (ug - self.data.Zeta_G)*(ul - self.data.Zeta_L)
                        if K_val > best_K:
                            best_K, best_ug, best_ul = K_val, ug, ul
                    UG_path.append(best_ug)
                    UL_path.append(best_ul)

                """
                elif method == "bezier":
                    # simple *visual* curve: quadratic Bézier using midpoint control
                    ctrl_x = (x1 + x2)/2
                    ctrl_y = max(y1, y2) + 0.6*(max(y1, y2)-min(y1, y2))   # lift a bit
                    t      = np.linspace(0, 1, n_pts)
                    UG_path = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
                    UL_path = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
                """

                # ------------------------------------------------------------------------
                plt.plot(UG_path, UL_path,
                        color="darkorange",  linestyle="-.", linewidth=2,
                        label="curvilinear link")
            else:
                plt.scatter(UG_Low_Mopt, UL_Low_Mopt, color='green', marker='o', s=100, label=f'V1 M* = ({M_SR /8760*1e3:.2f} MWh)')
                plt.scatter(UG_High_Mopt, UL_High_Mopt, color='green', marker='*', s=150, label=f'V2 M* = ({M_SU/8760*1e3:.2f} MWh)')

            # Draw straight line between optimal points
            plt.plot([UG_Low_Mopt, UG_High_Mopt], 
                    [UL_Low_Mopt, UL_High_Mopt], 
                'g--', 
                label='Line between optimal points',
                alpha=0.3)  # alpha makes line slightly transparent        # Plot Nash bargaining terms
            """
            # Assume V_1_Low and V_2_High are (n,2) arrays: [:,0] = UG, [:,1] = UL
            dA = self.data.Zeta_G
            dB = self.data.Zeta_L

            # 1) Compute Nash product for both curves
            nash_product_low = (V_1_Low[:,0] - dA) * (V_1_Low[:,1] - dB)
            nash_product_high = (V_2_High[:,0] - dA) * (V_2_High[:,1] - dB)

            # 2) Locate the Nash point (maximum Nash-product) for each curve
            idx_N_low = np.argmax(nash_product_low)
            uA_N_low = V_1_Low[idx_N_low, 0]
            uB_N_low = V_1_Low[idx_N_low, 1]
            K_star_low = nash_product_low[idx_N_low]

            idx_N_high = np.argmax(nash_product_high)
            uA_N_high = V_2_High[idx_N_high, 0]
            uB_N_high = V_2_High[idx_N_high, 1]
            K_star_high = nash_product_high[idx_N_high]

            # 3) Generate the iso-Nash hyperbola that passes through the Nash point (for V_1_Low)
            uA_iso_low = np.linspace(dA + 1e-3, uA_N_low + 3, 400)
            uB_iso_low = dB + K_star_low / (uA_iso_low - dA)

            # (Optional) For V_2_High as well
            uA_iso_high = np.linspace(dA + 1e-3, uA_N_high + 3, 400)
            uB_iso_high = dB + K_star_high / (uA_iso_high - dA)

            # Plotting example for V_1_Low
            plt.scatter(uA_N_low, uB_N_low, color='orange', marker='o', s=100, label='Nash Point (Low)')
            plt.plot(uA_iso_low, uB_iso_low, 'k--', label='Iso-Nash Hyperbola (Low)')

            # Plotting example for V_2_High
            plt.scatter(uA_N_high, uB_N_high, color='green', marker='o', s=100, label='Nash Point (High)')
            plt.plot(uA_iso_high, uB_iso_high, 'g--', label='Iso-Nash Hyperbola (High)')
            """
            # Plot Utility G and L for the contract at optimal solution 
            #plt.scatter(self.results.utility_G, self.results.utility_L, color='orange', marker='o', s=100, label='Optimal Solution (G,L)')
            #Plot Results from Gurobi resulting utility 
            plt.scatter(self.results.utility_G, self.results.utility_L, color='red', marker='o', s=100, label='Optimization Result (G,L)')
           
            
            plt.scatter(threat_point_x, vertical_intersect_y, 
                    color='purple', marker='x', s=100,) 
                    #label='Vertical Line Intersection')
            plt.scatter(horizontal_intersect_x, threat_point_y, 
                    color='purple', marker='x', s=100,) 
                    #label='Horizontal Line Intersection')
            # Plot intersection points
            if cond_MU == True:
            
                vertices = np.array([
                [threat_point_x, threat_point_y],        # Start at threat point
                [threat_point_x, vertical_intersect_y],  # Go right to first optimal point
                [horizontal_intersect_x, threat_point_y], # Go to second optimal point      # Go down vertically
                [threat_point_x, threat_point_y]    # Back to start
                ])
            else:
                vertices = np.array([
                [threat_point_x, threat_point_y],        # Start at threat point
                [threat_point_x, vertical_intersect_y],  # Go right to first optimal point
                [horizontal_intersect_x, threat_point_y], # Go to second optimal point      # Go down vertically
                [threat_point_x, threat_point_y]    # Back to start
                ])

            
            # Create and add the polygon
            polygon = plt.Polygon(vertices, facecolor='gray', label = "Barter Set Region", alpha=0.2, edgecolor=None)
            plt.gca().add_patch(polygon)
      
        plt.axvline(x=threat_point_x, color='black', linestyle='--', alpha=0.3)
        plt.axhline(y=threat_point_y, color='black', linestyle='--', alpha=0.3)
        # Original threat point plotting
        plt.scatter(threat_point_x, threat_point_y, color='black', marker='o', s=100, label='Disagreement point')
        #line_length = 20
        #x_vals_45 = np.array([threat_point_x, threat_point_x + line_length])
        #y_vals_45 = np.array([threat_point_y, threat_point_y + line_length])
        #plt.plot(x_vals_45, y_vals_45, color='black', linestyle=':', linewidth=2, label='45° Line from Threatpoint')


        plt.xlabel('Utility G')
        plt.ylabel('Utility L')

        plt.title(f'Barter Set Type: {self.data.contract_type} A_G={self.data.A_G:.2f}, A_L={self.data.A_L:.2f}, K_G={self.data.K_G_lambda_Sigma:.2f}, K_L={self.data.K_L_lambda_Sigma:.2f}')

        plt.legend()
        plt.grid()
        plt.show()

   
        print("Done")

    def plot_utility_contours(self):
        """
        Create a contour plot of Utility G (UG) and Utility L (UL) 
        as functions of strike price and contract amount.
        """
        # Define grid for strike price and contract amount
        S_grid = np.linspace(self.data.strikeprice_min, self.data.strikeprice_max, 200)
        if self.data.contract_type == "PAP":
            M_grid = np.linspace(1, 1, 200)
        else:
            M_grid = np.linspace(self.data.contract_amount_min, self.data.contract_amount_max, 200)
        S_mesh, M_mesh = np.meshgrid(S_grid, M_grid)

        UG_mesh = np.zeros_like(S_mesh)
        UL_mesh = np.zeros_like(S_mesh)

        Delta_UG_mesh = np.zeros_like(S_mesh)
        Delta_UL_mesh = np.zeros_like(S_mesh)


        # Compute utilities on the grid
        for i in range(S_mesh.shape[0]):
            for j in range(S_mesh.shape[1]):
                UG_mesh[i, j] = self.Utility_G(S_mesh[i, j], M_mesh[i, j])
                UL_mesh[i, j] = self.Utility_L(S_mesh[i, j], M_mesh[i, j])

        Delta_UG_mesh = UG_mesh - self.data.Zeta_G
        Delta_UL_mesh = UL_mesh - self.data.Zeta_L      
        nash_product_mesh = (UG_mesh - self.data.Zeta_G) * (UL_mesh - self.data.Zeta_L)

        max_idx = np.unravel_index(np.argmax(nash_product_mesh), nash_product_mesh.shape)
        max_S = S_mesh[max_idx] * 1e3  # If you plot S_mesh*1e3
        max_M = M_mesh[max_idx]
        max_val = nash_product_mesh[max_idx]
        print(f"Maximum Nash product: {max_val:.4f} at Strike Price: {max_S:.4f}, Contract Amount: {max_M:.4f}")

        
            
        # Plot contours
        fig, ax = plt.subplots(figsize=(10, 8))
        CS = ax.contour(S_mesh*1e3, M_mesh, UG_mesh, levels=10, cmap='Blues', linestyles='solid')
        CL = ax.contour(S_mesh*1e3, M_mesh, UL_mesh, levels=10, cmap='Reds', linestyles='dashed')
        #CS_threat_G = ax.contour(S_mesh*1e3, M_mesh, UG_mesh, levels=[self.data.Zeta_G], colors='black', linestyles='dashdot', linewidths=2)
        #ax.clabel(CS_threat_G, fmt={self.data.Zeta_G: 'UG = Zeta_G'}, inline=True, fontsize=10)

        # Add contour line for UL = Zeta_L
        #CL_threat_L = ax.contour(S_mesh*1e3, M_mesh, UL_mesh, levels=[self.data.Zeta_L], colors='gray', linestyles='dotted', linewidths=2)
        #ax.clabel(CL_threat_L, fmt={self.data.Zeta_L: 'UL = Zeta_L'}, inline=True, fontsize=10)
        #plt.scatter(self.data.Zeta_G, self.data.Zeta_L, color='black', marker='o', s=100, label='Threatpoint')
        #plt.scatter(self.results.utility_G, self.results.utility_L, color='orange', marker='o', s=100, label='Optimization Result (G,L)')
        ax.clabel(CS, inline=True, fontsize=10, fmt="UG: %.1f")
        ax.clabel(CL, inline=True, fontsize=10, fmt="UL: %.1f")
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Contract Amount")
        ax.set_title("Contour Plot of Utility G (solid blue) and Utility L (dashed red)")
        plt.grid(True, alpha=0.3)
        plt.show()


        fig, ax = plt.subplots(figsize=(10, 8))
        CS = ax.contour(S_mesh*1e3, M_mesh, Delta_UG_mesh, levels=10, cmap='Blues', linestyles='solid')
        CL = ax.contour(S_mesh*1e3, M_mesh, Delta_UL_mesh, levels=10, cmap='Reds', linestyles='dashed')
        #plt.scatter(self.results.utility_G-self.data.Zeta_G, self.results.utility_L-self.data.Zeta_L, color='orange', marker='o', s=100, label='Optimization Result Delta (G,L)')
        ax.clabel(CS, inline=True, fontsize=10, fmt="UG: %.1f")
        ax.clabel(CL, inline=True, fontsize=10, fmt="UL: %.1f")
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Contract Amount")
        ax.set_title("Contour Plot of Delta Utility G (solid blue) and Delta Utility L (dashed red)")
        plt.grid(True, alpha=0.3)
        plt.show()

         # Plot Nash product contour
        fig, ax = plt.subplots(figsize=(10, 8))
        CS = ax.contourf(S_mesh*1e3, M_mesh, nash_product_mesh, levels=20, cmap='viridis')
        plt.colorbar(CS, ax=ax, label='Nash Product')
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Contract Amount")
        ax.set_title("Nash Product Contour (UG - Zeta_G) * (UL - Zeta_L)")
        #f self.data.contract_type == "PAP":
        #    plt.scatter(self.results.strike_price,self.results.gamma, color='orange', marker='o', s=100, label='Optimization Result')
        #else:
        #    plt.scatter(self.results.strike_price, self.results.contract_amount, color='orange', marker='o', s=100, label='Optimization Result')
        # Plot the maximum point
        ax.scatter(max_S, max_M, color='red', marker='*', s=200, label='Max Nash Product')
        ax.legend()
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_utility_cvar_vs_M(self, strike_price='min'):
        """
        Diagnostic plot: show how Expected, CVaR, and Utility evolve with contract amount M
        for fixed strike price (S = strikeprice_min or strikeprice_max).
        """
        if strike_price == 'min':
            S = self.BS_strike_min
            title_suffix = "S*1e3 = S^R (Min) EUR/MWh"
        elif strike_price == 'max':
            S = self.BS_strike_max
            title_suffix = "S*1e3 = S^U (Max) EUR/MWh"
        elif isinstance(strike_price, (float, int)):
            S = strike_price
            title_suffix = f"S = {S*1e3:.2f} EUR/MWh"
        else:
            raise ValueError(f"Invalid strike_price: {strike_price}")
        
        if self.data.contract_type == "PAP":
            # For PAP, we need to calculate the contract amount as a percentage of production
            M_space = np.linspace(0, 1, self.n)
        else:
            M_space = np.linspace(self.data.contract_amount_min, self.data.contract_amount_max, self.n)

        UG_exp = []
        UG_cvar = []
        UG_total = []

        UL_exp = []
        UL_cvar = []
        UL_total = []


        for M in M_space:
            # Contract revenue per scenario

            if self.data.contract_type == "PAP":
                # generator 
                earnings_G = ((1-M) * self.data.production * self.data.price_G * self.data.capture_rate + M * self.data.production * S).sum(axis=0)

                CVaR_G = calculate_cvar_left(earnings_G, self.data.alpha)
                Utility_G = (1-self.data.A_G)*earnings_G.mean() + self.data.A_G * CVaR_G
                # Load contract revenue
                EuL = (-self.data.price_L * self.data.load_CR * self.data.load_scenarios).sum(axis=0)
                SML = (M * self.data.production * self.data.price_L * self.data.capture_rate - M * S * self.data.production).sum(axis=0)
                earnings_L = EuL + SML
                CVaR_L = calculate_cvar_left(earnings_L, self.data.alpha)
                Utility_L = (1-self.data.A_L)*earnings_L.mean() + self.data.A_L * CVaR_L
            else:
                rev_contract_G = M  * (S  - self.data.price_G)
                rev_contract_total_G = rev_contract_G.sum(axis=0)
                Expected_G = self.data.net_earnings_no_contract_priceG_G 
                earnings_G = Expected_G + rev_contract_total_G
                CVaR_G = calculate_cvar_left(earnings_G, self.data.alpha)
        
                Utility_G = (1-self.data.A_G)*earnings_G.mean() + self.data.A_G * CVaR_G
            
        
                rev_contract_L = M  * ( self.data.price_L - S )
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

       