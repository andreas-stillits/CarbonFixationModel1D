# Functionality to solve the 1D leaf model with hypostomatous boundary conditions

import numpy as np
import matplotlib.pyplot as plt


class Leaf:
    def __init__(self, tau, gamma, chi_, rho=(1, 1, 0.5), resolution=500):
        self.tau        = tau                           # absorption balance
        self.gamma      = gamma                         # transport balance
        self.chi_       = chi_                          # CO2 compensation point relative to atmospheric CO2 conc.
        self.rho        = rho                           # triplet encoding spongy/palisade mesophyll proportions (D_PM/D_SM, K_SM/K_PM, L_PM/L)
        self.resolution = resolution                    # number of points to cast solution over
        self.boundary   = int(resolution*(1-rho[2]))    # index of spongy/palisade mesophyll interface
        #
        self.deltas     = np.zeros(self.resolution)     # array for rescaled effective IAS diffusivity D/<D>
        self.kappas     = np.zeros(self.resolution)     # array for rescaled absorption capacity K/<K>
        self.solution   = np.zeros(self.resolution)     # array to contain the solution - CO2 as function of depth
        self.alpha      = 0                             # rescaled assimilation rate An/(D*Ca/L)
        # END


    def calculate_delta_and_kappa_distributions(self):
        '''  
        construct delta and kappa distributions
        '''
        self.deltas = np.zeros(self.resolution)
        self.kappas = np.zeros(self.resolution)
        rho_delta, rho_kappa, rho_lambda = self.rho
        # finding percentage volumes occupied by the different comparments
        boundary = int(self.resolution*(1-rho_lambda))
        # calculate dimensionless scalars
        delta_sm = 1/((1-rho_lambda) + rho_lambda*rho_delta)
        delta_pm = rho_delta*delta_sm
        #
        kappa_pm = 1/((1-rho_lambda)*rho_kappa + rho_lambda)
        kappa_sm = rho_kappa*kappa_pm
        # fill in arrays
        self.deltas[:boundary] = delta_sm
        self.deltas[boundary:] = delta_pm
        #
        self.kappas[:boundary] = kappa_sm 
        self.kappas[boundary:] = kappa_pm
        self.boundary = boundary
        # END 


    def calculate_steady_state_solution(self):
        self.calculate_delta_and_kappa_distributions()
        self.solution = np.zeros(self.resolution)
        rho_delta, rho_kappa, rho_lambda = self.rho
        # calculate fractional indexes
        kappa_sm   = self.kappas[0]
        kappa_pm   = self.kappas[-1]
        delta_sm   = self.deltas[0]
        delta_pm   = self.deltas[-1]
        # calculate hyperbolic growth rates
        beta_sm   = self.tau*np.sqrt(kappa_sm/delta_sm)
        beta_pm   = self.tau*np.sqrt(kappa_pm/delta_pm)
        # calculate exponential coefficients
        exp_sm = np.exp(beta_sm*(1-rho_lambda))
        exp_pm = np.exp(beta_pm*(1-rho_lambda))
        # setup coefficient matrix (A_sm, B_sm, A_pm, B_sm) which are the exponential coefficients of the domain solutions
        matrix = np.array([[beta_sm - self.gamma/delta_sm,  -beta_sm - self.gamma/delta_sm,   0                      ,   0                      ], # continuous flux at stomata
                           [exp_sm                       ,   1/exp_sm                     ,  -exp_pm                 ,  -1/exp_pm               ], # continuous solution at boundary
                           [delta_sm*beta_sm*exp_sm      ,  -delta_sm*beta_sm/exp_sm      ,  -delta_pm*beta_pm*exp_pm,   delta_pm*beta_pm/exp_pm], # continuous flux at boundary 
                           [0                            ,   0                            ,   beta_pm*np.exp(beta_pm),  -beta_pm/np.exp(beta_pm)]])# vanishing  flux at epidermis        
        #
        target_vector = np.array([-self.gamma/delta_sm*(1-self.chi_), 0, 0, 0])
        coefficients = np.matmul(np.linalg.inv(matrix), target_vector)
        #
        A_sm, B_sm, A_pm, B_pm = coefficients
        # create domain and sub-domains
        domain = np.linspace(0, 1, self.resolution)
        sm = domain[:self.boundary]
        pm = domain[self.boundary:]
        # fill in the solution
        self.solution[:self.boundary] = self.chi_ + A_sm*np.exp(beta_sm*sm) + B_sm*np.exp(-beta_sm*sm)
        self.solution[self.boundary:] = self.chi_ + A_pm*np.exp(beta_pm*pm) + B_pm*np.exp(-beta_pm*pm)
        # return quantities
        return domain, self.solution
        # END


    def display_solution(self):
        fig = plt.figure(figsize=(10,6))
        domain = np.linspace(0, 1, self.resolution)
        domain_stomata = np.linspace(-0.1, 0, 10)
        stomata_grad = -10*(1-self.solution[0])*domain_stomata + self.solution[0]
        plt.plot(domain_stomata, stomata_grad, '-', color='purple', linewidth=3)
        plt.plot(domain, self.deltas, 'b--', label=r'$\delta (\lambda)$', linewidth=2)
        plt.plot(domain, self.kappas, 'r--', label=r'$\kappa (\lambda)$', linewidth=2)
        plt.plot(domain, self.solution, '-', color='purple', label=r'$\chi(\lambda)$', linewidth=3)
        #
        max = np.max([np.max(self.deltas), np.max(self.kappas)])
        plt.fill_between([0, 1-self.rho[-1]], 0, max, color='green', alpha=0.3)
        plt.fill_between([1-self.rho[-1], 1], 0, max, color='green', alpha=0.6)
        plt.fill_between([-0.1, 0], 0, max, color='grey', alpha=0.3)
        #
        plt.grid(linestyle='-.')
        plt.legend(loc='upper right', fontsize=15)
        plt.xlabel(r'percentage depth $\lambda$', fontsize=20)
        plt.ylabel(r'distributions $\delta, \kappa, \chi$', fontsize=20)
        plt.title(r'Leaf Model Solution', fontsize=20, fontweight='bold')
        plt.show()
