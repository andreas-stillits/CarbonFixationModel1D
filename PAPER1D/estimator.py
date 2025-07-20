import numpy as np 
from tqdm import tqdm
import numpy.random as r 
from matplotlib import pyplot as plt



def estimate_Cis(Ans, dAns, gss, dgss, Ca, sim_size=1000, plot_sample=True):
    '''
    Estimate Cis from quoted An [µmol/m2/s] and gs [mol/m2/s] values
    Ans: An values
    dAns: An errors
    gss: gs values
    dgss: gs errors
    Ca: atmospheric CO2 concentration
    sim_size: number of simulations to run
    plot_sample: if True, will plot a histogram of the Ci distribution for a random sample
    '''
    sample_size = len(Ans)
    Cis = np.zeros(sample_size)
    dCis = np.zeros(sample_size)

    rand = r.randint(0, sample_size)
    for i in range(sample_size):
        Ci_dist = np.zeros(sim_size)
        for j in range(sim_size):
            Ci_dist[j] = -1
            while Ci_dist[j] <= 100:
                An_sim = -1
                while An_sim <= 0:
                    An_sim = r.normal(Ans[i], dAns[i])
                gs_sim = -1
                while gs_sim <= 0:
                    gs_sim = r.normal(gss[i], dgss[i])
                Ci_dist[j] = Ca - An_sim/gs_sim
        if plot_sample and i == rand:
            plt.hist(Ci_dist, bins=25)
            plt.show()
        Cis[i] = np.mean(Ci_dist)
        dCis[i] = np.std(Ci_dist)
    return Cis, dCis



def estimate_gm_star(Ans, dAns, Cis, dCis, C_star, dCstar, sim_size=1000, plot_sample=True):
    '''
    Estoimate gm_star from An [µmol/m2/s], Cis [µmol/mol] and C_star [µmol/mol] values
    Ans: An values
    dAns: An errors
    Cis: Cis values
    dCis: Cis errors
    C_star: C_star value
    dCstar: C_star error
    sim_size: number of simulations to run
    plot_sample: if True, will plot a histogram of the gm_star distribution for a random sample
    '''
    sample_size = len(Ans)
    gm_ = np.zeros(sample_size)
    dgm_ = np.zeros(sample_size)

    rand = r.randint(0, sample_size)
    for i in range(sample_size):
        gm_dist = np.zeros(sim_size)
        for j in range(sim_size):
            gm_dist[j] = -1
            while gm_dist[j] < 0:
                An_sim = -1
                while An_sim < 0:
                    An_sim = r.normal(Ans[i], dAns[i])
                Cis_sim = -1
                while Cis_sim < 0:
                    Cis_sim = r.normal(Cis[i], dCis[i])
                C_star_sim = -1
                while C_star_sim < 0:
                    C_star_sim = r.normal(C_star, dCstar)
                gm_dist[j] = An_sim/(Cis_sim - C_star_sim)
        if plot_sample and i == rand:
            plt.hist(gm_dist, bins=50)
            plt.title('Distribution of gm_star')
            plt.xlabel('gm_star')
            plt.ylabel('Frequency')
            plt.show
        gm_[i] = np.mean(gm_dist)
        dgm_[i] = np.std(gm_dist)

    return gm_, dgm_




def f(gc, gias, gm_):
                return np.sqrt(np.abs(gc*gias/2))*np.tanh(np.sqrt(np.abs(2*gc/gias))) - gm_

def dfdx(gc, gias, gm_):
    return 0.5*(np.sqrt(np.abs(gias/(2*gc)))*np.tanh(np.sqrt(np.abs(2*gc/gias))) + 1/(gias*np.cosh(np.sqrt(np.abs(2*gc/gias)))**2))

def newton(gc0, gias, gm_, step_size = 0.4, max_iterations = 1000, tolerance = 1e-6): #original step_size = 0.1
    gc = gc0
    for i in range(max_iterations):
        f_ = f(gc, gias, gm_)
        if abs(f_) < 1e-6:
            return np.abs(gc)
        df_ = dfdx(gc, gias, gm_)
        if df_ == 0:
            break
        gc = gc - step_size*f_/df_
    print('maxed out all iterations without convergence')


def estimate_principle_parameters(dataframe, sim_samples = 500, plot_sample = True):
    '''
    data_frame: pandas dataframe with columns 'gm_', 'gs', 'g_ias' as well as their errors
    sim_samples: number of samples to draw from the error distributions
    print_example_distributions: if True, will plot example distributions of gamma and tau

    gias is assumed to be defined as D_eff/(0.5*L) as opposed to our own definition lacking the 0.5
    '''
    # unpack data
    # gm* in mol/m2/s
    gm_ = dataframe['gm_'].to_numpy()
    dgm_ = dataframe['dgm_'].to_numpy() 
    # gs in mol/m2/s
    gs = dataframe['gs'].to_numpy()
    dgs = dataframe['dgs'].to_numpy()
    # gias in mol/m2/s
    gias = dataframe['g_ias'].to_numpy()
    dgias = dataframe['dg_ias'].to_numpy()
    # 
    samples = len(gm_) # assume equal for all conductances 
    #
    gammas = np.zeros(samples) # mean
    dgammas = np.zeros((samples,2)) # 16th and 84th percentiles matching +- 1 sigma for a normal distribution
    
    taus = np.zeros(samples)
    dtaus = np.zeros((samples,2))
    
    random_int = r.randint(0,samples) # integer for picking random instance to plot if requested

    for i in tqdm(range(samples)):
        # gamma
        gamma_dist = np.zeros(sim_samples)
        for j in range(sim_samples):
            gs_sim = -1
            while gs_sim <= 0:
                gs_sim = r.normal(gs[i],dgs[i])
            gias_sim = -1
            while gias_sim <= 0:
                gias_sim = r.normal(gias[i],dgias[i])
            gamma_dist[j] = 2*gs_sim/gias_sim
        gammas[i] = np.mean(gamma_dist)
        dgammas[i,0] = np.abs(np.percentile(gamma_dist,16) - gammas[i])
        dgammas[i,1] = np.abs(np.percentile(gamma_dist,84) - gammas[i])
        
        # tau 
        tau_dist = np.zeros(sim_samples)
        for j in range(sim_samples):
            gm_sim = -1
            while gm_sim <= 0:
                gm_sim = r.normal(gm_[i],dgm_[i])
            gias_sim = -1
            while gias_sim <= 0:
                gias_sim = r.normal(gias[i],dgias[i])
            gc0 = gm_[i] # initial guess for gc which will be true in the low tau limit
            gc_sim = newton(gc0, gias_sim, gm_sim)
            tau_dist[j] = np.sqrt(2*gc_sim/gias_sim)
        taus[i] = np.mean(tau_dist)
        dtaus[i,0] = np.abs(np.percentile(tau_dist,16) - taus[i])
        dtaus[i,1] = np.abs(np.percentile(tau_dist,84) - taus[i])

        # plot if requested
        if i == random_int and plot_sample:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            # gamma in ax1
            ax1.hist(gamma_dist, bins=20, color='darkblue');
            ax1.plot([gammas[i], gammas[i]], [0, 100], 'r--')
            ax1.plot([gammas[i] - dgammas[i,0], gammas[i] - dgammas[i,0]], [0, 100], 'g--')
            ax1.plot([gammas[i] + dgammas[i,1], gammas[i] + dgammas[i,1]], [0, 100], 'g--')
            ax1.set_title('Example distribution of individual gamma sim.')
            # tau in ax2
            ax2.hist(tau_dist, bins=20, color='darkblue');
            ax2.plot([taus[i], taus[i]], [0, 100], 'r--')
            ax2.plot([taus[i] - dtaus[i,0], taus[i] - dtaus[i,0]], [0, 100], 'g--')
            ax2.plot([taus[i] + dtaus[i,1], taus[i] + dtaus[i,1]], [0, 100], 'g--')
            ax2.set_title('Example distribution of individual tau sim.')
            plt.show()
    # add to dataframe
    dataframe['gamma'] = gammas
    dataframe['dgamma_low'] = dgammas[:,0]
    dataframe['dgamma_high'] = dgammas[:,1]
    dataframe['tau'] = taus
    dataframe['dtau_low'] = dtaus[:,0]
    dataframe['dtau_high'] = dtaus[:,1]
    #
    return dataframe, (taus, dtaus), (gammas, dgammas)