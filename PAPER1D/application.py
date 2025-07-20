import numpy as np
import matplotlib.pyplot as plt
import streamlit as st 

gs = 0.1  #mol/m^2/s
K  = 4.2   #Hz
L  = 300.0 #µm
D  = 0.016 #cm^2/s #10% of the diffusivity of CO2 in air
Ca = 250.0 #ppm
C_ = 40.0  #ppm

gs = st.number_input(r'Specify stomatal conductance $g_s$ in units of $\frac{mol}{m^2 s}$', value = gs, min_value=0.01, max_value=2.0, step=0.01, format='%.2f')
K  = st.number_input(r'Specify the reactivity $K$ in units of $Hz$', value = K, min_value=0.04, max_value=40.0, step=0.2, format='%.2f')
L  = st.number_input(r'Specify the mesophyll thickness $L$ in units of $\mu m$', value = L, min_value=150.0, max_value=800.0, step=25.0, format='%.0f')
D  = st.number_input(r'Specify the diffusivity $D$ in units of $\frac{cm^2}{s}$', value = D, min_value=0.002, max_value=0.16, step=0.002, format='%.3f')
C_ = st.number_input(r'Specify the apparent compensation point $C^*$ in units of $ppm$', value = C_, min_value=20.0, max_value=100.0, step=5.0, format='%.0f')
Ca = st.number_input(r'Specify the atmospheric CO2 concentration $C_a$ in units of $ppm$', value = Ca, min_value=C_+0.01, max_value=500.0, step=25.0, format='%.0f')

# transfer to SI units

L  = L*1e-6     # m
D  = D*1e-4     # m^2/s
gs = gs*0.02241 # m/s

# Correct for the units here
tau = np.sqrt(K*L**2/D)
gamma = (gs)/(D/L)
zeta_ = C_/Ca

st.write(r'The corresponding dimensionless parameters are:  $\tau = \sqrt{\frac{KL^2}{D}} = $', f'{tau:.2e}', r'and $\gamma = \frac{g_s L}{D} = $', f'{gamma:.2e}')

# calculate distributions

x = np.linspace(0, 1, 100)
zeta = lambda x: zeta_ + (1-zeta_)/(1 + tau*np.tanh(tau)/gamma) * (np.cosh(tau*(1 - x)))/(np.cosh(tau))



#_____________________________________________________
fs = 14


def draw_rescaled_profile(ax):
    ax.plot(x, zeta(x), color='forestgreen', linewidth=2)
    ax.plot([-0.1, 0], [1, zeta(0)], color='forestgreen', linewidth=2)
    ax.set_xlabel(r'$\lambda = z/L$', fontsize=fs)
    ax.set_ylabel(r'$\zeta(\lambda) = C/C_a$', fontsize=fs)
    ax.set_title(r'Absolute CO$_2$ profile', fontsize=fs)
    ax.set_xlim(-0.1, 1)
    ax.set_ylim(0, 1)
    ax.fill_betweenx([0, 1], -0.1, 0, color='lightgray', alpha=0.3, label='stomatal region')
    ax.fill_betweenx([0, 1], 0, 1, color='lightgreen', alpha=0.3, label='mesophyll region')
    ax.legend()


def draw_phase_diagram(ax):
    taus = np.exp(np.linspace(np.log(0.01), np.log(100), 400))
    border_x = lambda taus, x, s=1: taus*np.tanh(taus)*(1/x**2 - 1)**(s*1/4)

    # the xs are chosen to match 10:1, 3:1, 1:1 relative strengths.
    # In terms of ci/ca the bonds correspond to {80, 70, 57, 45, 35}
    for x in [0.995, 0.95, 1/np.sqrt(2)]:
        ax.fill_between(taus, border_x(taus, x, s=1), 0.01, color='burlywood', alpha=0.5)
        ax.fill_between(taus[taus <= 1], 100, border_x(taus[taus <= 1], x, s=-1), color='mediumseagreen', alpha=0.5)
        ax.fill_between(taus[taus >= 1], 100, border_x(taus[taus >= 1], x, s=-1), color='salmon', alpha=0.5)

    ax.text(0.04, 2, r'    $A_N \propto KL$'+'\n single cells', fontsize=12, color='white')
    ax.text(7, 0.1, r'  $A_N \propto g_s$'+'\n stomata', fontsize=12, color='white')
    ax.text(1.5, 30, r'    $A_N \propto \sqrt{KD}$'+'\n IAS diffusion', fontsize=12, color='white')
    ax.set_xlabel(r'internal balance $\tau = \sqrt{\frac{KL^2}{D}}$', fontsize=14)
    ax.set_ylabel(r'influx balance $\gamma = \frac{g_s L}{D}$', fontsize=14)
    ax.set_title('Phase diagram', fontsize=14)
    ax.plot([1, 1], [0.01, 100], color='gray', linestyle='--')
    ax.plot([0.01, 100], [1, 1],  color='gray', linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.01, 100)
    ax.plot(tau, gamma, 'ko', markersize=8)

#_____________________________________________________

#fig1, ax1 = plt.subplots(1, 1, figsize=(8,4))
#fig2, ax2 = plt.subplots(1, 1, figsize=(5,5))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

draw_rescaled_profile(ax1)
draw_phase_diagram(ax2)

st.pyplot(fig)
#st.pyplot(fig1)
#st.pyplot(fig2)
