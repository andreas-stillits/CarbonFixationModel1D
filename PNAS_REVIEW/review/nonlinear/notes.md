### Notes on nonlinear implementation 

We require $\tau, \gamma$ to have the same definition and thus arrive at the problem:

$$
\partial_t \chi = \nabla \cdot (\delta(\lambda) \nabla \chi) - \tau^2 \kappa(\lambda) \frac{\chi - \chi^*}{1 + \mu \chi} 
$$

Where we have defined: $\mu = C_a/K_m$. This implies that a factor $K_m$ is implicitly absorbed into our $K(z)$ but that was anyhow the case in the linearized version.

$$
V_c \frac{C-\Gamma}{C+K_m} - R_d \simeq \frac{V_c}{\Gamma+K_m}(C-\Gamma) - R_d = \frac{V_c}{\Gamma+K_m}(C-C^*)
$$

Where we have defined: $C^* = \Gamma + R_d \frac{\Gamma + K_m}{V_c}$

In litterature we find the following ranges for $C_a, K_m$:
- $C_a$:
    - atmospheric around 400 ppm
    - max in gas exchange around ... ppm

- $K_m$:
    - ...-... ppm