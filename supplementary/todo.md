## Follow up questions after PNAS reviewers' and editor's comments

Code bases to build:
1. Solver for non-linear non-step distribution of D, K using FEniCSx on a line (equilibrium)
2. Extend solver to do dynamics where (Ca, K, gs) are oscillating
3. Comparison between 0D, 1D, and 3D in speciel cases. What is the overlap between 3D model and 1D model? (R/L << 1)

### 1. Equilibirum solver for general D, K 

We will still use a linar CO2 response. We have argued for it, and it will make simulation much easier. **But we should rewrite the section where we argue to make sure the reasoning is crystal clear and stands out.**

The system is then:

$$
\nabla \cdot (D(z) \cdot \nabla C) = K(z)(C-C^*)
\newline
Using: \nabla \cdot(vV) = v \nabla \cdot V + \nabla v \cdot V
\newline
\int_\Omega \nabla \cdot (D(z) \nabla C) v dx = -\int_\Omega D(z) \nabla C \cdot \nabla v dx + \int_{\partial \Omega} D(z) \nabla C \cdot \hat{n} v ds = \int_\Omega K(z)(C-C^*)
\newline
$$

In rescaled units:
$$
a(\chi,v) = -\int_\Omega \delta(\lambda) \nabla \chi \cdot \nabla v dx \pm \int_{\lambda=0} \gamma \chi v ds - \int_\Omega \tau^2 \kappa(\lambda)\chi v
\newline
L(v) = -\int_\Omega \tau^2 \kappa(\lambda)\chi^*vdx \pm \int_{\lambda=0} \gamma v ds

$$

The pipeline will go as follows:
- make a standard 1D mesh in GMSH and store
- create a library of $\delta, \kappa$ with average unity and SM >< PM respectively 
- parameter search over $\tau, \gamma$ and extract $1-\chi_i$

