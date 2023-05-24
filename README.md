# Bayesian Optimal Prediction
Implementation of numerical procedures in Bayesian Optimal Prediction[1].

The code is mainly written for plot generation for [1]. Thus, the code has not been optimized further than for this goal.

## Free-boundary equation
The file `main_fbe.py` calculates the boundary function $b$ and the value function $V$ for a given `Os_problem` which is to be defined in `functions.py`. An `Os_problem` consists of a name, Mayer functional $M$, a function $\overline{M}$ which is the infinitesimal generator applied to the Mayer functional(i.e. $\mathbb{L}_Z M=\overline{M}$), and a Lagrange functional $L$. For the $M$ and $\overline{M}$ a limiting function at maturity should also be provided.
When `calc_b=True` the file calculates $b$ iteratively in 
<!-- ![equation](https://latex.codecogs.com/svg.image?&space;\begin{aligned}&space;&space;&space;&space;\int_0^\infty&space;M(T,y)p(1-t;b(t),y)dy&space;&space;&space;&space;=&M(t,b(t))\\&space;&space;&space;&space;&&plus;\int_0^{T-t}&space;\int_{b(t&plus;s)}^\infty&space;\overline{M}(t&plus;s,y)p(s;b(t),y)dyds\\&space;&space;&space;&space;&-\int_0^{T-t}&space;\int_0^{b(t&plus;s)}&space;L(t&plus;s,y)p(s;b(t),y)dyds\end{aligned}) -->
<!-- $$
\begin{aligned}
    \int_0^\infty M(T,y)p(1-t;b(t),y)dy
    =&M(t,b(t))\\
    &+\int_0^{T-t} \int_{b(t+s)}^\infty \overline{M}(t+s,y)p(s;b(t),y)dyds\\
    &-\int_0^{T-t} \int_0^{b(t+s)} L(t+s,y)p(s;b(t),y)dyds
\end{aligned}
$$ -->
$$
    \int_0^\infty M(T,y)p(1-t;b(t),y)dy=M(t,b(t))+\int_0^{T-t} \int_{b(t+s)}^\infty \overline{M}(t+s,y)p(s;b(t),y)dyds
$$
$$
    -\int_0^{T-t} \int_0^{b(t+s)} L(t+s,y)p(s;b(t),y)dyds
$$ 
where $p(s,x,y)$ is the transition density of a Brownian motion reflected in $0$. 

When `calc_v=True` then the file calculates $V$ in 
<!-- ![equation](https://latex.codecogs.com/svg.image?\begin{aligned}&space;&space;&space;&space;W_*(t,x)&space;=&&space;\int_0^\infty&space;M(T,y)p(1-t;b(t),y)dy\\&space;&space;&space;&space;&-\int_0^{T-t}&space;\int_{b(t&plus;s)}^\infty&space;\overline{M}(t&plus;s,y)p(s;b(t),y)dyds\\&space;&space;&space;&space;&&plus;\int_0^{T-t}&space;\int_0^{b(t&plus;s)}&space;L(t&plus;s,y)p(s;b(t),y)dyds\end{aligned}) -->
<!-- $$
\begin{aligned}
    W_*(t,x) =& \int_0^\infty M(T,y)p(1-t;b(t),y)dy\\
    &-\int_0^{T-t} \int_{b(t+s)}^\infty \overline{M}(t+s,y)p(s;b(t),y)dyds\\
    &+\int_0^{T-t} \int_0^{b(t+s)} L(t+s,y)p(s;b(t),y)dyds
\end{aligned}
$$ -->
$$
W_*(t,x) = \int_0^\infty M(T,y)p(1-t;b(t),y)dy
    -\int_0^{T-t} \int_{b(t+s)}^\infty \overline{M}(t+s,y)p(s;b(t),y)dyds
$$
$$
    +\int_0^{T-t} \int_0^{b(t+s)} L(t+s,y)p(s;b(t),y)dyds
$$
as described in Appendix A.1 in [1].

## Risk calculations
The files `risk_space.py` and `risk_time.py` calculate Monte-Carlo approximations of 
$$\mathbb{E}\left[c(\tau-\theta)+(\tau-\theta)\mid \theta = t\right]$$
and 
$$\mathbb{E}\left[e^{\alpha(S_1-\hat{S})}-\alpha(S_1-\hat{S})-1\mid S_1=s\right]$$
using accept-reject sampling as described in Appendix A.2 in [1].

### Setup
To run the code make a virtual environment and install dependencies.
```bash
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```


### References
[1]: Engelund, K. K. (2023). *Bayesian Optimal Prediction - A Bayesian approach to Optimal Prediction of the ultimate maximum with asymmetric loss functions*, Master’s Thesis in Actuarial Science, University of Copenhagen.