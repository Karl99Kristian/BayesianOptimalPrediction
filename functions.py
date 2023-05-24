from utils import norm
from typing import Literal
import numpy as np

def trans_dens(t,x,y):
    """Retrun transition density of Brownian motion reflected at 0"""
    return 1/(np.sqrt(2*np.pi*t))*(np.exp(-(x-y)**2/(2*t))+np.exp(-(x+y)**2/(2*t)))

def posterior_process(t,x):
    """Retrun the postirior process at time t on x"""
    return 2*norm.cdf(x/np.sqrt(1-t))-1

class Os_problem():
    """Class to store defining functions of the particular Optimal stopping problem"""
    def __init__(self,loss:Literal["LINEX","Mixed_analytic", "Mixed"]) -> None:
        self.loss = loss

    def mayer(self,t,x,a):
        if self.loss == "LINEX":
            return np.exp(a*x)*posterior_process(t,x)+2*np.exp((a**2)*(1-t)/2)*(1-norm.cdf((x-a*(1-t))/np.sqrt(1-t)))
        elif self.loss == "Mixed_analytic":
            return (1-t)*(1-posterior_process(t,x))
        elif self.loss == "Mixed":
            return 1-posterior_process(t,x)

    def inf_gen_mayer(self,t,x,a):
        if self.loss == "LINEX":
            return a**2/2*(np.exp(a*x)*posterior_process(t,x)-2*np.exp((a**2)*(1-t)/2)*(1-norm.cdf((x-a*(1-t))/np.sqrt(1-t))))
        elif self.loss == "Mixed_analytic":
            return posterior_process(t,x)-1
        elif self.loss == "Mixed":
            return 0
        
    def lagrange(self,t,x,a):
        if self.loss == "LINEX":
            return 0
        elif self.loss == "Mixed_analytic":
            return a*posterior_process(t,x)
        elif self.loss == "Mixed":
            return a*posterior_process(t,x)

    def mayer_at_mat(self,a,x):
        if self.loss == "LINEX":
            return  np.exp(a*x)
        elif self.loss == "Mixed_analytic":
            return 0
        elif self.loss == "Mixed":
            return 0

    def inf_gen_mayer_at_mat(self,a,x):
        if self.loss == "LINEX":
            return a**2/2*np.exp(a*x)
        elif self.loss == "Mixed_analytic":
            return -1
        elif self.loss == "Mixed":
            return 0
    