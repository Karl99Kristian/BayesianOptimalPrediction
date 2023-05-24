##########################
# Calculate risk in time #
##########################

#imports
from utils import *
import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
from matplotlib import pyplot as plt


#### Options
identifyer = "_test2"   # Identifyer for file names
# np.random.seed(2023)  # Set a seed  

# Variance reduction flags
accept_reject = True    # Use accept-reject for variance reduction
antithetic = False      # Use antithetic samples for variance reduction

# Simultaion params
N=100000    # Number of paths pr simulation iteration
n=500       # Number of time steps
Nk = 700    # Number of elements in strata

# Risk params
rf = "wl"   # Name of loss function: Literal["wl","ent"]
c = 1       # Loss function parameter

#Accuracy
accuracy = True # Test different variance reductions


# Functions
def simulate_paths(N,n,antithetic=False):
    pth = np.random.normal(0,1/np.sqrt(n),(n+1,N))
    
    # Variance reduction
    if antithetic:
        pth2 = np.zeros((n+1,2*N))
        print(pth2.shape)
        print(N)
        pth2[:,:N] = pth
        pth2[:,N:] = -pth
        pth=pth2
    
    pth[0,:]=0
    pth=pth.cumsum(0)

    return pth

def simulate_path_accept_reject(N,Nk,n):
    res_pth = np.random.normal(0,1/np.sqrt(n),(n+1,Nk*(n+1)))
    res_pth_tensor = np.zeros((n+1,n+1,Nk)) # path, strats, simulations 
    pth_cnt = np.zeros(n)
    temp=0
    while np.sum(pth_cnt)<n*Nk:
        print(temp)
        pth = simulate_paths(N,n)
        thetas_id=np.argmax(pth,0)

        for i,cnt in enumerate(pth_cnt):
            icnt = int(cnt)
            if icnt < Nk:
                idx = partition_theta_idx(i,thetas_id)
                potetntial_pths = pth[:,idx][:,:Nk-icnt]
                num_pot_pths = potetntial_pths.shape[1]
                res_pth_tensor[:,i,icnt:num_pot_pths+icnt]=potetntial_pths
                pth_cnt[i] += num_pot_pths
        temp+=1

    for i in range(n):
        res_pth[:,i*Nk:(i+1)*Nk]=res_pth_tensor[:,i,:]

    return res_pth

def calc_attrib(pth,n):
    thetas_id=np.argmax(pth,0)
    thetas = thetas_id/(n)
    S=np.maximum.accumulate(pth,0)

    return thetas, thetas_id, S

def partition_theta_idx(theta_id, thetas_id):
    return (theta_id == thetas_id)

def stop_time(z, b_res, N, dt_inv):
    rule = np.tile(b_res[::-1],(N,1)).transpose()
    return np.argmax(S-pth>=z*rule,axis=0)/dt_inv

def mc_est(func, stop_times, theta,retvar=False):
    if not retvar:
        return np.mean(func(stop_times,theta))
    else:
        return np.var(func(stop_times,theta))

def risk_wl(c,tau,theta):
    return theta*((tau<=theta).astype(int)-c*(tau>theta))+tau*((c*(tau>theta)-(tau<= theta).astype(int)))

def risk_ent(c,tau,theta):
    return np.log(1-theta)*(c*(tau> theta)-(tau<= theta).astype(int))+np.log(1-tau)*((tau<=theta).astype(int)-c*(tau>theta))

def calculate_risk(trng,thetas_id,pth,S,stop_times,func, retvar=False):
    condexps = np.zeros_like(trng)
    for i, theta in enumerate(trng):
        idx = partition_theta_idx(i,thetas_id)
        sts = stop_times[idx]
        condexps[i] = mc_est(func, sts, theta,retvar)
        
    return condexps

def calc_for_bayes(opt_func,b_res,N,dt_inv,trng,thetas_id,pth,S,func,retvar=False):
    z_x = newton(opt_func,1)
    stop_times_optimal = stop_time(z_x,b_res,N,dt_inv)
    condexps = calculate_risk(trng,thetas_id,pth,S,stop_times_optimal,func)
    if not retvar:
        return condexps, z_x
    else:
        condvar = calculate_risk(trng,thetas_id,pth,S,stop_times_optimal,func,retvar)
        return condexps, condvar
    
# main
if __name__ == "__main__":
    fig,ax = plt.subplots()

    if rf == "wl":
        func = lambda tau,theta: risk_wl(c,tau,theta)
        opt_func = lambda z_x: 4*norm.cdf(z_x)-2*z_x*norm.pdf(z_x)-3+(c-1)/(c+1)
    elif rf == "ent":
        func = lambda tau,theta: risk_ent(c,tau,theta)
        opt_func = lambda z_x: (3-(c-1)/(c+1))/4*quad(lambda x: np.exp(x**2/2),0,z_x)[0]-quad(lambda x: np.exp(x**2/2)*norm.cdf(x),0,z_x)[0]
    else: 
        raise

    # Setup grid for boundary
    dt_inv = n
    dt = 1/dt_inv
    trng = (np.arange(0,dt_inv+1)*dt)[::-1]
    b_res= np.sqrt(1-trng)

    if not accuracy:
        # Simulate and calculate tau
        if not accept_reject:
            pth = simulate_paths(N,n,antithetic)
        else:
            pth = simulate_path_accept_reject(N,Nk,n)
        
        thetas, thetas_id, S = calc_attrib(pth,n)

        # Fix N if variance reduction changes number of sim
        N = pth.shape[1]
        
        zs = np.arange(0.5,1.6,0.15)
        for i, z in enumerate(zs):
            stop_times = stop_time(z,b_res,N,dt_inv)

            condexps=calculate_risk(trng,thetas_id,pth,S,stop_times,func)
            lbl = "_"
            if i == 0 or i==len(zs)-1:
                lbl = f"{z:.02f}"
            ax.plot(trng[:-1],condexps[:-1],label=lbl,alpha=0.7)

        # Calculate for Bayes rule
        condexps,z_x = calc_for_bayes(opt_func,b_res,N,dt_inv,trng,thetas_id,pth,S,func)
        ax.plot(trng[:-1],condexps[:-1], color='black',label="Bayes")

        # Plot options
        if rf == "ent":
            ax.set_ylim(0,4)
        fig.suptitle(rf"Risk in {rf} $c=${c}")
        ax.set_xlabel(r"$\theta$")
        ax.legend(loc='upper center')

        fig.savefig(DIR_PLOTS.joinpath(f"Admis_MC_{rf}_{c}{identifyer}.png"))

    if accuracy:
        fig,ax = plt.subplots(ncols=2)
        # No var reduction
        pth = simulate_paths(2*N,n,False)
        thetas, thetas_id, S = calc_attrib(pth,n)

        condexps, condvar = calc_for_bayes(opt_func,b_res,2*N,dt_inv,trng,thetas_id,pth,S,func,True)
        ax[0].plot(trng[:-1],condexps[:-1],label=f"Standard, N={2*N}")
        ax[1].plot(trng[:-1],condvar[:-1],label=f"Standard, N={2*N}")

        # Synthetic
        pth = simulate_paths(N,n,True)
        thetas, thetas_id, S = calc_attrib(pth,n)

        condexps, condvar = calc_for_bayes(opt_func,b_res,2*N,dt_inv,trng,thetas_id,pth,S,func,True)
        ax[0].plot(trng[:-1],condexps[:-1],label=f"Antithetic, N={N}")
        ax[1].plot(trng[:-1],condvar[:-1],label=f"Antithetic, N={N}")

        # Accept rejcet
        pth = simulate_path_accept_reject(N,Nk,n)
        N=pth.shape[1]
        thetas, thetas_id, S = calc_attrib(pth,n)

        condexps, condvar = calc_for_bayes(opt_func,b_res,N,dt_inv,trng,thetas_id,pth,S,func,True)
        ax[0].plot(trng[:-1],condexps[:-1],label=f"Acc/Rej, N={N}")
        ax[1].plot(trng[:-1],condvar[:-1],label=f"Acc/Rej, N={N}")

        ax[1].legend()
        plt.show()