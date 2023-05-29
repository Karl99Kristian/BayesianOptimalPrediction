###########################
# Calculate risk in space #
###########################

#imports
from utils import *
import numpy as np
from matplotlib import pyplot as plt


#### Options
identifyer = ""   # Identifyer for file names
# np.random.seed(2023)  # Set a seed    

# Variance reduction
accept_reject = True # Use accept-reject for variance reduction

# Simultaion params
N=400000                            # Number of paths pr simulation iteration
n=500                               # Number of time steps
Nk=1500                             # Number of elements in strata
srng = np.arange(0.15,1.75,1/200)   # Space partition for strata
simple = False                      # Use S(j)(False) or S_k(True)

# Risk params
rf = "LINEX"    # Name of loss function: Literal["LINEX"]...
alpha = 2       # Loss function parameter
mu_x = 0.9512   # Bayes offset

# Plot options
plotdiff = True  # Plot difference to Bayes risk

# Which comparison
compare = ["time_z","time_mu","alt_c","alt_a"][1] # Compare against which?


# Functions
def simulate_paths(N,n):
    pth = np.random.normal(0,1/np.sqrt(n),(n+1,N))
    pth[0,:]=0
    pth=pth.cumsum(0)
    return pth

def simulate_path_accept_reject(N,Nk,n,srng):
    res_pth = np.random.normal(0,1/np.sqrt(n),(n+1,Nk*len(srng)))
    res_pth_tensor = np.zeros((n+1,len(srng),Nk)) # path, strats, simulations 
    pth_cnt = np.zeros(len(srng))
    temp=0
    while np.sum(pth_cnt)<(len(srng)-1)*Nk:
        print(temp,np.min(pth_cnt[1:-1]),"/",Nk)
        pth = simulate_paths(N,n)
        S=np.maximum.accumulate(pth,0)

        for i,cnt in enumerate(pth_cnt[:-1]):
            icnt = int(cnt)
            if icnt < Nk:
                idx = partition_s_idx(srng[i],srng[i+1],S[-1,:])
                potetntial_pths = pth[:,idx][:,:Nk-icnt]
                num_pot_pths = potetntial_pths.shape[1]
                res_pth_tensor[:,i,icnt:num_pot_pths+icnt]=potetntial_pths
                pth_cnt[i] += num_pot_pths
        temp+=1

    for i in range(len(srng)-1):
        res_pth[:,i*Nk:(i+1)*Nk]=res_pth_tensor[:,i,:]

    return res_pth

def calc_attrib(pth,n):
    thetas_id=np.argmax(pth,0)
    thetas = thetas_id/(n)
    S=np.maximum.accumulate(pth,0)

    return thetas, thetas_id, S

def partition_s_idx(s_low,s_high, S):
    return (s_low <= S)*(s_high>S)

def stop_price(z,b_res, N, S, pth, opt=False):
    rule = np.tile(b_res[::-1],(N,1)).transpose()
    t_idx = np.argmax(S-pth>=z*rule,axis=0)
    if opt:
        price = [S[t_idx[i],i] for i in np.arange(N)]        
    else:
        price = [pth[t_idx[i],i] for i in np.arange(N)]
    return np.array(price)

def estimate_linex(alpha, identifyer, dt_inv, N, S, pth):
    res = np.genfromtxt(DIR_DATA.joinpath(f"LINEX_{alpha}_{dt_inv+1}{identifyer}.csv"),delimiter=",")
    b_res=res[:,1]
    return stop_price(1,b_res, N, S, pth)

def estimate_alter(c,dt_inv, N, S, pth):
    res = np.genfromtxt(DIR_DATA.joinpath(f"ALT_{c}_{dt_inv+1}{identifyer}.csv"),delimiter=",")
    b_res=res[:,1]
    return stop_price(1,b_res, N, S, pth,True)

def estimate_time(z,dt_inv,dt, N, S, pth):
    trng = (np.arange(0,dt_inv+1)*dt)[::-1]
    b_res=z*np.sqrt(1-trng)
    return stop_price(z,b_res, N, S, pth)

def mc_est(func, estimates, goal):
    return np.mean(func(estimates, goal))

def risk_linex(alpha,estimate,goal):
    return np.exp(alpha*(goal-estimate))-alpha*(goal-estimate)-1

def calculate_risk(srng,S,estimates,func,simple=True):
    condexps = np.zeros_like(srng)
    for i, s in enumerate(srng[:-1]):
        idx = partition_s_idx(s,srng[i+1],S[-1,:])
        ests = estimates[idx]
        if simple:
            condexps[i] = mc_est(func, ests, s)
        else:
            condexps[i] = mc_est(func, ests, S[-1,idx])
    return condexps

# Main
if __name__ == "__main__":
    fig,ax = plt.subplots()

    if rf == "LINEX":
        func = lambda estimate,goal: risk_linex(alpha, estimate,goal)
    
    title_params = ""

    # Setup grid for boundary
    dt_inv = n
    dt = 1/dt_inv
    res = np.genfromtxt(DIR_DATA.joinpath(f"LINEX_{alpha}_{dt_inv+1}{identifyer}.csv"),delimiter=",")
    trng =res[:,0]
    b_res=res[:,1]

    trng = (np.arange(0,dt_inv+1)*dt)[::-1]
    b_res= np.sqrt(1-trng)


    # Simulate and calculate tau
    if not accept_reject:
        pth = simulate_paths(N,n)
    else:
        pth = simulate_path_accept_reject(N,Nk,n,srng)
    
    # Fix N if variance reduction changes number of sim    
    N = pth.shape[1]
    
    thetas, thetas_id, S = calc_attrib(pth,n)
    

        
    # Calculate for Bayes rule
    estimate_bayes = estimate_linex(alpha, identifyer, dt_inv, N, S, pth)+mu_x
    condexps_bayes = calculate_risk(srng,S,estimate_bayes,func,simple)

    if plotdiff:
        offset = condexps_bayes
        ax.plot(srng[:-1],0*srng[:-1],color="gray")
    else:
        offset = np.zeros_like(condexps_bayes)



    if compare == "time_z":
        mu_x = np.sqrt(2/np.pi)
        title = mu_x
        for z in [1.0,1.3,1.15]:
            estimate = estimate_time(z,dt_inv,dt,N,S,pth)+mu_x
            condexps = calculate_risk(srng,S,estimate,func,simple)
            ax.plot(srng[:-1],condexps[:-1]-offset[:-1],label=f"{z:.2f}")
        title_params = r"$\mu=\sqrt{2/\pi}$"
        ax.legend(title=r"$z$")
    elif compare == "time_mu":
        z = 1.15
        title = z
        for mu in [0.7,0.8,0.9,1]:
            estimate = estimate_time(z,dt_inv,dt,N,S,pth)+mu
            condexps = calculate_risk(srng,S,estimate,func,simple)
            ax.plot(srng[:-1],condexps[:-1]-offset[:-1],label=f"{(mu):.2f}")
        title_params = rf"$z$={z:.2f}"
        ax.legend(title=r"$\mu$")
    elif compare == "alt_c":
        a=1
        title = a
        for c in [0.25,0.5,0.75]:
            estimate = a*estimate_alter(c,dt_inv,N,S,pth)
            condexps = calculate_risk(srng,S,estimate,func,simple)
            ax.plot(srng[:-1],condexps[:-1]-offset[:-1],label=f"{c:.2f}")
        title_params = rf"$a$={a:.2f}"
        ax.legend(title=r"$c$")
    elif compare == "alt_a":
        c=0.5
        title = c
        for a in [0.8,0.9,1,1.1,1.4]:
            estimate = a*estimate_alter(c,dt_inv,N,S,pth)
            condexps = calculate_risk(srng,S,estimate,func,simple)
            ax.plot(srng[:-1],condexps[:-1]-offset[:-1],label=f"{a:.2f}")
        title_params = rf"$c$={c:.2f}"
        ax.legend(title=r"$a$")


    if not plotdiff:
        ax.plot(srng[:-1],condexps_bayes[:-1],label="bayes", color="black")

    fig.suptitle(rf"Risk in {rf} $\alpha=${alpha}, {title_params}")
    # plt.show()
    fig.savefig(DIR_PLOTS.joinpath(f"Admis_MC_{rf}_{alpha}_space_{compare}_{title}{identifyer}.pdf"))

