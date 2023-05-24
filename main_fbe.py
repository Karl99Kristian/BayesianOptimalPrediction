####################################
# Calculate free boundary equation #
####################################

# Imports
from utils import *
from functions import *
import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
from matplotlib import pyplot as plt


# Flags
calc_b = True # Calculate boundary or take from file
calc_v = False # Calculate value function or plot boundary function


# Model parameters
loss_name = "Mixed" # Name of loss function: Literal["LINEX","Mixed_analytic", "Mixed"]
a_params = [1]      # Loss function parameter

# Grid parameters
dt_inv = 30             # number of time steps
dx_inv = 50             # number of space steps in value calc
x_max = 1.5             # Upper limit for value plot


# Other options
lgnd_title = r"$\alpha$"    # Title in legend
identifyer = "_test1"       # Identifyer for file names
plot_squared = True         # Plot boundary from squared OP problem
non_unif = False            # Non-uniform grid
verbose = True              # Print thourgh loop
initial_point = 0           # B(1)
plot_max = False            # Plot the numerical max i b
newton_start_offset = 0     # offset from prevous guess in newton-raphson
infinity = 10               # Bound of integrals to avoid overflow

# Functions
def calc_bound(bt,bs,ts,a,osprob: Os_problem):
    """
    Return value of Volterra integral equation to be rooted
        bt is guess for current b(t)
        t is timepoint
        bs is prev. calced b(t+s). Assumed to be decreasing in time such that bs[0]=b(1)=0
        ts is prev. timepoints index maches bs(But including current time)
        a is loss parameter
        osprob is Os_problem object (i.e. collection of mayer, lagrange and infgen terms) 
    """

    t=ts[-1]
    term1 = quad(lambda x: osprob.mayer_at_mat(a,x)*trans_dens(1-t,bt,x),0,infinity)[0]
    term2 = osprob.mayer(t,bt,a)
    
    # Trapezoidal backwards from s=1-t
    ints1 = np.zeros_like(bs)
    ints2 = np.zeros_like(bs)

    ints1[0] = quad(lambda x: osprob.inf_gen_mayer_at_mat(a,x)*trans_dens(1-t,bt,x),0,infinity)[0]
    ints2[0] = 0
    for i, b in enumerate(bs[1:-1]):
        ints1[i+1] = quad(lambda x: osprob.inf_gen_mayer(ts[i+1],x,a)*trans_dens(ts[i+1]-t,bt,x),b,infinity)[0]
        ints2[i+1] = quad(lambda x: osprob.lagrange(ts[i+1],x,a)*trans_dens(ts[i+1]-t,bt,x),0,b)[0]

    ints1[-1] = osprob.inf_gen_mayer(t,bt,a) 
    ints2[-1] = osprob.lagrange(t,bt,a) 


    ws = ts[0:-2]-ts[1:-1]
    term3 = 1/2*(np.matmul(ws,ints1[1:])+np.matmul(ws,ints1[:-1]))
    term4 = 1/2*(np.matmul(ws,ints2[1:])+np.matmul(ws,ints2[:-1]))

    return -term1+term2+term3-term4

def calc_value(t,x, bt, bs,ts,a, osprob:Os_problem):
    """ Return value function """

    if t==1 or x>=bt:
        return osprob.mayer(t,x,a) 
    
    term1 = quad(lambda y: np.exp(a*y)*trans_dens(1-t,x,y),0,infinity)[0]
    
    # Trapezoidal backwards from s=1-t
    ints1 = np.zeros_like(bs)
    ints2 = np.zeros_like(bs)

    ints1[0] = quad(lambda y: osprob.inf_gen_mayer_at_mat(a,x)*trans_dens(1-t,x,y),0,infinity)[0]
    ints2[0] = 0
    for i, b in enumerate(bs[1:-1]):
        ints1[i+1] = quad(lambda y: osprob.inf_gen_mayer(ts[i+1],y,a)*trans_dens(ts[i+1]-t,x,y),b,infinity)[0]
        ints2[i+1] = quad(lambda y: osprob.lagrange(ts[i+1],y,a)*trans_dens(ts[i+1]-t,x,y),0,b)[0]

    ints1[-1] = osprob.inf_gen_mayer(t,x,a) 
    ints2[-1] = osprob.lagrange(t,x,a) 


    ws = ts[0:-2]-ts[1:-1]
    term3 = 1/2*(np.matmul(ws,ints1[1:])+np.matmul(ws,ints1[:-1]))
    term4 = 1/2*(np.matmul(ws,ints2[1:])+np.matmul(ws,ints2[:-1]))

    return term1-term3+term4

# Main
if __name__=="__main__":
    fig,ax = plt.subplots()
    # Calculate from input
    dt = 1/dt_inv
    dx = 1/dx_inv
    osprob = Os_problem(loss_name)

    if plot_squared:
        z_x_sqr = newton(lambda z: 4*norm.cdf(z)-2*z*norm.pdf(z)-3,1)

    for j, a in enumerate(a_params):
        
        if calc_b:
            if non_unif:
                trng = (np.arange(0,dt_inv+1)*dt)
                trng = np.sqrt(1-trng)
            else:
                trng = (np.arange(0,dt_inv+1)*dt)[::-1]
            
            # Resulting boundary vector
            b_res = np.zeros_like(trng)

            for i, bt in enumerate(b_res[1:]):
                if verbose: print(a,i) 

                if i == 0:
                    b_res[i+1]=initial_point
                    continue
                val=newton(calc_bound,b_res[i]+newton_start_offset,args=(b_res[:i+1],trng[:i+2],a,osprob,))
                b_res[i+1]=val
        
            # Save values in text file
            res = np.zeros((len(trng),2))
            res[:,0]=trng
            res[:,1]=b_res
            np.savetxt(DIR_DATA.joinpath(f"{loss_name}_{a}_{len(trng)}{identifyer}.csv"),res,delimiter=",")
    
        else:
            res = np.genfromtxt(DIR_DATA.joinpath(f"{loss_name}_{a}_{dt_inv+1}{identifyer}.csv"),delimiter=",")
            trng=res[:,0]
            b_res=res[:,1]


        if calc_b:
            ax.plot(trng,b_res, label=f"b(t), numt={dt_inv}(equidistant)")
            ax.scatter(trng,np.zeros_like(trng)-0.1,marker="x")
            if plot_max:
                print("Hej")
                maxidx = np.where(b_res==np.max(b_res))
                ax.scatter([trng[maxidx]],[b_res[maxidx]],color=cmap(j))
            
            fig.savefig(DIR_PLOTS.joinpath(f"{loss_name}_{a}_{len(trng)}{identifyer}.png"))
        elif not calc_v:
            ax.plot(trng,b_res, label=f"{a}", color=cmap(j))
            # ax.plot(trng,b_res[-1]*np.sqrt(1-trng), color=cmap(j), alpha=0.25)
            if plot_squared:
                ax.plot(trng,z_x_sqr*np.sqrt(1-trng), color="black",linestyle="dashed", alpha=0.25)

        
        if calc_v: 
            fig,ax = plt.subplots()  
            xrng =  np.arange(0,dx_inv+1)*dx*x_max
            value = np.zeros_like(xrng)

            # To reduce computation take only some of the points from b and t
            factor = 10
            b_res = b_res[0::factor]
            trng = trng[0::factor]

            for i,x in enumerate(xrng):
                if verbose: print(i)
                value[i] = calc_value(0,x,b_res[-1],b_res[:-1],trng,a,osprob)
            xrng_twoside = np.concatenate(((-xrng)[::-1],xrng))
            ax.plot(xrng_twoside[np.where(np.abs(xrng_twoside)<b_res[-1])],osprob.mayer(0,np.abs(xrng_twoside[np.where(np.abs(xrng_twoside)<b_res[-1])]),a),color="black",alpha=0.5,linestyle="dashed")
            
            ax.plot(xrng,value,color=cmap(0))
            ax.plot(-xrng,value,color=cmap(0))
            
            ax.set_xticks([-b_res[-1],0,b_res[-1]])
            ax.set_xticklabels([r"$-b(0)$",0,r"$b(0)$"])
            ax.set_yticks([value[0]])
            ax.set_yticklabels([r"$W_*(0)$"])
            fig.suptitle("Value function")
            
            fig.savefig(DIR_PLOTS.joinpath(f"{loss_name}_value_{a}{identifyer}.png"))
            print(f"a={a},val={value[0]}, c_*={np.log(value[0])/a}, b0={b_res[-1]}")

    if not (calc_b or calc_v):
        ax.legend(title=lgnd_title)
        ax.set_ylabel(r"$b(t)$")
        ax.set_xlabel(r"$t$")
        plt.suptitle("Numerical boundary function")
        fig.savefig(DIR_PLOTS.joinpath(f"{loss_name}_together{identifyer}.png"))