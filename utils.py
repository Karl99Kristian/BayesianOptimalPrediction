from pathlib import Path
from scipy import stats
from matplotlib import pyplot as plt

# Set global paths
DIR = Path(__file__).resolve().parent
DIR_PLOTS = DIR.joinpath("plots")
DIR_DATA = DIR.joinpath("data")

# Set colormat and define std. normal
cmap = plt.get_cmap("tab10")
norm = stats.norm(0,1)