import warnings
warnings.filterwarnings('ignore')

# Enable import of other modules
import sys
ROOT = r"C:\Users\g.varvounis\Documents\RicTool"
if ROOT not in sys.path:
    sys.path.append(ROOT)
###############################

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def plot_simulation(
        simulation_df: pd.DataFrame,
        font_size: int = 12,
        use_grid: bool = True):
    """Plots all paths (scenarios) of conducted simulation."""

    plt.plot(simulation_df)

    plt.title("Simulated Scenarios (Paths)", fontweight="bold", fontsize=font_size)

    adjusted_font_size = font_size - 2
    plt.xlabel("Time", fontweight="bold", fontsize=adjusted_font_size)
    plt.ylabel("Values", fontweight="bold", fontsize=adjusted_font_size)

    plt.grid(use_grid)
    plt.show()

    