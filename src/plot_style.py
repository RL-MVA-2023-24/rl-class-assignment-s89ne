import matplotlib.pyplot as plt
# set default color palette to plt.cm.rainbow
from matplotlib.pyplot import cm
import matplotlib as mpl
import numpy as np

plt.show()
plot_colors = ["#FF1493",  # Light Pink
               "#9C27B0",  # Purple
               "#FF5722",  # Deep Orange
               "#FF9800",  # Orange
               "#3F51B5",  # Indigo
               "#00BCD4",  # Cyan
               "#4CAF50",  # Strong Green
               "#2196F3",  # Blue
               "#FFC107",  # Amber
               "#E91E63",  # Pink
               "#8BC34A"]  # Light Green# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=plot_colors)
# Set the default font size
mpl.rcParams.update({'font.size': 20})
# Set the default axes title size
mpl.rcParams.update({'axes.titlesize': 25})
# Set the default axes label size
mpl.rcParams.update({'axes.labelsize': 20})
# Set the default line width
mpl.rcParams['lines.linewidth'] = 2
# Set automatic tight layout
mpl.rcParams['figure.autolayout'] = True
# Set automatic grid with seaborn style
# plt.style.use('seaborn-darkgrid')
# Set axis background to #FF1493
mpl.rcParams['axes.facecolor'] = "#F7F6F6"
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.color'] = 'k'
# Set the default font family to Computer Modern
mpl.rcParams['font.family'] = 'serif'
# Set the default figure size
mpl.rcParams['figure.figsize'] = [10, 6]
# Add white background to the figure
mpl.rcParams['figure.facecolor'] = 'w'
# Set the changes
plt.rcParams.update(mpl.rcParams)