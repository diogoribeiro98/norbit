import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from .sizes import single_column_width
from .sizes import double_column_width
from .sizes import golden_ratio, silver_ratio

def Splot():
    """Simple example of a thing!

    Returns:
        _type_: _description_
    """
    
    #Configuration
    figsize = (single_column_width,single_column_width/golden_ratio)
    dpi     = 600

    fig,ax = plt.subplots(figsize = figsize, dpi = dpi)
    
    return fig, ax

def Eplot(wf = 0.0):
    """another

    Args:
        wf (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """

    #Configguration
    figsize = (double_column_width,(golden_ratio/2)*double_column_width)
    dpi     = 600

    #Create figure
    fig = plt.figure(figsize = figsize, dpi = dpi)
    gs  = gridspec.GridSpec( nrows=3, ncols=2 ,
                            width_ratios= [ 1+wf, 1], 
                            height_ratios=[1/3,1/3,1/3]) 

    #First plot
    ax  = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 1])

    return fig, ax, ax1, ax2, ax3
