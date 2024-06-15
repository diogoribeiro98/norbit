from ..plotemp.plotting import Eplot

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.pyplot as plt

import numpy as np

pn_color_list = ['#e66971', '#224b8b']

def pn_mural_plot(
        x0       = 15/1000,   
        y0       = 80/1000,  
        yscale   = 0.22   ,  
        time_range      = None ,
        comparison_plot = True,
        grid = True):

    #width fraction helper quantity
    wf = 0.6

    #Plot limits
    dx = wf*yscale
    dy = yscale

    #Create figure
    fig, ax , ax1 , ax2 , ax3 = Eplot(wf = wf)

    #Set orbit plot (left panel)
    ax.set_xlabel('$\\alpha$ [arcsec]')
    ax.set_ylabel('$\\beta$ [arcsec]')
    ax.set_xlim(x0 - dx/2,x0 + dx/2)
    ax.set_ylim(y0-dy/2,y0+dy/2)
    ax.grid(False)
    ax.axes.set_aspect('equal')

    #Create right panel for separate components
    if comparison_plot == True:
        ax1.set_xlabel('$t$ [years]')
        ax1.set_ylabel('$\\Delta \\alpha$ [$\mu as$]')
        ax1.grid(grid)
        ax2.set_xlabel('$t$ [years]')
        ax2.set_ylabel('$\\Delta \\beta$ [$\mu as$]')
        ax2.grid(grid)
        ax3.set_xlabel('$t$ [years]')
        ax3.set_ylabel('$ \\Delta \\theta$ [$\mu as$]')
        ax3.grid(grid)
    else:
        ax1.set_xlabel('$t$ [years]')
        ax1.set_ylabel('$\\alpha$ [$\mu as$]')
        ax1.set_ylim(x0 - dx/2,x0 + dx/2)
        
        ax2.set_xlabel('$t$ [years]')
        ax2.set_ylabel('$\\beta$ [$\mu as$]')
        ax2.set_ylim(y0-dy/2,y0+dy/2)
        ax3.set_xlabel('$t$ [years]')
        ax3.set_ylabel('$v_z$ [$m/s$]')

    if time_range != None:
        ax1.set_xlim(time_range[0],time_range[1])
        ax2.set_xlim(time_range[0],time_range[1])
        ax3.set_xlim(time_range[0],time_range[1])

    fig.tight_layout()
    
    return fig, ax, ax1, ax2, ax3

def add_inset(ax,xcenter=0, ycenter=0, xyrange = 1, loc1 = 2, loc2 =1):

    inset = inset_axes(ax, width=1.7, height=2.0, 
                   bbox_to_anchor=(0.82,0.75),
                   bbox_transform=ax.transAxes)
    
    inset.set_xlim(xcenter-xyrange/2, xcenter+xyrange/2)
    inset.set_ylim(ycenter-xyrange/2, ycenter+xyrange/2)
    inset.axes.get_xaxis().set_visible(False)
    inset.axes.get_yaxis().set_visible(False)
    inset.axes.set_aspect('equal')
    mark_inset(ax, inset, loc1=loc1, loc2=loc2, lw=0.2, fc="none",ec = 'black' , zorder=200)

    return inset


def pn_s2_plot( 
        x0       = 42/1000,   
        y0       = 100/1000,  
        yscale   = 280/1000,
        dpi      = 600):

    w = 3.375
    h = 2.3864
    aspect_ratio = (h/w)
    dx = aspect_ratio*yscale
    dy = yscale

    fig, ax = plt.subplots( figsize = (w , 2*h   ), dpi = dpi)
    
    ax.set_xlabel('$\\alpha$ [arcsec]') 
    ax.set_ylabel('$\\beta$ [arcsec]')
    ax.set_ylabel('$\\beta$ [arcsec]')
    ax.set_xlim(x0 - dx/2,x0 + dx/2)
    ax.set_ylim(y0-dy/2,y0+dy/2)


    return fig,ax


def draw_arrow(p1,p2,axis,**kwargs):
    start = [p1[0],p1[1]]
    end   = [p2[0],p2[1]]
    axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", **kwargs))


def draw_line(p1,p2,axis,**kwargs):
    axis.plot( [p1[0],p2[0]], [p1[1],p2[1]], **kwargs)

def add_scale(axis,
              x0, y0,
              scale,
              rotation = 0.0,
              capzise  = 0.3,
              offset   = 0.0,
              offsetlw = 0.4,
              **kwargs):
    
    #Draw line
    p1 = np.array([ x0 - (scale/2)*np.cos(np.deg2rad(rotation)) , y0 - (scale/2)*np.sin(np.deg2rad(rotation))]) 
    p2 = np.array([ x0 + (scale/2)*np.cos(np.deg2rad(rotation)) , y0 + (scale/2)*np.sin(np.deg2rad(rotation))]) 
    
    if offset != 0.0:
        p1off = p1 + np.array([ -(offset)*np.sin(np.deg2rad(rotation)) , (offset)*np.cos(np.deg2rad(rotation))])
        p2off = p2 + np.array([ -(offset)*np.sin(np.deg2rad(rotation)) , (offset)*np.cos(np.deg2rad(rotation))])
        draw_line(p1off,p2off,axis,**kwargs)
    
        #Add caps to points
        cap = np.array([ -(capzise*scale/2)*np.sin(np.deg2rad(rotation)) , (capzise*scale/2)*np.cos(np.deg2rad(rotation))])
    
        draw_line(p1off-cap,p1off+cap,axis,**kwargs)
        draw_line(p2off-cap,p2off+cap,axis,**kwargs)

        #Add offset lines
        draw_line(p1,p1off,axis,lw = offsetlw , ls = '--', color= 'gray',zorder = 0)
        draw_line(p2,p2off,axis,lw = offsetlw , ls = '--', color= 'gray',zorder = 0)

    
    else:
        draw_line(p1,p2,axis,**kwargs)

        #Add caps to points
        cap = np.array([ -(capzise*scale/2)*np.sin(np.deg2rad(rotation)) , (capzise*scale/2)*np.cos(np.deg2rad(rotation))])
    
        draw_line(p1-cap,p1+cap,axis,**kwargs)
        draw_line(p2-cap,p2+cap,axis,**kwargs)




