from .basic_templates import Eplot
import numpy as np
    
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import ArrowStyle
from matplotlib.patches import FancyArrowPatch

pn_color_list = ['#e66971', '#224b8b']

def pn_mural_plot(
        x0       = -15/1000,   
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

def pn_s2_plot( 
        x0       = -30/1000,   
        y0       = 33/1000,  
        yscale   = 320/1000,
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

    ax.grid()
    ax.set_aspect('equal')
    plt.tight_layout()

    return fig,ax

def add_inset(ax,x0,y0,dl, 
              anchor      = [0.95,0.95], 
              corners     = [2,3] , 
              minset      = None  ,
              show_scale  = False  ,
              scale_label = None  ,
              zorder      = 10    ):

    #Create inset and set limits
    inset = inset_axes(ax, width=0.7, height=0.7, bbox_to_anchor=(anchor[0],anchor[1]),bbox_transform=ax.transAxes, loc="upper right")
    
    inset.set_xlim(x0-dl, x0+dl)
    inset.set_ylim(y0-dl, y0+dl)
    inset.axes.get_xaxis().set_visible(False)
    inset.axes.get_yaxis().set_visible(False)
    inset.axes.set_aspect('equal')
    
    if minset != None:
        mark_inset(minset, inset, loc1=corners[0], loc2=corners[1], lw=0.2, fc="none",ec = 'gray' , zorder=zorder)

    if show_scale:
        #Create arrow 
        arrow_style = ArrowStyle(stylename='<->', head_length=2.4, head_width=2.4)
        arrow_path  = Path([[x0-dl, y0-dl*1.3], [x0+dl, y0-dl*1.3]])

        inset.add_patch(
            FancyArrowPatch(path=arrow_path,arrowstyle=arrow_style,lw = 0.8,capstyle='butt', clip_on=False)
        )
    
        if scale_label != None:
            #Add label
            inset.text(x0, y0-dl*1.65, scale_label, ha='center', va='center', size=9, zorder=500.0)

    return inset

def draw_arrow(axis,x0,y0,offx=0.0,offy=0.0,scale=0.6,**kwargs):
    
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()

    lx = -scale*x0/np.sqrt(x0**2+y0**2)*(xlim[1]-xlim[0])/2
    ly = -scale*y0/np.sqrt(x0**2+y0**2)*(ylim[1]-ylim[0])/2
    
    ox = offx*(xlim[1]-xlim[0])/2
    oy = offy*(ylim[1]-ylim[0])/2
    
    start = [x0+ox , y0+oy]
    end   = [x0+ox+lx,y0+oy+ly]
    axis.annotate("", xy=end, xytext=start,xycoords ='data', arrowprops=dict(arrowstyle="->", **kwargs,clip_on=False),clip_on=False)

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



def plot_astrometric_data(ax,ax1,ax2,instrument, escale=1, color='black', fitter=None, plot_residuals=False):
    
    error_style = {'fmt': 'none', 'elinewidth' :0.8, 'capsize': 1.0, 'capthick': 0.8, 'ecolor': color}

    if plot_residuals==False:
        ax.errorbar(
        x   =fitter.astrometric_data[instrument]['xdata'],
        y   =fitter.astrometric_data[instrument]['ydata'],
        xerr=fitter.astrometric_data[instrument]['xdata_err']*escale,
        yerr=fitter.astrometric_data[instrument]['ydata_err']*escale,
        **error_style
        )

        ax1.errorbar(
        x   =fitter.astrometric_data[instrument]['tdata'],
        y   =fitter.astrometric_data[instrument]['xdata'],
        yerr=fitter.astrometric_data[instrument]['xdata_err']*escale,
        **error_style    
        )

        ax2.errorbar(
        x   =fitter.astrometric_data[instrument]['tdata'],
        y   =fitter.astrometric_data[instrument]['ydata'],
        yerr=fitter.astrometric_data[instrument]['ydata_err']*escale,
        **error_style
        )
    
    else:

        xmodel  = -fitter.minimize_sol.RA(fitter.astrometric_data[instrument]['tdata']-fitter.minimize_result.params['t0'].value)
        ymodel  = fitter.minimize_sol.DEC(fitter.astrometric_data[instrument]['tdata']-fitter.minimize_result.params['t0'].value)

        ax.errorbar(
        x   =fitter.astrometric_data[instrument]['xdata'],
        y   =fitter.astrometric_data[instrument]['ydata'],
        xerr=fitter.astrometric_data[instrument]['xdata_err']*escale,
        yerr=fitter.astrometric_data[instrument]['ydata_err']*escale,
        **error_style
        )

        ax1.errorbar(
        x   =fitter.astrometric_data[instrument]['tdata'],
        y   =fitter.astrometric_data[instrument]['xdata']-xmodel,
        yerr=fitter.astrometric_data[instrument]['xdata_err']*escale,
        **error_style    
        )

        ax2.errorbar(
        x   =fitter.astrometric_data[instrument]['tdata'],
        y   =fitter.astrometric_data[instrument]['ydata']-ymodel,
        yerr=fitter.astrometric_data[instrument]['ydata_err']*escale,
        **error_style
        )


def plot_spectroscopic_data(ax3,instrument,escale=1, color='black', fitter=None, plot_residuals=False):

    error_style = {'fmt': 'none', 'elinewidth' :0.8, 'capsize': 1.0, 'capthick': 0.8,'ecolor': color}

    if plot_residuals==False:
        ax3.errorbar(
        x   =fitter.spectroscopic_data[instrument]['tdata'],
        y   =fitter.spectroscopic_data[instrument]['vdata'],
        yerr=fitter.spectroscopic_data[instrument]['vdata_err']*escale,
        **error_style
        )
    else:
        vzmodel = fitter.minimize_sol.vrs(fitter.spectroscopic_data[instrument]['tdata']-fitter.minimize_result.params['t0'].value)

        ax3.errorbar(
        x   =fitter.spectroscopic_data[instrument]['tdata'],
        y   =fitter.spectroscopic_data[instrument]['vdata']-vzmodel,
        yerr=fitter.spectroscopic_data[instrument]['vdata_err']*escale,
        **error_style
        )



