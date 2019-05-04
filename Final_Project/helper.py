# importing libraries
import numpy
import os
from scipy import integrate, linalg
import math
from matplotlib import pyplot
#%matplotlib inline

#---------------------------------------------------------#
#-----------------Biplane plotting functions--------------#
#---------------------------------------------------------#

def subplots_plot_biplane_geo(x_m, y_m,x_bi_1, y_bi_1,x_bi_2, y_bi_2,x_bi_3, y_bi_3,x_bi_4, y_bi_4):
    fig, axs = pyplot.subplots(2, 2, figsize=(2*5.0, 6.0),sharex=False,sharey=False)
    axs[0, 0].plot(x_m, y_m)
    axs[0, 0].plot(x_bi_1, y_bi_1)
    axs[0, 0].set_title('Biplane Geometry: Gap/Chord=0.75', fontsize=16)
    #axs[0, 0].set_xlabel('x',fontsize=16)
    axs[0, 0].set_ylabel('y',fontsize=16)
    axs[0, 1].plot(x_m, y_m)
    axs[0, 1].plot(x_bi_2, y_bi_2)
    axs[0, 1].set_title('Biplane Geometry: Gap/Chord=1.00', fontsize=16)
    #axs[0, 1].set_xlabel('x',fontsize=16)
    #axs[0, 1].set_ylabel('y',fontsize=16)
    axs[1, 0].plot(x_m, y_m)
    axs[1, 0].plot(x_bi_3, y_bi_3)
    axs[1, 0].set_title('Biplane Geometry: Gap/Chord=1.25', fontsize=16)
    axs[1, 0].set_xlabel('x',fontsize=16)
    axs[1, 0].set_ylabel('y',fontsize=16)
    axs[1, 1].plot(x_m, y_m)
    axs[1, 1].plot(x_bi_4, y_bi_4)
    axs[1, 1].set_title('Biplane Geometry: Gap/Chord=1.50', fontsize=16)
    axs[1, 1].set_xlabel('x',fontsize=16)
    #axs[1, 1].set_ylabel('y',fontsize=16)
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)

def subplots_plot_biplane_panels(x_m, y_m,x_bi_1, y_bi_1,x_bi_2, y_bi_2,x_bi_3, y_bi_3,x_bi_4, y_bi_4,\
                              lower_panels_bi,upper_panels_bi_1, upper_panels_bi_2,upper_panels_bi_3, upper_panels_bi_4):
    
    fig, axs = pyplot.subplots(2, 2, figsize=(2*5.0, 6.0),sharex=False,sharey=False)
    axs[0, 0].plot(numpy.append([panel.xa for panel in lower_panels_bi], lower_panels_bi[0].xa),
            numpy.append([panel.ya for panel in lower_panels_bi], lower_panels_bi[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    axs[0, 0].plot(numpy.append([panel.xa for panel in upper_panels_bi_1], upper_panels_bi_1[0].xa),
            numpy.append([panel.ya for panel in upper_panels_bi_1], upper_panels_bi_1[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    axs[0, 0].plot(x_m, y_m)
    axs[0, 0].plot(x_bi_1, y_bi_1)
    axs[0, 0].set_title('Biplane Panels: Gap/Chord=0.75', fontsize=16)
    #axs[0, 0].set_xlabel('x',fontsize=16)
    axs[0, 0].set_ylabel('y',fontsize=16)
    axs[0, 1].plot(numpy.append([panel.xa for panel in lower_panels_bi], lower_panels_bi[0].xa),
            numpy.append([panel.ya for panel in lower_panels_bi], lower_panels_bi[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    axs[0, 1].plot(numpy.append([panel.xa for panel in upper_panels_bi_2], upper_panels_bi_2[0].xa),
            numpy.append([panel.ya for panel in upper_panels_bi_2], upper_panels_bi_2[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    axs[0, 1].plot(x_m, y_m)
    axs[0, 1].plot(x_bi_2, y_bi_2)
    axs[0, 1].set_title('Biplane Panels: Gap/Chord=1.00', fontsize=16)
    #axs[0, 1].set_xlabel('x',fontsize=16)
    #axs[0, 1].set_ylabel('y',fontsize=16)
    axs[1, 0].plot(numpy.append([panel.xa for panel in lower_panels_bi], lower_panels_bi[0].xa),
            numpy.append([panel.ya for panel in lower_panels_bi], lower_panels_bi[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    axs[1, 0].plot(numpy.append([panel.xa for panel in upper_panels_bi_3], upper_panels_bi_3[0].xa),
            numpy.append([panel.ya for panel in upper_panels_bi_3], upper_panels_bi_3[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    axs[1, 0].plot(x_m, y_m)
    axs[1, 0].plot(x_bi_3, y_bi_3)
    axs[1, 0].set_title('Biplane Panels: Gap/Chord=1.25', fontsize=16)
    axs[1, 0].set_xlabel('x',fontsize=16)
    axs[1, 0].set_ylabel('y',fontsize=16)
    axs[1, 1].plot(numpy.append([panel.xa for panel in lower_panels_bi], lower_panels_bi[0].xa),
            numpy.append([panel.ya for panel in lower_panels_bi], lower_panels_bi[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    axs[1, 1].plot(numpy.append([panel.xa for panel in upper_panels_bi_4], upper_panels_bi_4[0].xa),
            numpy.append([panel.ya for panel in upper_panels_bi_4], upper_panels_bi_4[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    axs[1, 1].plot(x_m, y_m)
    axs[1, 1].plot(x_bi_4, y_bi_4)
    axs[1, 1].set_title('Biplane Panels: Gap/Chord=1.50', fontsize=16)
    axs[1, 1].set_xlabel('x',fontsize=16)
    #axs[1, 1].set_ylabel('y',fontsize=16)
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    
#---------------------------------------------------------#
#-------------------------------create the panels --------#
#---------------------------------------------------------#

# create the panel class
class Panel:
    def __init__(self, xa, ya, xb, yb):

        self.xa, self.ya = xa, ya  # panel starting-point
        self.xb, self.yb = xb, yb  # panel ending-point
        
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2  # panel center
        self.length = numpy.sqrt((xb - xa)**2 + (yb - ya)**2)  # panel length
        
        # orientation of panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = numpy.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = numpy.pi + numpy.arccos(-(yb - ya) / self.length)
        
        # panel location
        if self.beta <= numpy.pi:
            self.loc = 'upper'  # upper surface
        else:
            self.loc = 'lower'  # lower surface
        
        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient

# define the panels
def define_panels(x, y, N):
    
    # becuase we were given the end-points in the geometry, we don't need to create and map the circle
    # create panels
    panels = numpy.empty(N, dtype=object)
    
    for i in range(N):
        panels[i] = Panel(x[i], y[i], x[i + 1], y[i + 1])
    
    return panels

#---------------------------------------------------------#
#-------create the source and vortex contributions--------#
#---------------------------------------------------------#
    
# create the integral 
def integral(x, y, panel, dxdk, dydk):
    def integrand(s):
        return (((x - (panel.xa - numpy.sin(panel.beta) * s)) * dxdk +
                 (y - (panel.ya + numpy.cos(panel.beta) * s)) * dydk) /
                ((x - (panel.xa - numpy.sin(panel.beta) * s))**2 +
                 (y - (panel.ya + numpy.cos(panel.beta) * s))**2) )
    return integrate.quad(integrand, 0.0, panel.length)[0]

# create the source contribution
def source_contribution_normal(panelS):
    A = numpy.empty((len(panelS), len(panelS)), dtype=float)
    # source contribution on a panel from itself
    numpy.fill_diagonal(A, 0.5)
    # source contribution on a panel from others
    for i, panel_i in enumerate(panelS):
        for j, panel_j in enumerate(panelS):
            if i != j:
                A[i, j] = 0.5 / numpy.pi * integral(panel_i.xc, panel_i.yc, 
                                                    panel_j,
                                                    numpy.cos(panel_i.beta),
                                                    numpy.sin(panel_i.beta))
    return A

# create the vortex contribution
def vortex_contribution_normal(panelS):
    A = numpy.empty((len(panelS), len(panelS)), dtype=float)
    # vortex contribution on a panel from itself
    numpy.fill_diagonal(A, 0.0)
    # vortex contribution on a panel from others
    for i, panel_i in enumerate(panelS):
        for j, panel_j in enumerate(panelS):
            if i != j:
                A[i, j] = -0.5 / numpy.pi * integral(panel_i.xc, panel_i.yc, 
                                                     panel_j,
                                                     numpy.sin(panel_i.beta),
                                                     -numpy.cos(panel_i.beta))
    return A

#---------------------------------------------------------#
#---------create the linear system for the biplane--------#
#---------------------------------------------------------#

# create the kutta conditions
# 1st kutta condition will go from 1 to N-2
# 2nd kutta condition will go from N-1 to 2(N-1)
def kutta_condition_1(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+2), dtype=float)
    
    # i = row that we need to work with
    b[:,:-2] = B_vortex[0,:] + B_vortex[(N-1),:]
    b[:,-2] = - numpy.sum(A_source[0,:N]) - numpy.sum(A_source[(N-1),:N])
    b[:,-1] = - numpy.sum(A_source[0,N:]) - numpy.sum(A_source[(N-1),N:])
    
    return b 

def kutta_condition_2(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+2), dtype=float)
    
    # i = row that we need to work with
    b[:,:-2] = B_vortex[N,:] + B_vortex[-1,:]
    b[:,-2] = - numpy.sum(A_source[N,:N]) - numpy.sum(A_source[-1,:N])
    b[:,-1] = - numpy.sum(A_source[N,N:]) - numpy.sum(A_source[-1,N:])
    
    return b 

# create the singularity matrix (an amalgamation of the kutta condition, source term, and the vortex term)
def build_singularity_matrix(A_source, B_vortex, N):

    A = numpy.empty((A_source.shape[0] + 2, A_source.shape[1] + 2), dtype=float)
    # source contribution matrix
    A[:-2, :-2] = A_source
    # vortex contribution array 1
    panda=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        panda[i,:]=numpy.sum(B_vortex[i,:N])
    A[:-2, -2] = panda[:,0]
    # vortex contribution array 2
    dolphin=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        dolphin[i,:]=numpy.sum(B_vortex[i,N:])
    A[:-2, -1] = dolphin[:,0]
    # Kutta condition #1
    A[-2, :] = kutta_condition_1(A_source, B_vortex, N)
    # Kutta condition #2
    A[-1, :] = kutta_condition_2(A_source, B_vortex, N)
    return A

# create the freestream
def build_freestream_rhs(panelS, freestream, N):

    b = numpy.empty(len(panelS) + 2, dtype=float)
    # freestream contribution on each panel
    for i, panel in enumerate(panelS):
        b[i] = -freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)
    # freestream contribution on the Kutta condition
    b[-2] = -freestream.u_inf * (numpy.sin(freestream.alpha - panelS[0].beta) +
                                 numpy.sin(freestream.alpha - panelS[(N-1)].beta) )
    b[-1] = -freestream.u_inf * (numpy.sin(freestream.alpha - panelS[N].beta) +
                                 numpy.sin(freestream.alpha - panelS[-1].beta) )
    return b

def compute_tangential_velocity(panelS, freestream, gamma, A_source, B_vortex, N):
    
    A = numpy.empty((len(panelS), len(panelS) + 2), dtype=float)
    
    # B_vortex contribution
    A[:, :-2] = B_vortex

    # A_source contribution from foil
    pangolin=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        pangolin[i,:]=-numpy.sum(A_source[i,:N])
    A[:, -2] = pangolin[:,0]

    # A_source contribution from flap
    lemur=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        lemur[i,:]=-numpy.sum(A_source[i,N:])
    A[:, -1] = lemur[:,0]
    
    # freestream contribution
    b = freestream.u_inf * numpy.sin([freestream.alpha - panel.beta 
                                      for panel in panelS])
    
    strengths = numpy.append([panel.sigma for panel in panelS], gamma)
    
    tangential_velocities = numpy.dot(A, strengths) + b
    
    for i, panel in enumerate(panelS):
        panel.vt = tangential_velocities[i]
        
# get the pressure coefficients
def compute_pressure_coefficient(panels, freestream):

    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2

#-------------------------------------------------------------------------#
#---------create the function to solve the system and get the lift--------#
#-------------------------------------------------------------------------#

def solve_biplane(panelS, freestream): # fix foil_panelS
    N = int(len(panelS)/2)
    # find the source and vortex contributions
    A_source = source_contribution_normal(panelS)
    B_vortex = vortex_contribution_normal(panelS)
    
    # build the singularity matrix
    A = build_singularity_matrix(A_source, B_vortex, N)
    b = build_freestream_rhs(panelS, freestream, N)
    
    # solve for the singularity matrices
    strengths = numpy.linalg.solve(A,b)
    
    #store the strengths on each panel
    for i, panel in enumerate(panelS):
        panel.sigma = strengths[i]
        
    # store the circulation density
    gamma = strengths[-2:]
    
    # tangential velocity at each panel center.
    compute_tangential_velocity(panelS, freestream, gamma, A_source, B_vortex, N)
    
    # surface pressure coefficient
    compute_pressure_coefficient(panelS, freestream)
    
    # check that the work so far is correct => for a closed body the sum of the strengths must = 0
    accuracy = sum([panel.sigma*panel.length for panel in panelS])
    #print('sum of singularity strengths: {:0.6f}'.format(accuracy))
    
    # what is the value of the lift for the system (main + flap)
    # assume that P_inf = 0, density = 1 => use bernouli's equation
    P_inf=0
    den=1
    u_inf=1.0
    
    # the parts needed for the integral of of lift are:
    
    # tangential velocity of the panels
    foil_vt = [panel.vt for panel in panelS[:N]]
    flap_vt = [panel.vt for panel in panelS[N:]]
    
    # length of the panels
    foil_len = [panel.length for panel in panelS[:N]]
    flap_len = [panel.length for panel in panelS[N:]]
    
    # angle of the panels called n*j (beta)
    foil_angle = [panel.beta for panel in panelS[:N]]
    flap_angle = [panel.beta for panel in panelS[N:]]
    
    # need to find the lift of the center of each panel individually and add them up
    
    # foil lift array
    loquat = numpy.empty(N, dtype=float)
    for i in range (N):
        loquat[i]=-(P_inf+0.5*den*(u_inf**2-foil_vt[i]**2))*foil_len[i]*math.sin(foil_angle[i])
        
    # flap lift array
    mango = numpy.empty(N, dtype=float)
    for i in range (N):
        mango[i]=-(P_inf+0.5*den*(u_inf**2-flap_vt[i]**2))*flap_len[i]*math.sin(flap_angle[i])
    
    #print(loquat)
    #print(mango)
    
    # summing the lift arrays and ading them up
    L=numpy.sum(loquat)+numpy.sum(mango)
    #print('This is the value of the lift for the system {main+flap} for 100 panels: '+'\n'+str(L_100)+'\n')
    
    return L

#-------------------------------------------------------------------------#
#-------------------Solve the system for the monoplane--------------------#
#-------------------------------------------------------------------------#

def kutta_condition_mono(A_source, B_vortex):

    b = numpy.empty(A_source.shape[0] + 1, dtype=float)
    # matrix of source contribution on tangential velocity
    # is the same than
    # matrix of vortex contribution on normal velocity
    b[:-1] = B_vortex[0, :] + B_vortex[-1, :]
    # matrix of vortex contribution on tangential velocity
    # is the opposite of
    # matrix of source contribution on normal velocity
    b[-1] = - numpy.sum(A_source[0, :] + A_source[-1, :])
    return b

def build_singularity_matrix_mono(A_source, B_vortex):

    A = numpy.empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype=float)
    # source contribution matrix
    A[:-1, :-1] = A_source
    # vortex contribution array
    A[:-1, -1] = numpy.sum(B_vortex, axis=1)
    # Kutta condition array
    A[-1, :] = kutta_condition_mono(A_source, B_vortex)
    return A

def build_freestream_rhs_mono(panels, freestream):

    b = numpy.empty(panels.size + 1, dtype=float)
    # freestream contribution on each panel
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)
    # freestream contribution on the Kutta condition
    b[-1] = -freestream.u_inf * (numpy.sin(freestream.alpha - panels[0].beta) +
                                 numpy.sin(freestream.alpha - panels[-1].beta) )
    return b

def compute_tangential_velocity_mono(panels, freestream, gamma, A_source, B_vortex):

    A = numpy.empty((panels.size, panels.size + 1), dtype=float)
    # matrix of source contribution on tangential velocity
    # is the same than
    # matrix of vortex contribution on normal velocity
    A[:, :-1] = B_vortex
    # matrix of vortex contribution on tangential velocity
    # is the opposite of
    # matrix of source contribution on normal velocity
    A[:, -1] = -numpy.sum(A_source, axis=1)
    # freestream contribution
    b = freestream.u_inf * numpy.sin([freestream.alpha - panel.beta 
                                      for panel in panels])
    
    strengths = numpy.append([panel.sigma for panel in panels], gamma)
    
    tangential_velocities = numpy.dot(A, strengths) + b
    
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]

def compute_pressure_coefficient_mono(panels, freestream):

    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2

def solve_monoplane(panelS, freestream): # fix foil_panelS
    
    # find the source and vortex contributions
    A_source = source_contribution_normal(panelS)
    B_vortex = vortex_contribution_normal(panelS)
    
    # build the singularity matrix
    A = build_singularity_matrix_mono(A_source, B_vortex)
    b = build_freestream_rhs_mono(panelS, freestream)
    
    # solve for the singularity matrices
    strengths = numpy.linalg.solve(A,b)
    
    #store the strengths on each panel
    for i, panel in enumerate(panelS):
        panel.sigma = strengths[i]
        
    # store the circulation density
    gamma = strengths[-1]
    
    # tangential velocity at each panel center.
    compute_tangential_velocity_mono(panelS, freestream, gamma, A_source, B_vortex)
    
    # surface pressure coefficient
    compute_pressure_coefficient_mono(panelS, freestream)
    
    # check that the work so far is correct => for a closed body the sum of the strengths must = 0
    accuracy = sum([panel.sigma*panel.length for panel in panelS])
    #print('sum of singularity strengths: {:0.6f}'.format(accuracy))
    
    # what is the value of the lift for the system (main + flap)
    # assume that P_inf = 0, density = 1 => use bernouli's equation
    P_inf=0
    den=1
    u_inf=1.0
    
    # the parts needed for the integral of of lift are:
    
    # tangential velocity of the panels
    foil_vt = [panel.vt for panel in panelS]
    
    # length of the panels
    foil_len = [panel.length for panel in panelS]
    
    # angle of the panels called n*j (beta)
    foil_angle = [panel.beta for panel in panelS]

    # foil lift array
    loquat = numpy.empty(len(foil_angle), dtype=float)
    for i in range (len(foil_angle)):
        loquat[i]=-(P_inf+0.5*den*(u_inf**2-foil_vt[i]**2))*foil_len[i]*math.sin(foil_angle[i])
    
    # summing the lift arrays and ading them up
    L=numpy.sum(loquat)
    #print('This is the value of the lift for the system {main+flap} for 100 panels: '+'\n'+str(L_100)+'\n')
    
    return L

#--------------------------------------------------------#
#-------function to do everything for the biplane--------#
#--------------------------------------------------------#

def translate_geo(x_foil, y_foil, x_new, y_new):
    x = x_foil+x_new
    y = y_foil+y_new
    return x,y

def everything_biplane(x_m, y_m, x_n1, y_n,lower_panels_bi, freestream):
    x_bi, y_bi = translate_geo(x_m, y_m, x_n1, y_n)
    n = len(x_m)
    upper_panels_bi = define_panels(x_bi, y_bi, n-1)
    panelS_bi = numpy.concatenate((lower_panels_bi, upper_panels_bi))
    L_bi = solve_biplane(panelS_bi, freestream)
    return L_bi

def gap_chord_6(gap,Lift_percent_tot_6,Lift_mono):
    # plot the biplane info
    # plot geometry
    width = 5
    pyplot.figure(figsize=(width*1.5, width))
    pyplot.title('Gap/Chord vs Lift')
    pyplot.grid()
    pyplot.xlabel('Gap/Chord', fontsize=16)
    pyplot.ylabel('Lift [%]', fontsize=16)
    pyplot.plot(gap, Lift_percent_tot_6, color='k', linestyle='-', linewidth=2)
    pyplot.plot(gap, Lift_mono, color='r', linestyle='-', linewidth=2)
    pyplot.xlim(0.5, 4.0)
    pyplot.ylim(75, 105);

#---------------------------------------------------------#
#--------create the linear system for the triplane--------#
#---------------------------------------------------------#

def create_triplane (x_foil, y_foil, x_n1,y_gap):
    x_t_1, y_t_1 = translate_geo(x_foil, y_foil, x_n1, y_gap)
    x_t_2, y_t_2 = translate_geo(x_foil, y_foil, x_n1, 2*y_gap)
    N = len(x_foil)
    print(N-1)
    bot_panels = define_panels(x_foil, y_foil, N-1)
    mid_panels = define_panels(x_t_1, y_t_1, N-1)
    top_panels = define_panels(x_t_2, y_t_2, N-1)
    panelS_tri = numpy.concatenate((bot_panels, mid_panels,top_panels ))
    return x_t_1, y_t_1, x_t_2, y_t_2, bot_panels, mid_panels, top_panels, panelS_tri

# plot the geometry
def plot_triplane_panels(x_bot, y_bot,x_mid, y_mid,x_top, y_top,bot_panels,mid_panels,top_panels):
    # plot discretized geometry
    pyplot.figure(figsize=(5, 5))
    pyplot.title('Triplane with Gap/Chord = 1.25')
    pyplot.grid()
    pyplot.xlabel('x', fontsize=16)
    pyplot.ylabel('y', fontsize=16)
    pyplot.plot(x_bot, y_bot, color='k', linestyle='-', linewidth=2)
    pyplot.plot(numpy.append([panel.xa for panel in bot_panels], bot_panels[0].xa),
            numpy.append([panel.ya for panel in bot_panels], bot_panels[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    pyplot.plot(x_mid, y_mid, color='k', linestyle='-', linewidth=2)
    pyplot.plot(numpy.append([panel.xa for panel in mid_panels], mid_panels[0].xa),
            numpy.append([panel.ya for panel in mid_panels], mid_panels[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    pyplot.plot(x_top, y_top, color='k', linestyle='-', linewidth=2)
    pyplot.plot(numpy.append([panel.xa for panel in top_panels], top_panels[0].xa),
            numpy.append([panel.ya for panel in top_panels], top_panels[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    #pyplot.axis('scaled', adjustable='box')
    pyplot.xlim(-0.1, 1.1)
    pyplot.ylim(-0.2, 2.75);

# create the kutta conditions
# 1st kutta condition will go from 1 to (N)/3
# 2nd kutta condition will go from 1+(N)/3 to 2(N)/3
# 3rd kutta condition will go from 1+2(N)/3 to (-1)
def tri_kutta_condition_1(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+3), dtype=float)
    
    # i = row that we need to work with
    b[:,:-3] = B_vortex[0,:] + B_vortex[N-1,:]
    b[:,-3] = - numpy.sum(A_source[0,:N]) - numpy.sum(A_source[N-1,:N])
    b[:,-2] = - numpy.sum(A_source[0,N:(2*N)-1]) - numpy.sum(A_source[N-1,N:(2*N)-1])
    b[:,-1] = - numpy.sum(A_source[0,(2*N)-1:]) - numpy.sum(A_source[N-1,(2*N)-1:])
    
    return b 

def tri_kutta_condition_2(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+3), dtype=float)
    
    # i = row that we need to work with
    b[:,:-3] = B_vortex[N,:] + B_vortex[(N*2)-1,:]
    b[:,-3] = - numpy.sum(A_source[N,:N]) - numpy.sum(A_source[(2*N)-1,:N])
    b[:,-2] = - numpy.sum(A_source[N,N:2*N-1]) - numpy.sum(A_source[2*N-1,N:2*N-1])
    b[:,-1] = - numpy.sum(A_source[N,2*N-1:]) - numpy.sum(A_source[2*N-1,2*N-1:])
    
    return b 

def tri_kutta_condition_3(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+3), dtype=float)
    
    # i = row that we need to work with
    b[:,:-3] = B_vortex[(2*N),:] + B_vortex[-1,:]
    b[:,-3] = - numpy.sum(A_source[(2*N),:N]) - numpy.sum(A_source[-1,:N])
    b[:,-2] = - numpy.sum(A_source[(2*N),N:2*N-1]) - numpy.sum(A_source[-1,N:2*N-1])
    b[:,-1] = - numpy.sum(A_source[(2*N),2*N-1:]) - numpy.sum(A_source[-1,2*N-1:])
    
    return b 

# create the singularity matrix (an amalgamation of the kutta condition, source term, and the vortex term)
def build_singularity_matrix_tri(A_source, B_vortex, N):
    
    A = numpy.empty((A_source.shape[0] + 3, A_source.shape[1] + 3), dtype=float)
    # source contribution matrix
    A[:-3, :-3] = A_source
    # vortex contribution array 1
    panda=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        panda[i,:]=numpy.sum(B_vortex[i,:N])
    A[:-3, -3] = panda[:,0]
    # vortex contribution array 2
    dolphin=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        dolphin[i,:]=numpy.sum(B_vortex[i,N:2*N-1])
    A[:-3, -2] = dolphin[:,0]
    # vortex contribution array 3
    tiger=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        tiger[i,:]=numpy.sum(B_vortex[i,2*N-1:])
    A[:-3, -1] = tiger[:,0]
    # Kutta condition #1
    A[-3, :] = tri_kutta_condition_1(A_source, B_vortex, N)
    # Kutta condition #2
    A[-2, :] = tri_kutta_condition_2(A_source, B_vortex, N)
    # Kutta condition #2
    A[-1, :] = tri_kutta_condition_3(A_source, B_vortex, N)
    return A

# create the freestream
def build_freestream_rhs_tri(panelS, freestream, N):
    
    b = numpy.empty(len(panelS) + 3, dtype=float)
    # freestream contribution on each panel
    for i, panel in enumerate(panelS):
        b[i] = -freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)
    # freestream contribution on the Kutta condition
    b[-3] = -freestream.u_inf * (numpy.sin(freestream.alpha - panelS[0].beta) +
                                 numpy.sin(freestream.alpha - panelS[N-1].beta) )
    b[-2] = -freestream.u_inf * (numpy.sin(freestream.alpha - panelS[N].beta) +
                                 numpy.sin(freestream.alpha - panelS[(2*N)-1].beta) )
    b[-1] = -freestream.u_inf * (numpy.sin(freestream.alpha - panelS[(2*N)].beta) +
                                 numpy.sin(freestream.alpha - panelS[-1].beta) )
    return b

def compute_tangential_velocity_tri(panelS, freestream, gamma, A_source, B_vortex, N):
    
    A = numpy.empty((len(panelS), len(panelS) + 3), dtype=float)
    
    # B_vortex contribution
    A[:, :-3] = B_vortex

    # A_source contribution from bottom
    pangolin=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        pangolin[i,:]=-numpy.sum(A_source[i,:N])
    A[:, -3] = pangolin[:,0]

    # A_source contribution from middle
    lemur=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        lemur[i,:]=-numpy.sum(A_source[i,N:2*N-1])
    A[:, -2] = lemur[:,0]
    
    # A_source contribution from middle
    sea=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        sea[i,:]=-numpy.sum(A_source[i,2*N-1:])
    A[:, -1] = sea[:,0]
    
    # freestream contribution
    b = freestream.u_inf * numpy.sin([freestream.alpha - panel.beta 
                                      for panel in panelS])
    
    strengths = numpy.append([panel.sigma for panel in panelS], gamma)
    
    tangential_velocities = numpy.dot(A, strengths) + b
    
    for i, panel in enumerate(panelS):
        panel.vt = tangential_velocities[i]
        
# get the pressure coefficients
def compute_pressure_coefficient_tri(panels, freestream):

    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2

#-------------------------------------------------------------------------#
#---------create the function to solve the system and get the lift--------#
#-------------------------------------------------------------------------#

def solve_triplane(panelS, freestream, N): # fix foil_panelS
    
    # find the source and vortex contributions
    A_source = source_contribution_normal(panelS)
    B_vortex = vortex_contribution_normal(panelS)
    
    # build the singularity matrix
    A = build_singularity_matrix_tri(A_source, B_vortex, N)
    b = build_freestream_rhs_tri(panelS, freestream, N)
    
    # solve for the singularity matrices
    strengths = numpy.linalg.solve(A,b)
    
    #store the strengths on each panel
    for i, panel in enumerate(panelS):
        panel.sigma = strengths[i]
        
    # store the circulation density
    gamma = strengths[-3:]
    
    # tangential velocity at each panel center.
    compute_tangential_velocity_tri(panelS, freestream, gamma, A_source, B_vortex, N)
    
    # surface pressure coefficient
    compute_pressure_coefficient_tri(panelS, freestream)
    
    # check that the work so far is correct => for a closed body the sum of the strengths must = 0
    accuracy = sum([panel.sigma*panel.length for panel in panelS])
    #print('sum of singularity strengths: {:0.6f}'.format(accuracy))
    
    # what is the value of the lift for the system (main + flap)
    # assume that P_inf = 0, density = 1 => use bernouli's equation
    P_inf=0
    den=1
    u_inf=1.0
    
    # the parts needed for the integral of of lift are:
    #print(N)
    # tangential velocity of the panels
    bottom_vt = [panel.vt for panel in panelS[:N]]
    middle_vt = [panel.vt for panel in panelS[N:2*N-1]]
    top_vt = [panel.vt for panel in panelS[2*N-1:]]
    #print(len(bottom_vt))
    #print(len(middle_vt))
    #print(len(top_vt))
    
    # length of the panels
    bottom_len = [panel.length for panel in panelS[:N]]
    middle_len = [panel.length for panel in panelS[N:2*N-1]]
    top_len = [panel.length for panel in panelS[2*N-1:]]
    #print(len(bottom_len))
    #print(len(middle_len))
    #print(len(top_len))
    
    # angle of the panels called n*j (beta)
    bottom_angle = [panel.beta for panel in panelS[:N]]
    middle_angle = [panel.beta for panel in panelS[N:-N]]
    top_angle = [panel.beta for panel in panelS[-N:]]
    #print(len(bottom_angle))
    #print(len(middle_angle))
    #print(len(top_angle))
    
    # need to find the lift of the center of each panel individually and add them up
    
    # foil lift array
    loquat = numpy.empty(len(bottom_vt), dtype=float)
    for i in range (len(bottom_vt)):
        loquat[i]=-(P_inf+0.5*den*(u_inf**2-bottom_vt[i]**2))*bottom_len[i]*math.sin(bottom_angle[i])
        
    # flap lift array
    mango = numpy.empty(len(middle_len), dtype=float)
    for i in range (len(middle_len)):
        mango[i]=-(P_inf+0.5*den*(u_inf**2-middle_vt[i]**2))*middle_len[i]*math.sin(middle_angle[i])
        
    # flap lift array
    plant = numpy.empty(len(top_angle), dtype=float)
    for i in range (len(top_angle)):
        plant[i]=-(P_inf+0.5*den*(u_inf**2-top_vt[i]**2))*top_len[i]*math.sin(top_angle[i])
    
    #print(loquat)
    #print(mango)
    
    
    # summing the lift arrays and ading them up
    L=numpy.sum(loquat)+numpy.sum(mango)+numpy.sum(plant)
    #print('This is the value of the lift for the system {main+flap} for 100 panels: '+'\n'+str(L_100)+'\n')
    
    return L

#-------------------------------------------------------------------------#
#--------------------Biplane: Lift of individual wings--------------------#
#-------------------------------------------------------------------------#
def biplane_bottom_wing(panelS_bi, freestream, N):
    # find the source and vortex contributions
    A_source = source_contribution_normal(panelS)
    B_vortex = vortex_contribution_normal(panelS)
    
    # build the singularity matrix
    A = build_singularity_matrix(A_source, B_vortex, N)
    b = build_freestream_rhs(panelS, freestream, N)
    
    # solve for the singularity matrices
    strengths = numpy.linalg.solve(A,b)
    
    #store the strengths on each panel
    for i, panel in enumerate(panelS):
        panel.sigma = strengths[i]
        
    # store the circulation density
    gamma = strengths[-2:]
    
    # tangential velocity at each panel center.
    compute_tangential_velocity(panelS, freestream, gamma, A_source, B_vortex, N)
    
    # surface pressure coefficient
    compute_pressure_coefficient(panelS, freestream)
    
    # check that the work so far is correct => for a closed body the sum of the strengths must = 0
    accuracy = sum([panel.sigma*panel.length for panel in panelS])
    #print('sum of singularity strengths: {:0.6f}'.format(accuracy))
    
    # what is the value of the lift for the system (main + flap)
    # assume that P_inf = 0, density = 1 => use bernouli's equation
    P_inf=0
    den=1
    u_inf=1.0
    
    # the parts needed for the integral of of lift are:
    
    # tangential velocity of the panels
    foil_vt = [panel.vt for panel in panelS[:-(N-1)]]
    flap_vt = [panel.vt for panel in panelS[-(N-1):]]
    
    # length of the panels
    foil_len = [panel.length for panel in panelS[:-(N-1)]]
    flap_len = [panel.length for panel in panelS[-(N-1):]]
    
    # angle of the panels called n*j (beta)
    foil_angle = [panel.beta for panel in panelS[:-(N-1)]]
    flap_angle = [panel.beta for panel in panelS[-(N-1):]]
    
    # need to find the lift of the center of each panel individually and add them up
    
    # foil lift array
    loquat = numpy.empty(N-1, dtype=float)
    for i in range (N-1):
        loquat[i]=-(P_inf+0.5*den*(u_inf**2-foil_vt[i]**2))*foil_len[i]*math.sin(foil_angle[i])
        
    # flap lift array
    mango = numpy.empty(N-1, dtype=float)
    for i in range (N-1):
        mango[i]=-(P_inf+0.5*den*(u_inf**2-flap_vt[i]**2))*flap_len[i]*math.sin(flap_angle[i])
    
    #print(loquat)
    #print(mango)
    
    # summing the lift arrays and ading them up
    L=numpy.sum(loquat)+numpy.sum(mango)
    #print('This is the value of the lift for the system {main+flap} for 100 panels: '+'\n'+str(L_100)+'\n')
    
    return L

def biplane_ind_wings(x_m, y_m, x_n1, y_n,lower_panels_bi, freestream):
    x_bi, y_bi = translate_geo(x_m, y_m, x_n1, y_n)
    N = len(x_m)
    upper_panels_bi = define_panels(x_bi, y_bi, N-1)
    panelS_bi = numpy.concatenate((lower_panels_bi, upper_panels_bi))
    L_bi = solve_biplane(panelS_bi, freestream, N)
    return L_bi