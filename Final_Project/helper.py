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
    b[:,:-2] = B_vortex[0,:] + B_vortex[(N-2),:]
    b[:,-2] = - numpy.sum(A_source[0,:-(N-1)]) - numpy.sum(A_source[(N-2),:-(N-1)])
    b[:,-1] = - numpy.sum(A_source[0,-(N-1):]) - numpy.sum(A_source[(N-2),-(N-1):])
    
    return b 

def kutta_condition_2(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+2), dtype=float)
    
    # i = row that we need to work with
    b[:,:-2] = B_vortex[(N-1),:] + B_vortex[-1,:]
    b[:,-2] = - numpy.sum(A_source[(N-1),:-(N-1)]) - numpy.sum(A_source[-1,:-(N-1)])
    b[:,-1] = - numpy.sum(A_source[(N-1),-(N-1):]) - numpy.sum(A_source[-1,-(N-1):])
    
    return b 

# create the singularity matrix (an amalgamation of the kutta condition, source term, and the vortex term)
def build_singularity_matrix(A_source, B_vortex, N):

    A = numpy.empty((A_source.shape[0] + 2, A_source.shape[1] + 2), dtype=float)
    # source contribution matrix
    A[:-2, :-2] = A_source
    # vortex contribution array 1
    panda=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        panda[i,:]=numpy.sum(B_vortex[i,:-(N-1)])
    A[:-2, -2] = panda[:,0]
    # vortex contribution array 2
    dolphin=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        dolphin[i,:]=numpy.sum(B_vortex[i,-(N-1):])
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
                                 numpy.sin(freestream.alpha - panelS[(N-2)].beta) )
    b[-1] = -freestream.u_inf * (numpy.sin(freestream.alpha - panelS[N-1].beta) +
                                 numpy.sin(freestream.alpha - panelS[-1].beta) )
    return b

def compute_tangential_velocity(panelS, freestream, gamma, A_source, B_vortex, N):
    
    A = numpy.empty((len(panelS), len(panelS) + 2), dtype=float)
    
    # B_vortex contribution
    A[:, :-2] = B_vortex

    # A_source contribution from foil
    pangolin=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        pangolin[i,:]=-numpy.sum(A_source[i,:-(N-1)])
    A[:, -2] = pangolin[:,0]

    # A_source contribution from flap
    lemur=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        lemur[i,:]=-numpy.sum(A_source[i,-(N-1):])
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

def solve_biplane(panelS, freestream, N): # fix foil_panelS
    
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

def solve_monoplane(panelS, freestream, N): # fix foil_panelS
    
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

def alt_monplane(panelS, freestream, N):
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
    lift = gamma * sum(panel.length for panel in panelS)
    return lift

#---------------------------------------------------------#
#---------create the linear system for the biplane--------#
#---------------------------------------------------------#

# create the kutta conditions
# 1st kutta condition will go from 1 to (N-1)/3
# 2nd kutta condition will go from 1+(N-1)/3 to 2(N-1)/3
# 3rd kutta condition will go from 1+2(N-1)/3 to (N-1)
def tri_kutta_condition_1(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+2), dtype=float)
    
    # i = row that we need to work with
    b[:,:-2] = B_vortex[0,:] + B_vortex[(N-2),:]
    b[:,-2] = - numpy.sum(A_source[0,:-(N-1)]) - numpy.sum(A_source[(N-2),:-(N-1)])
    b[:,-1] = - numpy.sum(A_source[0,-(N-1):]) - numpy.sum(A_source[(N-2),-(N-1):])
    
    return b 

def tri_kutta_condition_2(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+2), dtype=float)
    
    # i = row that we need to work with
    b[:,:-2] = B_vortex[(N-1),:] + B_vortex[-1,:]
    b[:,-2] = - numpy.sum(A_source[(N-1),:-(N-1)]) - numpy.sum(A_source[-1,:-(N-1)])
    b[:,-1] = - numpy.sum(A_source[(N-1),-(N-1):]) - numpy.sum(A_source[-1,-(N-1):])
    
    return b 

def tri_kutta_condition_3(A_source, B_vortex, N):
    b = numpy.empty((1,A_source.shape[0]+2), dtype=float)
    
    # i = row that we need to work with
    b[:,:-2] = B_vortex[(N-1),:] + B_vortex[-1,:]
    b[:,-2] = - numpy.sum(A_source[(N-1),:-(N-1)]) - numpy.sum(A_source[-1,:-(N-1)])
    b[:,-1] = - numpy.sum(A_source[(N-1),-(N-1):]) - numpy.sum(A_source[-1,-(N-1):])
    
    return b 

# create the singularity matrix (an amalgamation of the kutta condition, source term, and the vortex term)
def build_singularity_matrix(A_source, B_vortex, N):

    A = numpy.empty((A_source.shape[0] + 2, A_source.shape[1] + 2), dtype=float)
    # source contribution matrix
    A[:-2, :-2] = A_source
    # vortex contribution array 1
    panda=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        panda[i,:]=numpy.sum(B_vortex[i,:-(N-1)])
    A[:-2, -2] = panda[:,0]
    # vortex contribution array 2
    dolphin=numpy.empty((B_vortex.shape[0], 1), dtype=float)
    for i in range (len(B_vortex)):
        dolphin[i,:]=numpy.sum(B_vortex[i,-(N-1):])
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
                                 numpy.sin(freestream.alpha - panelS[(N-2)].beta) )
    b[-1] = -freestream.u_inf * (numpy.sin(freestream.alpha - panelS[N-1].beta) +
                                 numpy.sin(freestream.alpha - panelS[-1].beta) )
    return b

def compute_tangential_velocity(panelS, freestream, gamma, A_source, B_vortex, N):
    
    A = numpy.empty((len(panelS), len(panelS) + 2), dtype=float)
    
    # B_vortex contribution
    A[:, :-2] = B_vortex

    # A_source contribution from foil
    pangolin=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        pangolin[i,:]=-numpy.sum(A_source[i,:-(N-1)])
    A[:, -2] = pangolin[:,0]

    # A_source contribution from flap
    lemur=numpy.empty((A_source.shape[0], 1), dtype=float)
    for i in range (len(A_source)):
        lemur[i,:]=-numpy.sum(A_source[i,-(N-1):])
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

def solve_biplane(panelS, freestream, N): # fix foil_panelS
    
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
