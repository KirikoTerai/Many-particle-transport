from MEK_public import *
from EB_meanfield import *

import numpy as np
from scipy import linalg
import math

import sympy as sy
from sympy import *
#from sympy.solvers import checksol

#packages for plotting
import matplotlib
import matplotlib.pyplot as plt

import sys
import os

N = 50
beta_min = 5
beta_max = 38.91*1.7
beta_step = (beta_max - beta_min)/N #energy step size

slope = 0.15

flux_MF_list = []
flux_MEK = []
krl_orange_list = []
krl_green_list = []
flux_orange_list = []
flux_green_list = []
flux_average = []

for n in range(N):
    net = Network()
    net.beta = beta_min + n*beta_step
    """
    !!!!! Add Cofactors exactly in this order !!!!!
    D (two-e cofactor), L1, L2, H1, H2
    """ 

    D = Cofactor("D", [-0.4, 0.4])
    L1 = Cofactor("L1", [-0.4 + slope*1])
    L2 = Cofactor("L2", [-0.4 + slope*2])
    H1 = Cofactor("H1", [0.4 - slope*1])
    H2 = Cofactor("H2", [0.4 - slope*2])

    net.addCofactor(D)
    net.addCofactor(L1)
    net.addCofactor(L2)
    net.addCofactor(H1)
    net.addCofactor(H2)

    net.addConnection(D,L1,10)
    net.addConnection(D,L2,20)
    net.addConnection(D,H1,10)
    net.addConnection(D,H2,20)
    net.addConnection(L1,L2,10)   # Case 2
    net.addConnection(L1,H1,20)   # Case 1
    net.addConnection(L1,H2,30)   # Case 2
    net.addConnection(L2,H1,30)   # Case 2
    net.addConnection(L2,H2,40)
    net.addConnection(H1,H2,10)   # Case 2

    net.addReservoir("Two-electron donor", D, 2, 2, 0, 10**7)
    net.addReservoir("Low-potential reservoir", L2, 1, 1, 0, 10**7)
    net.addReservoir("High-potential reservoir", H2, 1, 1, 0, 10**7)

    net.constructStateList()

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    ###################
    ### Mean-Field ####
    p_ox, p_sq, p_L1, p_H1, p_L2, p_H2 = symbols('p_ox, p_sq, p_L1, p_H1, p_L2, p_H2')

    eq1 = getRateEquations_L1(net, L1, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    eq2 = getRateEquations_H1(net, H1, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    eq3 = getRateEquations_L2(net, L2, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    eq4 = getRateEquations_H2(net, H2, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    eq5 = getRateEquations_D(net, D, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)[0]
    eq6 = getRateEquations_D(net, D, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)[1]

    RateEqns = [eq1, eq2, eq3, eq4, eq5, eq6]
    pop = getpop(net, RateEqns, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)

    flux_MF = getReservoirFlux(net, "High-potential reservoir", pop)
    zeroflux_MF = getReservoirFlux(net, "Two-electron donor", pop)
    flux_MF_list.append(flux_MF)

    pop_MEK_init = np.zeros(net.adj_num_state)
    pop_MEK_init[0] = 1
    pop_MEK = net.evolve(10, pop_MEK_init)

    flux_MEK.append(net.getReservoirFlux("High-potential reservoir", pop_MEK))

    krl_orange = getHoppingRates(net, L1, D, 1)[0]       # getHoppingRates(net, cof_i: Cofactor, cof_f: Cofactor, cof_i_initial_redox: int)
    krl_orange_list.append(krl_orange)
    #print('krl_orange=', krl_orange)
    flux_orange = 2 * krl_orange * np.exp(-2*slope*net.beta)
    flux_orange_list.append(flux_orange)

    krl_green = getHoppingRates(net, L1, H1, 1)[0]
    krl_green_list.append(krl_green)
    flux_green = krl_green * np.exp(-2*slope*net.beta)
    flux_green_list.append(flux_green)

    #average = 100*np.sqrt(flux_orange * flux_green)
    average = np.sqrt(flux_orange * flux_green)
    flux_average.append(average)

x = np.linspace(beta_min, beta_max, N)

plt.rc('font', family='DejaVu Sans')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('text', usetex=True)

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)
ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
plt.plot(x, flux_MF_list,  color = 'darkviolet', label='Mean-field')
plt.plot(x, flux_MEK, color = 'black', linestyle='--', label='Exact')
plt.plot(x, flux_orange_list,  color = 'orange', linestyle='--', label='$J1$')
plt.plot(x, flux_green_list,  color = 'green', linestyle='--', label='$J2$')
plt.plot(x, flux_average,  color = 'hotpink', label='$J3$')
plt.yscale('log')

ax.set_xlabel('$1/k_{B}T$ (eV$^{-1}$)',size='x-large')
ax.set_ylabel('Flux (Sec$^{-1}$)',size='x-large') 
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.2)
plt.legend(loc='upper right')
plt.show()
