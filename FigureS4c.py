from MEK_public import *
from EB_meanfield import *

import numpy as np
from scipy import linalg
import math

import sympy as sy
from sympy import *

#packages for plotting
import matplotlib
import matplotlib.pyplot as plt

slope_L = 0.25
slope_H = 0.1

N = 100

res2emin = -0.2 #The range of energies of the 2-electron (D) reservoir to be plotted over
res2emax = 0.2
dx = (res2emax-res2emin)/N #energy step size
data_mf  = [] #initiate arrays to store data
data2_mf = []
data3_mf = []
data_mek  = [] #initiate arrays to store data
data2_mek = []
data3_mek = []

for n in range(N):
    net = Network()
    """
    !!!!! Add Cofactors exactly in this order !!!!!
    D (two-e cofactor), L1, L2, H1, H2
    """
    D = Cofactor("D", [-0.4, 0.4])   #inverted
    L1 = Cofactor("L1", [-0.4 + slope_L*1])
    L2 = Cofactor("L2", [-0.4 + slope_L*2])
    H1 = Cofactor("H1", [0.4 - slope_H*1])
    H2 = Cofactor("H2", [0.4 - slope_H*2])

    net.addCofactor(D)
    net.addCofactor(L1)
    net.addCofactor(L2)
    net.addCofactor(H1)
    net.addCofactor(H2)

    net.addConnection(D,L1,10)
    net.addConnection(D,L2,20)
    net.addConnection(D,H1,10)
    net.addConnection(D,H2,20)
    net.addConnection(L1,L2,10) 
    net.addConnection(L1,H1,20) 
    net.addConnection(L1,H2,30) 
    net.addConnection(L2,H1,30)
    net.addConnection(L2,H2,40)
    net.addConnection(H1,H2,10)

    net.addReservoir("Two-electron donor", D, 2, 2, -(res2emin + dx*n), 10**7)
    net.addReservoir("Low-potential reservoir", L2, 1, 1, 0, 10**7)
    net.addReservoir("High-potential reservoir", H2, 1, 1, 0, 10**7)

    net.constructStateList()

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    probability_init = getInitialProbabilityVector(net)       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]

    p_ox, p_sq, p_L1, p_H1, p_L2, p_H2 = symbols('p_ox, p_sq, p_L1, p_H1, p_L2, p_H2')

    eq1 = getRateEquations_L1(net, L1, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    eq2 = getRateEquations_H1(net, H1, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    eq3 = getRateEquations_L2(net, L2, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    eq4 = getRateEquations_H2(net, H2, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    eq5 = getRateEquations_D(net, D, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)[0]
    eq6 = getRateEquations_D(net, D, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)[1]

    RateEqns = [eq1, eq2, eq3, eq4, eq5, eq6]
    pop = getpop(net, RateEqns, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2)
    data_mf.append(getReservoirFlux(net, "Two-electron donor", pop))
    data2_mf.append(getReservoirFlux(net, "Low-potential reservoir", pop))
    data3_mf.append(getReservoirFlux(net, "High-potential reservoir", pop))

    pop_MEK_init = np.zeros(net.adj_num_state)
    pop_MEK_init[0] = 1
    pop_MEK = net.evolve(10, pop_MEK_init)
    data_mek.append(net.getReservoirFlux("Two-electron donor", pop_MEK))
    data2_mek.append(net.getReservoirFlux("Low-potential reservoir", pop_MEK))
    data3_mek.append(net.getReservoirFlux("High-potential reservoir", pop_MEK))


x = np.linspace(res2emin*1000, res2emax*1000, N) #*1000 to convert to meV

plt.rc('font', family='DejaVu Sans')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('text', usetex=True)

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)
ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')

plt.plot(x, data_mf, '#91009B', label='Two-electron reservoir (mean-field)')
plt.plot(x, data2_mf,'#B40005', label='Low-potential reservoir (mean-field)')
plt.plot(x, data3_mf,'#2D00C8', label='High-potential reservoir (mean-field)')
plt.plot(x, data_mek, '#91009B', linestyle='--', label='Two-electron reservoir (exact)')
plt.plot(x, data2_mek,'#B40005', linestyle='--', label='Low-potential reservoir (exact)')
plt.plot(x, data3_mek,'#2D00C8', linestyle='--', label='High-potential reservoir (exact)')

ax.set_xlabel('$\Delta G_{bifurc}$ (meV)',size='x-large')
ax.set_ylabel('Flux (Sec$^{-1}$)',size='x-large') 
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.2)
# plt.legend()
plt.show()
