from protonpump_Jiang import *
from MEK_protonpump import *

import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.ticker as ticker

def getPumpFlux_k12kp_Jiang(x, y):     # x: kappa_12, y: kp
    net = Network()

    # Intrisic free energies are from table 1 of "Kinetic models of redox-coupled proton pumping"
    H1 = Cofactor("H1", [3.83 * 25.7*10**(-3)])    # units converted from kBT -> eV   # proton site 1
    H2 = Cofactor("H2", [8.90 * 25.7*10**(-3)])    # proton site 2
    E3 = Cofactor("E3", [15.0 * 25.7*10**(-3)])    # electron site 3

    net.addCofactor(H1)
    net.addCofactor(H2)
    net.addCofactor(E3)

    net.addConnection(H1, H2, 10)   # Distance does not matter. The rate constants are not distance-dependent

    # Intrinsic rates
    kappa_11 = 7.58 * 10**6    #proton uptake rate from N-side  #reservoir -> cofactor
    kappa_22 = 2.84 * 10**4    #proton uptake rate from P-side
    kappa_33 = 2.77 * 10**4    #electron uptake rate from cytochrome c
    # kappa_12 = 5.61 * 10**4
    # kp = 10**5
    kappa_12 = x
    kp = y

    #Membrane potential
    Vm = 0.1

    # Electrostatic interactions
    eps = np.zeros((net.num_cofactor, net.num_cofactor), dtype = float)
    eps[0][1] = eps[1][0] = 12.4 * 25.7*10**(-3)
    eps[0][2] = eps[2][0] = -15.0 * 25.7*10**(-3)
    eps[1][2] = eps[2][1] = -22.5 * 25.7*10**(-3)

    # Intrinsic site free energies
    energy_H1_o = H1.energy[0]
    energy_H2_o = H2.energy[0]
    energy_E3_o = E3.energy[0]

    net.addReservoir("N-side", H1, 1, 1, -H1.energy[0], net.getRate(kappa_11, -H1.energy[0]))    # Proton drain of proton site H1
    net.addReservoir("P-side", H2, 1, 1, -H2.energy[0], kappa_22)    # proton site H2 pumps protons to P-side
    net.addReservoir("CytC", E3, 1, 1, -E3.energy[0], net.getRate(kappa_33, -E3.energy[0]))    # Cytochrome C feeds electrons to electron site E3
    net.addReservoir("oxygen1", H1, 1, 1, -0.1, kp)
    net.addReservoir("oxygen2", E3, 1, 1, -0.1+(E3.energy[0]-H1.energy[0]), kp)    # oxygen 1 and oxygen 2 are the same reservoir. Electrons can be drained when sites H1 and E3 are occupied simultaneously

    #### Self-consistent mean-field calculation ####
    ## The site energies (and thus the rate constants) changes depending on site occupations ##

    # Initial guess of pop = [p_H1, p_H2, p_E3]
    net.constructStateList()
    net.constructAdjacencyMatrix()
    net.constructRateMatrix(H1, H2, E3, kappa_11, kappa_22, kappa_33, kappa_12, kp)
    net.constructRateMatrix_Mod(H1, H2, E3, eps, kappa_11, kappa_22, kappa_33, kappa_12, kp, Vm)

    pop_init = np.zeros(net.adj_num_state)
    pop_init[0] = 1
    pop_MEK = net.evolve_mod(10, pop_init)
    pop = getInitialVector(net, pop_MEK, H1, H2, E3)

    std_data = []   #Stores % deviation from every iteration
    iteration = 0
    while True:
        iteration += 1
        # print('######### iteration =', iteration, '############')
        # print('old pop =', pop)
        # Construct rate equation
        # Different pop gives different rate constants so this process is inside the while loop
        def rateEquations(pop):
            eq_H1 = get_RateEquation_H1(net, H1, H2, E3, kappa_11, kappa_12, kp, energy_H1_o, energy_H2_o, pop, eps, Vm)
            eq_H2 = get_RateEquation_H2(net, H1, H2, kappa_22, kappa_12, energy_H1_o, energy_H2_o, pop, eps, Vm)
            eq_E3 = get_RateEquation_E3(net, H1, E3, kappa_33, kp, energy_E3_o, pop, eps, Vm)

            return (eq_H1, eq_H2, eq_E3)

        # Do the mean-field calculation! Find pop_new based on your pop_old
        pop_new = getpop(net, rateEquations, pop)
        #print('new pop =', pop_new)

        # Check how different pop_new is from your old pop
        std = 0
        for cof_id in range(net.num_cofactor):
            std += abs(pop_new[cof_id] - pop[cof_id])/pop[cof_id] * 100  # % deviation
        std_data.append(std)   #Stores % deviation from every iteration

        # Is the difference between pop_new and pop within given tolerance?
        tolerance = 10**(-20)
        N = 4
        if std < tolerance:    # Percentage deviation is smaller than given tolerance
            Jpump = getPumpFlux(net, H2, kappa_22, energy_H2_o, pop_new, eps, Vm)
            Jelec = getElectronFlux(net, E3, kappa_33, energy_E3_o, pop_new, eps, Vm)
            Jprod = getProductFlux(net, H1, E3, kp, pop_new)
            Jup = getUpFlux(net, H1, kappa_11, energy_H1_o, pop_new, eps, Vm)
            #if abs((Jup-Jpump)-Jelec) < 10**(-5) and abs(Jelec - Jprod) < 10**(-5) and abs(Jpump) < 35 and abs(Jelec) < 2000:
            if len(std_data) < 5:
                pop = pop_new
            else:
                if std_data[-2]<tolerance and std_data[-3]<tolerance and std_data[-4]<tolerance and std_data[-5]<tolerance:   # Requires N consecutive %deviation < tolerance to find the right solution
                    # print('Found p_H1, p_H2, p_E3 at steady state!')
                    break
                else:
                    # print('Found deviation<tolerance, but not for', N, 'consecutive iterations')
                    pop = pop_new
            # else:
            #     print('Unphysical(?) solutions. Try starting from different initial conditions...')
            #     pop = getProbabilityVector(net)
        else:
            # print('pop_new != pop, Another iteration...')
            pop = pop_new

    Jpump = getPumpFlux(net, H2, kappa_22, energy_H2_o, pop_new, eps, Vm)
    Jelec = getElectronFlux(net, E3, kappa_33, energy_E3_o, pop_new, eps, Vm)
    Jprod = getProductFlux(net, H1, E3, kp, pop_new)
    Jup = getUpFlux(net, H1, kappa_11, energy_H1_o, pop_new, eps, Vm)

    # print('Jpump=', Jpump)
    # print('Jelec=', Jelec)
    # print('Jprod=', Jprod)
    # print('Jup-Jpump=', Jup-Jpump)

    print('pop=',pop_new, 'Jpump=',Jpump, 'kappa_12=',kappa_12,'kp=',kp)

    return Jpump

def getPumpFlux_k12kp_Jiang_Matrix(X, Y):
    Z = []
    for i in range(len(Y)):
        y = Y[i][0]
        Z_sub = []
        for j in range(len(X)):
            x = X[0][j]
            Z_sub.append(getPumpFlux_k12kp_Jiang(x, y))
        Z.append(Z_sub)

    return Z

def getPumpFlux_k12kp_MEK(x, y):     # x: kappa_12, y: kp
    net = Network()

    # Intrisic free energies are from table 1 of "Kinetic models of redox-coupled proton pumping"
    H1 = Cofactor("H1", [3.83 * 25.7*10**(-3)])    # units converted from kBT -> eV   # proton site 1
    H2 = Cofactor("H2", [8.90 * 25.7*10**(-3)])    # proton site 2
    E3 = Cofactor("E3", [15.0 * 25.7*10**(-3)])    # electron site 3

    net.addCofactor(H1)
    net.addCofactor(H2)
    net.addCofactor(E3)

    net.addConnection(H1, H2, 10)   # Distance does not matter. The rate constants are not distance-dependent

    # Intrinsic rates
    kappa_11 = 7.58 * 10**6    #proton uptake rate from N-side  #reservoir -> cofactor
    kappa_22 = 2.84 * 10**4    #proton uptake rate from P-side
    kappa_33 = 2.77 * 10**4    #electron uptake rate from cytochrome c
    # kappa_12 = 5.61 * 10**4
    # kp = 10**5
    kappa_12 = x
    kp = y

    #Membrane potential
    Vm = 0.1    # in eV

    net.addReservoir("N-side", H1, 1, 1, -H1.energy[0], net.getRate(kappa_11, -H1.energy[0]))    # Proton drain of proton site H1
    net.addReservoir("P-side", H2, 1, 1, -H2.energy[0], kappa_22)    # proton site H2 pumps protons to P-side
    net.addReservoir("CytC", E3, 1, 1, -E3.energy[0], net.getRate(kappa_33, -E3.energy[0]))    # Cytochrome C feeds electrons to electron site E3
    net.addReservoir("oxygen1", H1, 1, 1, -0.1, kp)
    net.addReservoir("oxygen2", E3, 1, 1, -0.1+(E3.energy[0]-H1.energy[0]), kp)    # oxygen 1 and oxygen 2 are the same reservoir. Electrons can be drained when sites H1 and E3 are occupied simultaneously

    # Electrostatic interactions
    eps = np.zeros((net.num_cofactor, net.num_cofactor), dtype = float)
    # eps[0][1] = eps[1][0] = 12.4 * 25.7*10**(-3)
    # eps[0][2] = eps[2][0] = -15.0 * 25.7*10**(-3)
    # eps[1][2] = eps[2][1] = -22.5 * 25.7*10**(-3)

    net.constructStateList()
    net.constructAdjacencyMatrix()

    pop_init = np.zeros(net.adj_num_state)
    pop_init[0] = 1

    net.constructRateMatrix(H1, H2, E3, kappa_11, kappa_22, kappa_33, kappa_12, kp)
    #print(net.K)
    net.constructRateMatrix_Mod(H1, H2, E3, eps, kappa_11, kappa_22, kappa_33, kappa_12, kp, Vm)
    #print(net.modK)
    pop_MEK = net.evolve_mod(10, pop_init)
    Jpump = net.getPumpFlux_Mod(H2, pop_MEK)
    Jelec = net.getElectronFlux_Mod(E3, pop_MEK)
    Jprod = net.getProductFlux_Mod(H1, E3, pop_MEK)
    Jup = net.getUpFlux_Mod(H1, pop_MEK)

    print('Jpump=', Jpump)
    print('Jelec=', Jelec)
    print('Jprod=', Jprod)
    print('Jup-Jpump=', Jup-Jpump)

    return Jpump

def getPumpFlux_k12kp_MEK_Matrix(X, Y):
    Z = []
    for i in range(len(Y)):
        y = Y[i][0]
        Z_sub = []
        for j in range(len(X)):
            x = X[0][j]
            Z_sub.append(getPumpFlux_k12kp_MEK(x, y))
        Z.append(Z_sub)

    return Z

No = 40
kappa_12_rate = np.logspace(2, 7, No)
kp_rate = np.logspace(5, 9, No)
X, Y = np.meshgrid(kappa_12_rate, kp_rate)
Z1 = getPumpFlux_k12kp_Jiang_Matrix(X, Y)
Z2 = getPumpFlux_k12kp_MEK_Matrix(X, Y)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

# color_levels = np.arange(-2*10**(-3), 0+2*10**(-3)/10, 10)
color_levels = np.linspace(-2*10**(-3), 0, 10)
color_levels = [-2*10**(-3), -8*10**(-4), -4*10**(-4), -2*10**(-4), -8*10**(-5), -4*10**(-5), -8*10**(-6), -4*10**(-6), 0]


fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 5))

## Jiang et al approach plots ###
# jiang = ax1.contourf(X, Y, Z1, cmap='PuOr_r', levels=color_levels, norm=colors.SymLogNorm(linthresh=10**(-5), linscale=10**(-5)))
jiang = ax1.contourf(X, Y, Z1, cmap='Blues_r', levels=color_levels, norm=colors.SymLogNorm(linthresh=10**(-6), linscale=10**(-6)))
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$\kappa_{12}$ (sec$^{-1})$')
ax1.set_ylabel('$k_{p}$ (sec$^{-1})$')

## MEK plots ##
# mek = ax2.contourf(X, Y, Z2, cmap='PuOr_r', levels=color_levels, norm=colors.SymLogNorm(linthresh=10**(-5), linscale=10**(-5)))
mek = ax2.contourf(X, Y, Z2, cmap='Blues_r', levels=color_levels, norm=colors.SymLogNorm(linthresh=10**(-6), linscale=10**(-6)))
ax2.set_xscale('log')
ax2.set_yscale('log') 
ax2.set_xlabel('$\kappa_{12}$ (sec$^{-1})$')
ax2.set_ylabel('$k_{p}$ (sec$^{-1})$')


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%.3f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format

## Color bars ##
cbar = fig.colorbar(mek, ax=[ax1, ax2], orientation='vertical', format=OOMFormatter(-3, mathText=False))
cbar.set_label('$J_{pump}$ (sec$^{-1})$', labelpad=3)
plt.show()
