from protonpump_meanfield import *
from MEK_protonpump import *

import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.ticker as ticker

def getPumpFlux_k12kp(x, y):     # x: kappa_12, y: kp
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
            eq_H1 = get_RateEquation_H1(net, H1, H2, E3, kp, pop)
            eq_H2 = get_RateEquation_H2(net, H1, H2, E3, kp, pop)
            eq_E3 = get_RateEquation_E3(net, H1, H2, E3, kp, pop)

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
            Jpump = getPumpFlux(net, H1, H2, E3, kp, pop_new)
            Jelec = getElectronFlux(net, H1, H2, E3, kp, pop_new)
            Jprod = getProductFlux(net, H1, H2, E3, kp, pop_new)
            Jup = getUpFlux(net, H1, H2, E3, kp, pop_new)
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

    # print('solution=', pop_new)
    Jpump = getPumpFlux(net, H1, H2, E3, kp, pop_new)
    Jelec = getElectronFlux(net, H1, H2, E3, kp, pop_new)
    Jprod = getProductFlux(net, H1, H2, E3, kp, pop_new)
    Jup = getUpFlux(net, H1, H2, E3, kp, pop_new)

    # print('Jpump=', Jpump)
    # print('Jelec=', Jelec)
    # print('Jprod=', Jprod)
    # print('Jup-Jpump=', Jup-Jpump)

    print('pop=',pop_new, 'Jpump=',Jpump, 'kappa_12=',kappa_12,'kp=',kp)

    return Jpump

def getPumpFlux_k12kp_Matrix(X, Y):
    Z = []
    for i in range(len(Y)):
        y = Y[i][0]
        Z_sub = []
        for j in range(len(X)):
            x = X[0][j]
            """
            The mean-field model involves solving a high-order polynomials. There are multiple solutions to the polynomial that is physically correct (i.e. 0 < probability < 1).
            However, in this study, we always find one solution such that 0 < probability < 1 and Jpump is in the right order of magnitude.
            """
            success = False 
            while success == False:
                try:
                    Jpump = getPumpFlux_k12kp(x, y)     # getpop() in the function file ensures solutions such that 0 < probability < 1 are picked
                    print('Jpump=', Jpump)        
                    if abs(Jpump) < 100:    # This ensures that Jpump is in the right order of magnitude. This value is specific to this simulation.
                                            # It is necessary to explore what the right order of magnitude is before setting this number.
                        success = True
                        print('Picked the correct solution!')
                        Z_sub.append(Jpump)
                    else:
                        success = False
                        print('Picked the wrong solution...')
                except ValueError:
                    success = False
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
    eps[0][1] = eps[1][0] = 12.4 * 25.7*10**(-3)
    eps[0][2] = eps[2][0] = -15.0 * 25.7*10**(-3)
    eps[1][2] = eps[2][1] = -22.5 * 25.7*10**(-3)

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

    # print('Jpump=', Jpump)
    # print('Jelec=', Jelec)
    # print('Jprod=', Jprod)
    # print('Jup-Jpump=', Jup-Jpump)

    pop = [net.population(pop_MEK, H1, 1), net.population(pop_MEK, H2, 1), net.population(pop_MEK, E3, 1)]
    print('pop=',pop, 'Jpump=',Jpump, 'kappa_12=',kappa_12,'kp=',kp)

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
Z1 = getPumpFlux_k12kp_Matrix(X, Y)
Z2 = getPumpFlux_k12kp_MEK_Matrix(X, Y)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

color_levels = np.arange(-80, 20+10,10)

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 5))

## MEK plots ###
mek = ax1.contourf(X, Y, Z2, cmap='RdBu_r', levels=color_levels, norm=colors.SymLogNorm(linthresh=2.0, linscale=1.0))
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$\kappa_{12}$ (sec$^{-1})$')
ax1.set_ylabel('$k_{p}$ (sec$^{-1})$')

## Mean-field plots ##
mf = ax2.contourf(X, Y, Z1, cmap='RdBu_r', levels=color_levels, norm=colors.SymLogNorm(linthresh=2.0, linscale=1.0))
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('$\kappa_{12}$ (sec$^{-1})$')
ax2.set_ylabel('$k_{p}$ (sec$^{-1})$')

## Color bars ##
cbar = fig.colorbar(mf, ax=[ax1, ax2], orientation='vertical')
cbar.set_label('$J_{pump}$ (sec$^{-1})$', labelpad=3)
plt.show()
