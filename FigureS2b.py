from linear_meanfield import *
from linear_Jiang import *
from MEK_linear import *

import numpy as np
import pandas as pd

# Reorganization energies and 
reorgE_dict = {'λ_AB': 1080*10**(-3), 'λ_BC': 760*10**(-3), 'λ_CD': 880*10**(-3)}   #unit in eV
coupling_dict = {'V_AB': 2.17*10**(-3), 'V_BC': 3.08*10**(-3), 'V_CD': 2.08*10**(-3)}   #unit in eV


N = 100
res2emax = 0.3
res2emin = -0.3
dx = (res2emax-res2emin)/N
data1_mek = []
data2_mek = []
data1_mf = []
data2_mf = []
data1_mf_1 = []
data2_mf_1 = []
x_list = []

for n in range(N):
    net = Network()

    # All Cys linkages are included in the "final model"
    # Redox potentials of the hemes of the STC protein. Reference is table 3 of DOI:10.1007/s00775-008-0455-7
    A = Cofactor('A', [-243 * 10**(-3)])   # unit is eV
    B = Cofactor('B', [-222 * 10**(-3)])   #21, 33, 18
    C = Cofactor('C', [-189 * 10**(-3)])
    D = Cofactor('D', [-171 * 10**(-3)])

    net.addCofactor(A)
    net.addCofactor(B)
    net.addCofactor(C)
    net.addCofactor(D)

    # Distance between hemes are from table S3 of https://doi.org/10.1021/jacs.7b08831
    net.addConnection(A, B, 3.97)
    net.addConnection(B, C, 4.14)
    net.addConnection(C, D, 4.12)

    # net.addReservoir('Ares', A, 1, 1, -(0.172+res2emax)+dx*n, net.getRate(10**8, -(0.172+res2emax)+dx*n))
    net.addReservoir('Ares', A, 1, 1, 0.1-dx*n, net.getRate(10**8, 0.1-dx*n))
    net.addReservoir('Dres', D, 1, 1, -0.2, 10**8)
    x_list.append(((-D.redox[0]-0.2) - (-A.redox[0]+(0.1-dx*n)))*1000)

    net.constructStateList()
    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    # eps[b][a] is the change of reduction potential of heme a if heme b is in the oxidized state
    # Reference is table 3 of DOI:10.1007/s00775-008-0455-7
    eps=np.zeros((net.num_cofactor, net.num_cofactor), dtype = float)
    # eps[0][1]=eps[1][0]=28*10**(-3)    # unit is eV
    # eps[0][2]=eps[2][0]=21*10**(-3)
    # eps[0][3]=eps[3][0]=11*10**(-3)
    # eps[1][2]=eps[2][1]=72*10**(-3)
    # eps[1][3]=eps[3][1]=11*10**(-3)
    # eps[2][3]=eps[3][2]=29*10**(-3)

    net.constructRateMatrix_Mod(eps, reorgE_dict, coupling_dict)
    # print(net.modK)

    # Evolve by modified Rate Matrix
    pop_init_Mod = np.zeros(net.adj_num_state)
    pop_init_Mod[0] = 1
    pop_Mod = net.evolve_Mod(10, pop_init_Mod)

    data1_mek.append(net.getReservoirFlux_Mod('Ares', pop_Mod))
    data2_mek.append(net.getReservoirFlux_Mod('Dres', pop_Mod))

    #### Self-consistent mean-field calculation ####
    ## The site energies (and thus the rate constants) changes depending on site occupations ##

    # Initial guess of pop = [p_A, p_B, p_C]
    #pop = np.zeros(net.num_cofactor)
    pop = getInitialVector(net, pop_Mod, A, B, C, D)
    # print(pop)
    
    ### True mean-field
    std_data = []   #Stores % deviation from every iteration

    iteration = 0
    while True:
        iteration += 1
        #print('######### iteration =', iteration, '############')
        #print('old pop =', pop)
        # Construct rate equation
        # Different pop gives different rate constants so this process is inside the while loop
        def rateEquations(pop):
            eq_A = get_RateEquation_A(net, A, B, C, D, pop)
            eq_B = get_RateEquation_B(net, A, B, C, D, pop)
            eq_C = get_RateEquation_C(net, A, B, C, D, pop)
            eq_D = get_RateEquation_D(net, A, B, C, D, pop)

            return (eq_A, eq_B, eq_C, eq_D)
        
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
        if std < tolerance:    # Percentage deviation is smaller than given tolerance
            if len(std_data) < 5:
                pop = pop_new
            else:
                if std_data[-2]<tolerance and std_data[-3]<tolerance and std_data[-4]<tolerance and std_data[-5]<tolerance:   # Requires N consecutive %deviation < tolerance to find the right solution
                    #print('Found p_A, p_B, p_C, p_D at steady state!')
                    break
                else:
                    #print('Found deviation<tolerance, but not for', 4, 'consecutive iterations')
                    pop = pop_new
        else:
            #print('pop_new != pop, Another iteration...')
            pop = pop_new
            
    print("solution=", pop_new, 'Ares=', getReservoirFlux(net, 'Ares', pop_new, eps), 'Dres=', getReservoirFlux(net, 'Dres', pop_new, eps))
    data1_mf.append(getReservoirFlux(net, 'Ares', pop_new, eps))
    data2_mf.append(getReservoirFlux(net, 'Dres', pop_new, eps))
    
    ### Jiang et al ###
    std_data_1 = []   #Stores % deviation from every iteration

    iteration_1 = 0
    while True:
        iteration_1 += 1
        #print('######### iteration =', iteration, '############')
        #print('old pop =', pop)
        # Construct rate equation
        # Different pop gives different rate constants so this process is inside the while loop
        def rateEquations_1(pop):
            eq_A_1 = get_RateEquation_A_1(net, A, B, pop, eps, reorgE_dict, coupling_dict)
            eq_B_1 = get_RateEquation_B_1(net, A, B, C, pop, eps, reorgE_dict, coupling_dict)
            eq_C_1 = get_RateEquation_C_1(net, B, C, D, pop, eps, reorgE_dict, coupling_dict)
            eq_D_1 = get_RateEquation_D_1(net, C, D, pop, eps, reorgE_dict, coupling_dict)

            return (eq_A_1, eq_B_1, eq_C_1, eq_D_1)
        
        # Do the mean-field calculation! Find pop_new based on your pop_old
        pop_new_1 = getpop_1(net, rateEquations_1, pop)
        #print('new pop =', pop_new)

        # Check how different pop_new is from your old pop
        std_1 = 0
        for cof_id in range(net.num_cofactor):
            std_1 += abs(pop_new_1[cof_id] - pop[cof_id])/pop[cof_id] * 100  # % deviation
        std_data_1.append(std_1)   #Stores % deviation from every iteration

        # Is the difference between pop_new and pop within given tolerance?
        tolerance = 10**(-20)
        if std_1 < tolerance:    # Percentage deviation is smaller than given tolerance
            if len(std_data_1) < 5:
                pop = pop_new_1
            else:
                if std_data_1[-2]<tolerance and std_data_1[-3]<tolerance and std_data_1[-4]<tolerance and std_data_1[-5]<tolerance:   # Requires N consecutive %deviation < tolerance to find the right solution
                    #print('Found p_A, p_B, p_C, p_D at steady state!')
                    break
                else:
                    #print('Found deviation<tolerance, but not for', 4, 'consecutive iterations')
                    pop = pop_new_1
        else:
            #print('pop_new != pop, Another iteration...')
            pop = pop_new_1
            
    print("solution=", pop_new_1)
    data1_mf_1.append(getReservoirFlux_1(net, 'Ares', pop_new_1, eps))
    data2_mf_1.append(getReservoirFlux_1(net, 'Dres', pop_new_1, eps))

plt.rc('font', family='DejaVu Sans')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('text', usetex=True)

fig, ax1 = plt.subplots()
fig.set_size_inches([8,6])

ax1.plot(x_list, data2_mf, color = 'black')
ax1.plot(x_list, data2_mf_1, color = 'blue', linestyle = '--')
ax1.plot(x_list, data2_mek, color = 'red', linestyle = '--')
ax1.legend(["Reservoir IV (mean-field)", "Reservoir IV (Jiang $et$ $al$'s approach)", "Reservoir IV (exact)"], loc='best', prop={'size': 16})
# ax1.legend(["Reservoir IV (mean-field)", "Reservoir IV (exact)"], loc='best', prop={'size': 16})
ax1.set_xlabel('$\Delta G$ (meV)',size='xx-large')
ax1.set_ylabel('Flux (Sec$^{-1}$)',size='xx-large')

left, bottom, width, height = [0.6, 0.4, 0.25, 0.25]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x_list, data2_mf, color = 'black')
ax2.plot(x_list, data2_mf_1, color = 'blue', linestyle = '--')
ax2.plot(x_list, data2_mek, color = 'red', linestyle = '--')
ax2.set_xlabel('$\Delta G$ (meV)',size='xx-large')
ax2.set_ylabel('Flux (Sec$^{-1}$)',size='xx-large')
ax2.set_xlim([-50, 50])
ax2.set_ylim([-10000, 10000])

plt.show()
