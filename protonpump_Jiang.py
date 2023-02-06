import numpy as np
from sympy import *
from scipy.optimize import *

from MEK_protonpump import *

def getSiteEnergy_H1(net, H1: Cofactor, energy_H1_o: float, pop: np.array, eps: np.array, Vm: float):
    cof_H1_id = net.cofactor2id[H1]
    q1 = 1
    Z1 = 15
    L = 30
    energy_H1 = energy_H1_o + (q1*Z1*Vm)/L   # energy_H1_o is the free energy of site H1 when sites H2 and E3 are not occupied
                                             # (q1*Z1*Vm)/L is the contribution of membrane potential
    for cof_id in range(net.num_cofactor):
        if cof_id != cof_H1_id:
            energy_H1 += pop[cof_id]*eps[cof_H1_id][cof_id]   # Contribution of electrostatic interactions to the site energy 

    return energy_H1

def getSiteEnergy_H2(net, H2: Cofactor, energy_H2_o: float, pop: np.array, eps: np.array, Vm: float):
    cof_H2_id = net.cofactor2id[H2]
    q2 = 1
    Z2 = 25
    L = 30
    energy_H2 = energy_H2_o + (q2*Z2*Vm)/L   # energy_H2_o is the free energy of site H2 when sites H1 and E3 are not occupied
                                             # (q2*Z2*Vm)/L is the contribution of membrane potential
    for cof_id in range(net.num_cofactor):
        if cof_id != cof_H2_id:
            energy_H2 += pop[cof_id]*eps[cof_H2_id][cof_id]   # Contribution of electrostatic interactions to the site energy 
                
    return energy_H2

def getSiteEnergy_E3(net, E3: Cofactor, energy_E3_o: float, pop: np.array, eps: np.array, Vm: float):
    cof_E3_id = net.cofactor2id[E3]
    q3 = -1
    Z3 = 20
    L = 30
    energy_E3 = energy_E3_o + (q3*Z3*Vm)/L   # energy_E3_o is the free energy of site E3 when sites H1 and H2 are not occupied
                                             # (q3*Z3*Vm)/L is the contribution of membrane potential
    for cof_id in range(net.num_cofactor):
        if cof_id != cof_E3_id:
            energy_E3 += pop[cof_id]*eps[cof_E3_id][cof_id]   # Contribution of electrostatic interactions to the site energy 
                
    return energy_E3

def get_Rate_H1H2(net, H1: Cofactor, H2: Cofactor, kappa_12: float, energy_H1_o: float, energy_H2_o: float, pop: np.array, eps: np.array, Vm: float):   #rate for H1 <-> H2 proton step
    G_H2 = getSiteEnergy_H2(net, H2, energy_H2_o, pop, eps, Vm)
    G_H1 = getSiteEnergy_H1(net, H1, energy_H1_o, pop, eps, Vm)
    deltaG = G_H2 - G_H1

    k_H1_H2 = []
    k_H1_H2.append(kappa_12 * np.exp(-net.alpha*deltaG*net.beta))  #forward rate H1 -> H2
    k_H1_H2.append(kappa_12 * np.exp((1-net.alpha)*deltaG*net.beta))  #backward rate H2 -> H1

    return k_H1_H2

def get_Rate_NH1(net, H1: Cofactor, kappa_11: float, energy_H1_o: float, pop: np.array, eps: np.array, Vm: float):   #rate for reservoir(N-side) <-> H1 proton step
    G_H1 = getSiteEnergy_H1(net, H1, energy_H1_o, pop, eps, Vm)
    deltaG = G_H1

    k_N_H1 = []
    k_N_H1.append(kappa_11 * np.exp(-net.alpha*deltaG*net.beta))  #forward rate reservoir -> H1
    k_N_H1.append(kappa_11 * np.exp((1-net.alpha)*deltaG*net.beta))  #backward rate H1 -> reservoir

    return k_N_H1

def get_Rate_PH2(net, H2: Cofactor, kappa_22: float, energy_H2_o: float, pop: np.array, eps: np.array, Vm: float):   #rate for reservoir(P-side) <-> H2 proton step
    G_H2 = getSiteEnergy_H2(net, H2, energy_H2_o, pop, eps, Vm)
    deltaG = G_H2

    k_P_H2 = []
    k_P_H2.append(kappa_22 * np.exp(-net.alpha*deltaG*net.beta))  #forward rate reservoir -> H2
    k_P_H2.append(kappa_22 * np.exp((1-net.alpha)*deltaG*net.beta))  #backward rate H2 -> reservoir

    return k_P_H2

def get_Rate_CytE3(net, E3: Cofactor, kappa_33: float, energy_E3_o: float, pop: np.array, eps: np.array, Vm: float):   #rate for reservoir(CytC) <-> E3 electron step
    G_E3 = getSiteEnergy_E3(net, E3, energy_E3_o, pop, eps, Vm)
    deltaG = G_E3

    k_Cyt_E3 = []
    k_Cyt_E3.append(kappa_33 * np.exp(-net.alpha*deltaG*net.beta))  #forward rate reservoir -> E3
    k_Cyt_E3.append(kappa_33 * np.exp((1-net.alpha)*deltaG*net.beta))  #backward rate E3 -> reservoir

    return k_Cyt_E3

#kp is constantly 10**5? No backward rate?

def get_RateEquation_H1_1(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, kappa_11: float, kappa_12: float, kp: float, energy_H1_o: float, energy_H2_o: float, pop: np.array, eps: np.array, Vm: float):
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    cof_E3_id = net.cofactor2id[E3]
    ### Rate constants ###
    kf_H1_H2 = get_Rate_H1H2(net, H1, H2, kappa_12, energy_H1_o, energy_H2_o, pop, eps, Vm)[0]  # H1 -> H2
    kb_H1_H2 = get_Rate_H1H2(net, H1, H2, kappa_12, energy_H1_o, energy_H2_o, pop, eps, Vm)[1]  # H2 -> H1
    kf_N_H1 = get_Rate_NH1(net, H1, kappa_11, energy_H1_o, pop, eps, Vm)[0]  # N-side -> H1
    kb_N_H1 = get_Rate_NH1(net, H1, kappa_11, energy_H1_o, pop, eps, Vm)[1]  # H1 -> N-side
    #kp: rate of H2O production    # H1, E3 -> oxygen  # Assume irreversible
    ### Rate equation of H1: dH1/dt ###
    J_H1 = kb_H1_H2*pop[cof_H2_id]*(1-pop[cof_H1_id]) + kf_N_H1*(1-pop[cof_H1_id]) - kf_H1_H2*pop[cof_H1_id]*(1-pop[cof_H2_id]) - kb_N_H1*pop[cof_H1_id] - kp*pop[cof_H1_id]*pop[cof_E3_id]

    return J_H1

def get_RateEquation_H2_1(net, H1: Cofactor, H2: Cofactor, kappa_22: float, kappa_12: float, energy_H1_o: float, energy_H2_o: float, pop: np.array, eps: np.array, Vm: float):
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    ### Rate constants ###
    kf_H1_H2 = get_Rate_H1H2(net, H1, H2, kappa_12, energy_H1_o, energy_H2_o, pop, eps, Vm)[0]  # H1 -> H2
    kb_H1_H2 = get_Rate_H1H2(net, H1, H2, kappa_12, energy_H1_o, energy_H2_o, pop, eps, Vm)[1]  # H2 -> H1
    kf_P_H2 = get_Rate_PH2(net, H2, kappa_22, energy_H2_o, pop, eps, Vm)[0]  # P-side -> H2
    kb_P_H2 = get_Rate_PH2(net, H2, kappa_22, energy_H2_o, pop, eps, Vm)[1]  # H2 -> P-side
    ### Rate equation of H2: dH2/dt ###
    J_H2 = kf_H1_H2*pop[cof_H1_id]*(1-pop[cof_H2_id]) + kf_P_H2*(1-pop[cof_H2_id]) - kb_H1_H2*pop[cof_H2_id]*(1-pop[cof_H1_id]) - kb_P_H2*pop[cof_H2_id]

    return J_H2

def get_RateEquation_E3_1(net, H1: Cofactor, E3: Cofactor, kappa_33: float, kp: float, energy_E3_o: float, pop: np.array, eps: np.array, Vm: float):
    cof_H1_id = net.cofactor2id[H1]
    cof_E3_id = net.cofactor2id[E3]
    ### Rate constants ###
    kf_Cyt_E3 = get_Rate_CytE3(net, E3, kappa_33, energy_E3_o, pop, eps, Vm)[0]  # CytC -> E3
    kb_Cyt_E3 = get_Rate_CytE3(net, E3, kappa_33, energy_E3_o, pop, eps, Vm)[1]  # E3 -> CytC
    #kp: rate of H2O production    # H1, E3 -> oxygen  # Assume irreversible
    ### Rate equation of E3: dE3/dt ###
    J_E3 = kf_Cyt_E3*(1-pop[cof_E3_id]) - kb_Cyt_E3*pop[cof_E3_id] - kp*pop[cof_H1_id]*pop[cof_E3_id]

    return J_E3

def getPumpFlux_1(net, H2: Cofactor, kappa_22: float, energy_H2_o: float, pop: list, eps: np.array, Vm: float):
    cof_H2_id = net.cofactor2id[H2]
    p_H2 = pop[cof_H2_id]
    kout = get_Rate_PH2(net, H2, kappa_22, energy_H2_o, pop, eps, Vm)[1]     # H2 -> P-side
    kin = get_Rate_PH2(net, H2, kappa_22, energy_H2_o, pop, eps, Vm)[0]     # P-side -> H2
    #print('kout=', kout, 'kin=', kin, 'p_A=', p_A)
    flux = kout*p_H2 - kin*(1-p_H2)

    return flux  

def getElectronFlux_1(net, E3: Cofactor, kappa_33: float, energy_E3_o: float, pop: list, eps: np.array, Vm: float):
    cof_E3_id = net.cofactor2id[E3]
    p_E3 = pop[cof_E3_id]
    kout = get_Rate_CytE3(net, E3, kappa_33, energy_E3_o, pop, eps, Vm)[1]    # E3 -> CytC
    kin = get_Rate_CytE3(net, E3, kappa_33, energy_E3_o, pop, eps, Vm)[0]     # CytC -> E3
    flux = kin*(1-p_E3) - kout*p_E3

    return flux  

def getUpFlux_1(net, H1: Cofactor, kappa_11: float, energy_H1_o: float, pop: list, eps: np.array, Vm: float):
    cof_H1_id = net.cofactor2id[H1]
    p_H1 = pop[cof_H1_id]
    kout = get_Rate_NH1(net, H1, kappa_11, energy_H1_o, pop, eps, Vm)[1]    # H1 -> N-side
    kin = get_Rate_NH1(net, H1, kappa_11, energy_H1_o, pop, eps, Vm)[0]     # N-side -> H1
    flux = kin*(1-p_H1) - kout*p_H1

    return flux  

def getProductFlux_1(net, H1: Cofactor, E3: Cofactor, kp: float, pop: list):
    cof_H1_id = net.cofactor2id[H1]
    cof_E3_id = net.cofactor2id[E3]
    p_H1 = pop[cof_H1_id]
    p_E3 = pop[cof_E3_id]
    
    flux = kp * p_H1 * p_E3

    return flux
    

def getInitialVector(net, pop_MEK, H1: Cofactor, H2: Cofactor, E3: Cofactor):
    pop = np.zeros(net.num_cofactor)    # pop = [P_H1, P_H2, P_E3]
    #MEK solution of p_H1
    pop[0] = net.population(pop_MEK, H1, 1)
    #MEK solution of p_H2
    pop[1] = net.population(pop_MEK, H2, 1)
    #MEK solution of p_E3
    pop[2] = net.population(pop_MEK, E3, 1)
    return pop

def getProbabilityVector(net):
    num_unknowns = net.num_cofactor
    init_prob_list = []
    N = 0
    while N < num_unknowns:
        N += 1
        q = np.random.rand()
        #init_prob_list.append(0)
        init_prob_list.append(q)
    # print(init_prob_list)
    return init_prob_list

def getpop_1(net, RateEqns: list, pop_init):
    # Numerically solve set of non-linear equations
    # Core part of the mean-field model calculation!!
    pop = []    # pop = [P_H1, P_H2, P_E3]
    success = False
    while success == False:
        try:
            pop = fsolve(RateEqns, pop_init)
            for i in range(len(pop)):
                #if 0 < pop[i] < 1:
                if 0 < pop[0] < 1-10**(-3) and 0 < pop[1] < 1-10**(-3) and 0 < pop[2] < 1:
                #if 10**(-3) < pop[0] < 1-10**(-3) and 10**(-3) < pop[1] < 1-10**(-3) and 10**(-3) < pop[2] < 1-10**(-3):
                    success = True
                    # print('My initial guess succeeded!')
                else:
                    success = False
                    # print('The solutions given by your very first initial guess were unphysical. Try again...')

                    success2 = False
                    while success2 == False:
                        try:
                            probability_init = getProbabilityVector(net)       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
                            # print('Generated new intial guess:', probability_init)
                            pop = fsolve(RateEqns, probability_init)
                            # print('pop!', pop)
                            for i in range(len(pop)):
                                #if 0 < pop[i] < 1:
                                if 0 < pop[0] < 1 and 0 < pop[1] < 1 and 0 < pop[2] < 1:
                                #if 10**(-3) < pop[0] < 1-10**(-3) and 10**(-3) < pop[1] < 1-10**(-3) and 10**(-3) < pop[2] < 1-10**(-3):
                                    success2 = True    # Terminate the while success2 == False loop
                                    success = True     # Terminate the while success == False loop
                                    # print('Successfully found physical solutions!')
                                else:
                                    success2 = False
                                    # print('The solutions were unphysical again. Try another initial guess...')
                        except ValueError:
                            success2 = False
                            # print('Oops! Have to try another initial guess...')
        except ValueError:
            # print('Oops! Could not find root within given tolerance using given initial guess...')
            success3 = False
            while success3 == False:
                try:
                    probability_init = getProbabilityVector(net)       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
                    # print('Generated new intial guess:', probability_init)
                    pop = fsolve(RateEqns, probability_init)
                    for i in range(len(pop)):
                        #if 0 < pop[i] < 1:
                        if 0 < pop[0] < 1 and 0 < pop[1] < 1 and 0 < pop[2] < 1:
                        #if 10**(-3) < pop[0] < 1-10**(-3) and 10**(-3) < pop[1] < 1-10**(-3) and 10**(-3) < pop[2] < 1-10**(-3):
                            success3 = True    # Terminate the while success3 == False loop
                            success = True     # Terminate the while success == False loop
                            # print('Successfully found physical solutions!')
                except ValueError:
                    success3 = False
                    # print('Oops! Have to try another initial guess...')
    return pop


