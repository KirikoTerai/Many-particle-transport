import numpy as np
from sympy import *
from scipy.optimize import *

from MEK_linear import *

def getRedox_A(net, A: Cofactor, pop: np.array, eps: np.array):
    cof_A_id = net.cofactor2id[A]
    redox_A = A.redox[0]   # energy_H1_o is the free energy of site H1 when sites H2 and E3 are not occupied
    for cof_id in range(net.num_cofactor):
        if cof_id != cof_A_id:
            redox_A += (1-pop[cof_id])*eps[cof_A_id][cof_id]   # Contribution of electrostatic interactions to the site energy 

    return redox_A

def getRedox_B(net, B: Cofactor, pop: np.array, eps: np.array):
    cof_B_id = net.cofactor2id[B]
    redox_B = B.redox[0]   # energy_H1_o is the free energy of site H1 when sites H2 and E3 are not occupied
    for cof_id in range(net.num_cofactor):
        if cof_id != cof_B_id:
            redox_B += (1-pop[cof_id])*eps[cof_B_id][cof_id]   # Contribution of electrostatic interactions to the site energy 

    return redox_B

def getRedox_C(net, C: Cofactor, pop: np.array, eps: np.array):
    cof_C_id = net.cofactor2id[C]
    redox_C = C.redox[0]   # energy_H1_o is the free energy of site H1 when sites H2 and E3 are not occupied
    for cof_id in range(net.num_cofactor):
        if cof_id != cof_C_id:
            redox_C += (1-pop[cof_id])*eps[cof_C_id][cof_id]   # Contribution of electrostatic interactions to the site energy 

    return redox_C

def getRedox_D(net, D: Cofactor, pop: np.array, eps: np.array):
    cof_D_id = net.cofactor2id[D]
    redox_D = D.redox[0]   # energy_H1_o is the free energy of site H1 when sites H2 and E3 are not occupied
    for cof_id in range(net.num_cofactor):
        if cof_id != cof_D_id:
            redox_D += (1-pop[cof_id])*eps[cof_D_id][cof_id]   # Contribution of electrostatic interactions to the site energy 

    return redox_D

def get_Rate_AB(net, A: Cofactor, B: Cofactor, pop: np.array, eps: np.array, reorgE_dict: dict, coupling_dict: dict):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    G_A = -getRedox_A(net, A, pop, eps)
    G_B = -getRedox_B(net, B, pop, eps)

    deltaG = G_B - G_A
    reorgE = reorgE_dict['λ_AB']
    V = coupling_dict['V_AB']
    dis = net.D[cof_A_id][cof_B_id]

    kf = net.ET(deltaG, dis, reorgE, V)
    kb = kf * np.exp(net.beta*deltaG)
    
    k_AB = [kf, kb]
    # print('AB', k_AB)

    return k_AB

def get_Rate_BC(net, B: Cofactor, C: Cofactor, pop: np.array, eps: np.array, reorgE_dict: dict, coupling_dict: dict):
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    G_B = -getRedox_B(net, B, pop, eps)
    G_C = -getRedox_C(net, C, pop, eps)
    
    deltaG = G_C - G_B
    reorgE = reorgE_dict['λ_BC']
    V = coupling_dict['V_BC']
    dis = net.D[cof_B_id][cof_C_id]

    kf = net.ET(deltaG, dis, reorgE, V)
    kb = kf * np.exp(net.beta*deltaG)
    
    k_BC = [kf, kb]
    # print('BC', k_BC)

    return k_BC

def get_Rate_CD(net, C: Cofactor, D: Cofactor, pop: np.array, eps: np.array, reorgE_dict: dict, coupling_dict: dict):
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]
    G_C = -getRedox_C(net, C, pop, eps)
    G_D = -getRedox_D(net, D, pop, eps)
    
    deltaG = G_D - G_C
    reorgE = reorgE_dict['λ_CD']
    V = coupling_dict['V_CD']
    dis = net.D[cof_C_id][cof_D_id]

    kf = net.ET(deltaG, dis, reorgE, V)
    kb = kf * np.exp(net.beta*deltaG)
    
    k_CD = [kf, kb]
    # print('CD', k_CD)

    return k_CD

def get_Rate_Ares(net, A: Cofactor, pop: np.array, eps: np.array):
    reservoir_id = net.reservoir2id['Ares']
    name, cofactor, redox_state, num_electron, deltaG, rate = net.reservoirInfo[reservoir_id]
    kout = net.reservoirInfo[reservoir_id][5]
    #kin = net.getRate(kout, net.reservoirInfo[reservoir_id][4])
    kin = kout * np.exp(net.beta*net.reservoirInfo[reservoir_id][4])
    num_electron = net.reservoirInfo[reservoir_id][3]

    G_A = -getRedox_A(net, A, pop, eps)
    dA = G_A - (-A.redox[0])    # The difference between intrinsic free energy and free energy shifted due to electrostatic interactions
    deltaG_mod = deltaG - dA    # deltaG between A and Ares changes since the free energy of A shifts due to electrostaic interactions
    
    kf = rate    # forward rate is A -> Ares
    kb = kf * np.exp(net.beta*deltaG_mod)

    k_Ares = [kf, kb]
    # print('A -> Ares, Ares -> A:', k_Ares)

    return k_Ares

def get_Rate_Dres(net, D: Cofactor, pop: np.array, eps: np.array):
    reservoir_id = net.reservoir2id['Dres']
    name, cofactor, redox_state, num_electron, deltaG, rate = net.reservoirInfo[reservoir_id]
    kout = net.reservoirInfo[reservoir_id][5]
    #kin = net.getRate(kout, net.reservoirInfo[reservoir_id][4])
    kin = kout * np.exp(net.beta*net.reservoirInfo[reservoir_id][4])
    num_electron = net.reservoirInfo[reservoir_id][3]

    G_D = -getRedox_D(net, D, pop, eps)
    dD = G_D - (-D.redox[0])    # The difference between intrinsic free energy and free energy shifted due to electrostatic interactions
    deltaG_mod = deltaG - dD    # deltaG between A and Ares changes since the free energy of A shifts due to electrostaic interactions
    
    kf = rate    # forward rate is D -> Dres 
    kb = kf * np.exp(net.beta*deltaG_mod)

    k_Dres = [kf, kb]
    # print('D -> Dres, Dres -> D:', k_Dres)

    return k_Dres

def get_RateEquation_A_1(net, A: Cofactor, B: Cofactor, pop: np.array, eps: np.array, reorgE_dict: dict, coupling_dict: dict):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    ### Rate constants ###
    kf_AB = get_Rate_AB(net, A, B, pop, eps, reorgE_dict, coupling_dict)[0]  # A -> B
    kb_AB = get_Rate_AB(net, A, B, pop, eps, reorgE_dict, coupling_dict)[1]  # B -> A
    kf_Ares = get_Rate_Ares(net, A, pop, eps)[0]  # A -> Ares
    kb_Ares = get_Rate_Ares(net, A, pop, eps)[1]  # Ares -> A
    ### Rate equation of A: dA/dt ###
    J_A = kb_AB*pop[cof_B_id]*(1-pop[cof_A_id]) + kb_Ares*(1-pop[cof_A_id]) - kf_AB*pop[cof_A_id]*(1-pop[cof_B_id]) - kf_Ares*pop[cof_A_id]

    return J_A

def get_RateEquation_B_1(net, A: Cofactor, B: Cofactor, C: Cofactor, pop: np.array, eps: np.array, reorgE_dict: dict, coupling_dict: dict):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    ### Rate constants ###
    kf_AB = get_Rate_AB(net, A, B, pop, eps, reorgE_dict, coupling_dict)[0]  # A -> B
    kb_AB = get_Rate_AB(net, A, B, pop, eps, reorgE_dict, coupling_dict)[1]  # B -> A
    kf_BC = get_Rate_BC(net, B, C, pop, eps, reorgE_dict, coupling_dict)[0]  # B -> C
    kb_BC = get_Rate_BC(net, B, C, pop, eps, reorgE_dict, coupling_dict)[1]  # C -> B
    ### Rate equation of B: dB/dt ###
    J_B = kf_AB*pop[cof_A_id]*(1-pop[cof_B_id]) + kb_BC*pop[cof_C_id]*(1-pop[cof_B_id]) - kb_AB*pop[cof_B_id]*(1-pop[cof_A_id]) - kf_BC*pop[cof_B_id]*(1-pop[cof_C_id])

    return J_B

def get_RateEquation_C_1(net, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array, eps: np.array, reorgE_dict: dict, coupling_dict: dict):
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]
    ### Rate constants ###
    kf_BC = get_Rate_BC(net, B, C, pop, eps, reorgE_dict, coupling_dict)[0]  # B -> C
    kb_BC = get_Rate_BC(net, B, C, pop, eps, reorgE_dict, coupling_dict)[1]  # C -> B
    kf_CD = get_Rate_CD(net, C, D, pop, eps, reorgE_dict, coupling_dict)[0]  # C -> D
    kb_CD = get_Rate_CD(net, C, D, pop, eps, reorgE_dict, coupling_dict)[1]  # D -> C
    ### Rate equation of C: dC/dt ###
    J_C = kf_BC*pop[cof_B_id]*(1-pop[cof_C_id]) + kb_CD*pop[cof_D_id]*(1-pop[cof_C_id]) - kb_BC*pop[cof_C_id]*(1-pop[cof_B_id]) - kf_CD*pop[cof_C_id]*(1-pop[cof_D_id])

    return J_C

def get_RateEquation_D_1(net, C: Cofactor, D: Cofactor, pop: np.array, eps: np.array, reorgE_dict: dict, coupling_dict: dict):
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]
    ### Rate constants ###
    kf_CD = get_Rate_CD(net, C, D, pop, eps, reorgE_dict, coupling_dict)[0]  # C -> D
    kb_CD = get_Rate_CD(net, C, D, pop, eps, reorgE_dict, coupling_dict)[1]  # D -> C
    kf_Dres = get_Rate_Dres(net, D, pop, eps)[0]  # D -> Dres
    kb_Dres = get_Rate_Dres(net, D, pop, eps)[1]  # Dres -> D
    ### Rate equation of D: dD/dt ###
    J_D = kf_CD*pop[cof_C_id]*(1-pop[cof_D_id]) + kb_Dres*(1-pop[cof_D_id]) - kb_CD*pop[cof_D_id]*(1-pop[cof_C_id]) - kf_Dres*pop[cof_D_id]

    return J_D

def getProbabilityVector(net):
    """
    Initialize dictionary of probabilities that has to be solved for
    i.e. In the EB case, [p_ox, p_sq, p_L1, p_H1, p_L2,　p_H2]
    """
    num_unknowns = net.num_cofactor
    init_prob_list = []
    N = 0
    while N < num_unknowns:
        N += 1
        q = np.random.rand()
        init_prob_list.append(q)
    return init_prob_list

def getReservoirFlux_1(net, name: str, pop: list, eps: np.array):
    reservoir_id = net.reservoir2id[name]
    name, cofactor, redox_state, num_electron, deltaG, rate = net.reservoirInfo[reservoir_id]
    kout = net.reservoirInfo[reservoir_id][5]
    #kin = net.getRate(kout, net.reservoirInfo[reservoir_id][4])
    kin = kout * np.exp(net.beta*net.reservoirInfo[reservoir_id][4])
    num_electron = net.reservoirInfo[reservoir_id][3]
    
    flux = 0
    if name == 'Ares':
        p_A = pop[net.cofactor2id[cofactor]]
        kout = get_Rate_Ares(net, cofactor, pop, eps)[0]
        kin = get_Rate_Ares(net, cofactor, pop, eps)[1]
        flux = kout*p_A - kin*(1-p_A)
    if name == 'Dres':
        p_D = pop[net.cofactor2id[cofactor]]
        kout = get_Rate_Dres(net, cofactor, pop, eps)[0]
        kin = get_Rate_Dres(net, cofactor, pop, eps)[1]
        flux = kout*p_D - kin*(1-p_D)
        #flux = kout*p_D

    return flux

def getInitialVector(net, pop_MEK, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor):
    pop = np.zeros(net.num_cofactor)    # pop = [P_A, P_B, P_C, P_D]
    #MEK solution of p_A
    pop[0] = net.population(pop_MEK, A, 1)
    #MEK solution of p_B
    pop[1] = net.population(pop_MEK, B, 1)
    #MEK solution of p_C
    pop[2] = net.population(pop_MEK, C, 1)
    #MEK solution of p_D
    pop[3] = net.population(pop_MEK, D, 1)

    return pop

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
                if 0 < pop[0] < 1 and 0 < pop[1] < 1 and 0 < pop[2] < 1 and 0 < pop[3] < 1:
                    success = True
                    #print('My initial guess succeeded!')
                else:
                    success = False
                    #print('The solutions given by your very first initial guess were unphysical. Try again...')

                    success2 = False
                    while success2 == False:
                        try:
                            probability_init = getProbabilityVector(net)       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
                            #print('Generated new intial guess:', probability_init)
                            pop = fsolve(RateEqns, probability_init)
                            for i in range(len(pop)):
                                #if 0 < pop[i] < 1:
                                if 0 < pop[0] < 1 and 0 < pop[1] < 1 and 0 < pop[2] < 1 and 0 < pop[3] < 1:
                                    success2 = True    # Terminate the while success2 == False loop
                                    success = True     # Terminate the while success == False loop
                                    #print('Successfully found physical solutions!')
                                else:
                                    success2 = False
                                    #print('The solutions were unphysical again. Try another initial guess...')
                        except ValueError:
                            success2 = False
                            #print('Oops! Have to try another initial guess...')
        except ValueError:
            #print('Oops! Could not find root within given tolerance using given initial guess...')
            success3 = False
            while success3 == False:
                try:
                    probability_init = getProbabilityVector(net)       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
                    #print('Generated new intial guess:', probability_init)
                    pop = fsolve(RateEqns, probability_init)
                    for i in range(len(pop)):
                        #if 0 < pop[i] < 1:
                        if 0 < pop[0] < 1 and 0 < pop[1] < 1 and 0 < pop[2] < 1 and 0 < pop[3] < 1:
                            success3 = True    # Terminate the while success3 == False loop
                            success = True     # Terminate the while success == False loop
                            #print('Successfully found physical solutions!')
                except ValueError:
                    success3 = False
                    #print('Oops! Have to try another initial guess...')

    return pop


