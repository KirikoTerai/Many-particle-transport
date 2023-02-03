import numpy as np
from sympy import *
from scipy.optimize import *

from MEK_linear import *

def get_Rate_AB(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]

    # Search for the initial and final microstate, associated with a proton transfer from H1 -> H2
    rate_AB = 0
    rate_BA = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_A_id] == 1 and net.idx2state(i)[cof_B_id] == 0:   #Initially, A:1 B:0
                if net.idx2state(f)[cof_A_id] == 0 and net.idx2state(f)[cof_B_id] == 1:   #Finally, A:0 B:1
                    if net.idx2state(i)[cof_C_id] == net.idx2state(f)[cof_C_id] and net.idx2state(i)[cof_D_id] == net.idx2state(f)[cof_D_id]:    #The occupation state of C and D is unchanged
                        kf = net.modK[f][i]
                        kb = net.modK[i][f]
                        initial.append(net.idx2state(i))
                        final.append(net.idx2state(f))
                        kfs.append(kf)
                        kbs.append(kb)
                        if net.idx2state(i)[cof_C_id] == 0 and net.idx2state(i)[cof_D_id] == 0:
                            rate_AB += kf*(1-pop[cof_C_id])*(1-pop[cof_D_id])
                            rate_BA += kb*(1-pop[cof_C_id])*(1-pop[cof_D_id])
                        if net.idx2state(i)[cof_C_id] == 1 and net.idx2state(i)[cof_D_id] == 0:
                            rate_AB += kf*(pop[cof_C_id])*(1-pop[cof_D_id])
                            rate_BA += kb*(pop[cof_C_id])*(1-pop[cof_D_id])
                        if net.idx2state(i)[cof_C_id] == 0 and net.idx2state(i)[cof_D_id] == 1:
                            rate_AB += kf*(1-pop[cof_C_id])*(pop[cof_D_id])
                            rate_BA += kb*(1-pop[cof_C_id])*(pop[cof_D_id])
                        if net.idx2state(i)[cof_C_id] == 1 and net.idx2state(i)[cof_D_id] == 1:
                            rate_AB += kf*(pop[cof_C_id])*(pop[cof_D_id])
                            rate_BA += kb*(pop[cof_C_id])*(pop[cof_D_id])
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_AB, rate_BA]   

def get_Rate_BC(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]

    # Search for the initial and final microstate, associated with a proton transfer from H1 -> H2
    rate_BC = 0
    rate_CB = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_B_id] == 1 and net.idx2state(i)[cof_C_id] == 0:   #Initially, B:1 C:0
                if net.idx2state(f)[cof_B_id] == 0 and net.idx2state(f)[cof_C_id] == 1:   #Finally, B:0 C:1
                    if net.idx2state(i)[cof_A_id] == net.idx2state(f)[cof_A_id] and net.idx2state(i)[cof_D_id] == net.idx2state(f)[cof_D_id]:    #The occupation state of A and D is unchanged
                        kf = net.modK[f][i]
                        kb = net.modK[i][f]
                        initial.append(net.idx2state(i))
                        final.append(net.idx2state(f))
                        kfs.append(kf)
                        kbs.append(kb)
                        if net.idx2state(i)[cof_A_id] == 0 and net.idx2state(i)[cof_D_id] == 0:
                            rate_BC += kf*(1-pop[cof_A_id])*(1-pop[cof_D_id])
                            rate_CB += kb*(1-pop[cof_A_id])*(1-pop[cof_D_id])
                        if net.idx2state(i)[cof_A_id] == 1 and net.idx2state(i)[cof_D_id] == 0:
                            rate_BC += kf*(pop[cof_A_id])*(1-pop[cof_D_id])
                            rate_CB += kb*(pop[cof_A_id])*(1-pop[cof_D_id])
                        if net.idx2state(i)[cof_A_id] == 0 and net.idx2state(i)[cof_D_id] == 1:
                            rate_BC += kf*(1-pop[cof_A_id])*(pop[cof_D_id])
                            rate_CB += kb*(1-pop[cof_A_id])*(pop[cof_D_id])
                        if net.idx2state(i)[cof_A_id] == 1 and net.idx2state(i)[cof_D_id] == 1:
                            rate_BC += kf*(pop[cof_A_id])*(pop[cof_D_id])
                            rate_CB += kb*(pop[cof_A_id])*(pop[cof_D_id])
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_BC, rate_CB]  

def get_Rate_CD(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]

    # Search for the initial and final microstate, associated with a proton transfer from H1 -> H2
    rate_CD = 0
    rate_DC = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_C_id] == 1 and net.idx2state(i)[cof_D_id] == 0:   #Initially, C:1 D:0
                if net.idx2state(f)[cof_C_id] == 0 and net.idx2state(f)[cof_D_id] == 1:   #Finally, C:0 D:1
                    if net.idx2state(i)[cof_A_id] == net.idx2state(f)[cof_A_id] and net.idx2state(i)[cof_B_id] == net.idx2state(f)[cof_B_id]:    #The occupation state of A and B is unchanged
                        kf = net.modK[f][i]
                        kb = net.modK[i][f]
                        initial.append(net.idx2state(i))
                        final.append(net.idx2state(f))
                        kfs.append(kf)
                        kbs.append(kb)
                        if net.idx2state(i)[cof_A_id] == 0 and net.idx2state(i)[cof_B_id] == 0:
                            rate_CD += kf*(1-pop[cof_A_id])*(1-pop[cof_B_id])
                            rate_DC += kb*(1-pop[cof_A_id])*(1-pop[cof_B_id])
                        if net.idx2state(i)[cof_A_id] == 1 and net.idx2state(i)[cof_B_id] == 0:
                            rate_CD += kf*(pop[cof_A_id])*(1-pop[cof_B_id])
                            rate_DC += kb*(pop[cof_A_id])*(1-pop[cof_B_id])
                        if net.idx2state(i)[cof_A_id] == 0 and net.idx2state(i)[cof_B_id] == 1:
                            rate_CD += kf*(1-pop[cof_A_id])*(pop[cof_B_id])
                            rate_DC += kb*(1-pop[cof_A_id])*(pop[cof_B_id])
                        if net.idx2state(i)[cof_A_id] == 1 and net.idx2state(i)[cof_B_id] == 1:
                            rate_CD += kf*(pop[cof_A_id])*(pop[cof_B_id])
                            rate_DC += kb*(pop[cof_A_id])*(pop[cof_B_id])
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_CD, rate_DC]  

def get_Rate_Ares(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]

    # Search for the initial and final microstate, associated with a proton transfer from H1 -> N-side(reservoir)
    rate_Ares = 0
    rate_resA = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_A_id] == 1 and net.idx2state(f)[cof_A_id] == 0:   #Initially, A:1  #Finally, A:0 
                if net.idx2state(i)[cof_B_id]==net.idx2state(f)[cof_B_id] and net.idx2state(i)[cof_C_id]==net.idx2state(f)[cof_C_id] and net.idx2state(i)[cof_D_id]==net.idx2state(f)[cof_D_id]:    #The occupation state of H2 and E3 are unchanged
                    kf = net.modK[f][i]
                    kb = net.modK[i][f]
                    initial.append(net.idx2state(i))
                    final.append(net.idx2state(f))
                    kfs.append(kf)
                    kbs.append(kb)
                    if net.idx2state(i)[cof_B_id]==0 and net.idx2state(i)[cof_C_id]==0 and net.idx2state(i)[cof_D_id]==0:
                        rate_Ares += kf*(1-pop[cof_B_id])*(1-pop[cof_C_id])*(1-pop[cof_D_id])
                        rate_resA += kb*(1-pop[cof_B_id])*(1-pop[cof_C_id])*(1-pop[cof_D_id])
                    if net.idx2state(i)[cof_B_id]==1 and net.idx2state(i)[cof_C_id]==0 and net.idx2state(i)[cof_D_id]==0:
                        rate_Ares += kf*(pop[cof_B_id])*(1-pop[cof_C_id])*(1-pop[cof_D_id])
                        rate_resA += kb*(pop[cof_B_id])*(1-pop[cof_C_id])*(1-pop[cof_D_id])
                    if net.idx2state(i)[cof_B_id]==0 and net.idx2state(i)[cof_C_id]==1 and net.idx2state(i)[cof_D_id]==0:
                        rate_Ares += kf*(1-pop[cof_B_id])*(pop[cof_C_id])*(1-pop[cof_D_id])
                        rate_resA += kb*(1-pop[cof_B_id])*(pop[cof_C_id])*(1-pop[cof_D_id])
                    if net.idx2state(i)[cof_B_id]==0 and net.idx2state(i)[cof_C_id]==0 and net.idx2state(i)[cof_D_id]==1:
                        rate_Ares += kf*(1-pop[cof_B_id])*(1-pop[cof_C_id])*(pop[cof_D_id])
                        rate_resA += kb*(1-pop[cof_B_id])*(1-pop[cof_C_id])*(pop[cof_D_id])
                    if net.idx2state(i)[cof_B_id]==1 and net.idx2state(i)[cof_C_id]==1 and net.idx2state(i)[cof_D_id]==0:
                        rate_Ares += kf*(pop[cof_B_id])*(pop[cof_C_id])*(1-pop[cof_D_id])
                        rate_resA += kb*(pop[cof_B_id])*(pop[cof_C_id])*(1-pop[cof_D_id])
                    if net.idx2state(i)[cof_B_id]==1 and net.idx2state(i)[cof_C_id]==0 and net.idx2state(i)[cof_D_id]==1:
                        rate_Ares += kf*(pop[cof_B_id])*(1-pop[cof_C_id])*(pop[cof_D_id])
                        rate_resA += kb*(pop[cof_B_id])*(1-pop[cof_C_id])*(pop[cof_D_id])
                    if net.idx2state(i)[cof_B_id]==0 and net.idx2state(i)[cof_C_id]==1 and net.idx2state(i)[cof_D_id]==1:
                        rate_Ares += kf*(1-pop[cof_B_id])*(pop[cof_C_id])*(pop[cof_D_id])
                        rate_resA += kb*(1-pop[cof_B_id])*(pop[cof_C_id])*(pop[cof_D_id])
                    if net.idx2state(i)[cof_B_id]==1 and net.idx2state(i)[cof_C_id]==1 and net.idx2state(i)[cof_D_id]==1:
                        rate_Ares += kf*(pop[cof_B_id])*(pop[cof_C_id])*(pop[cof_D_id])
                        rate_resA += kb*(pop[cof_B_id])*(pop[cof_C_id])*(pop[cof_D_id])
    # print('initial:', initial)
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_Ares, rate_resA]

def get_Rate_Dres(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]

    # Search for the initial and final microstate, associated with a proton transfer from H1 -> N-side(reservoir)
    rate_Dres = 0
    rate_resD = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_D_id] == 1 and net.idx2state(f)[cof_D_id] == 0:   #Initially, D:1  #Finally, D:0 
                if net.idx2state(i)[cof_A_id]==net.idx2state(f)[cof_A_id] and net.idx2state(i)[cof_B_id]==net.idx2state(f)[cof_B_id] and net.idx2state(i)[cof_C_id]==net.idx2state(f)[cof_C_id]:    #The occupation state of H2 and E3 are unchanged
                    kf = net.modK[f][i]
                    kb = net.modK[i][f]
                    initial.append(net.idx2state(i))
                    final.append(net.idx2state(f))
                    kfs.append(kf)
                    kbs.append(kb)
                    if net.idx2state(i)[cof_B_id]==0 and net.idx2state(i)[cof_C_id]==0 and net.idx2state(i)[cof_A_id]==0:
                        rate_Dres += kf*(1-pop[cof_B_id])*(1-pop[cof_C_id])*(1-pop[cof_A_id])
                        rate_resD += kb*(1-pop[cof_B_id])*(1-pop[cof_C_id])*(1-pop[cof_A_id])
                    if net.idx2state(i)[cof_B_id]==1 and net.idx2state(i)[cof_C_id]==0 and net.idx2state(i)[cof_A_id]==0:
                        rate_Dres += kf*(pop[cof_B_id])*(1-pop[cof_C_id])*(1-pop[cof_A_id])
                        rate_resD += kb*(pop[cof_B_id])*(1-pop[cof_C_id])*(1-pop[cof_A_id])
                    if net.idx2state(i)[cof_B_id]==0 and net.idx2state(i)[cof_C_id]==1 and net.idx2state(i)[cof_A_id]==0:
                        rate_Dres += kf*(1-pop[cof_B_id])*(pop[cof_C_id])*(1-pop[cof_A_id])
                        rate_resD += kb*(1-pop[cof_B_id])*(pop[cof_C_id])*(1-pop[cof_A_id])
                    if net.idx2state(i)[cof_B_id]==0 and net.idx2state(i)[cof_C_id]==0 and net.idx2state(i)[cof_A_id]==1:
                        rate_Dres += kf*(1-pop[cof_B_id])*(1-pop[cof_C_id])*(pop[cof_A_id])
                        rate_resD += kb*(1-pop[cof_B_id])*(1-pop[cof_C_id])*(pop[cof_A_id])
                    if net.idx2state(i)[cof_B_id]==1 and net.idx2state(i)[cof_C_id]==1 and net.idx2state(i)[cof_A_id]==0:
                        rate_Dres += kf*(pop[cof_B_id])*(pop[cof_C_id])*(1-pop[cof_A_id])
                        rate_resD += kb*(pop[cof_B_id])*(pop[cof_C_id])*(1-pop[cof_A_id])
                    if net.idx2state(i)[cof_B_id]==1 and net.idx2state(i)[cof_C_id]==0 and net.idx2state(i)[cof_A_id]==1:
                        rate_Dres += kf*(pop[cof_B_id])*(1-pop[cof_C_id])*(pop[cof_A_id])
                        rate_resD += kb*(pop[cof_B_id])*(1-pop[cof_C_id])*(pop[cof_A_id])
                    if net.idx2state(i)[cof_B_id]==0 and net.idx2state(i)[cof_C_id]==1 and net.idx2state(i)[cof_A_id]==1:
                        rate_Dres += kf*(1-pop[cof_B_id])*(pop[cof_C_id])*(pop[cof_A_id])
                        rate_resD += kb*(1-pop[cof_B_id])*(pop[cof_C_id])*(pop[cof_A_id])
                    if net.idx2state(i)[cof_B_id]==1 and net.idx2state(i)[cof_C_id]==1 and net.idx2state(i)[cof_A_id]==1:
                        rate_Dres += kf*(pop[cof_B_id])*(pop[cof_C_id])*(pop[cof_A_id])
                        rate_resD += kb*(pop[cof_B_id])*(pop[cof_C_id])*(pop[cof_A_id])
    # print('initial:', initial)
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_Dres, rate_resD]

def get_RateEquation_A(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    ### Rate constants ###
    kf_AB = get_Rate_AB(net, A, B, C, D, pop)[0]  # A -> B
    kb_AB = get_Rate_AB(net, A, B, C, D, pop)[1]  # B -> A
    kf_Ares = get_Rate_Ares(net, A, B, C, D, pop)[0]  # A -> Ares
    kb_Ares = get_Rate_Ares(net, A, B, C, D, pop)[1]  # Ares -> A
    ### Rate equation of A: dA/dt ###
    J_A = kb_AB*pop[cof_B_id]*(1-pop[cof_A_id]) + kb_Ares*(1-pop[cof_A_id]) - kf_AB*pop[cof_A_id]*(1-pop[cof_B_id]) - kf_Ares*pop[cof_A_id]

    return J_A

def get_RateEquation_B(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_A_id = net.cofactor2id[A]
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    ### Rate constants ###
    kf_AB = get_Rate_AB(net, A, B, C, D, pop)[0]  # A -> B
    kb_AB = get_Rate_AB(net, A, B, C, D, pop)[1]  # B -> A
    kf_BC = get_Rate_BC(net, A, B, C, D, pop)[0]  # B -> C
    kb_BC = get_Rate_BC(net, A, B, C, D, pop)[1]  # C -> B
    ### Rate equation of B: dB/dt ###
    J_B = kf_AB*pop[cof_A_id]*(1-pop[cof_B_id]) + kb_BC*pop[cof_C_id]*(1-pop[cof_B_id]) - kb_AB*pop[cof_B_id]*(1-pop[cof_A_id]) - kf_BC*pop[cof_B_id]*(1-pop[cof_C_id])

    return J_B

def get_RateEquation_C(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_B_id = net.cofactor2id[B]
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]
    ### Rate constants ###
    kf_BC = get_Rate_BC(net, A, B, C, D, pop)[0]  # B -> C
    kb_BC = get_Rate_BC(net, A, B, C, D, pop)[1]  # C -> B
    kf_CD = get_Rate_CD(net, A, B, C, D, pop)[0]  # C -> D
    kb_CD = get_Rate_CD(net, A, B, C, D, pop)[1]  # D -> C
    ### Rate equation of C: dC/dt ###
    J_C = kf_BC*pop[cof_B_id]*(1-pop[cof_C_id]) + kb_CD*pop[cof_D_id]*(1-pop[cof_C_id]) - kb_BC*pop[cof_C_id]*(1-pop[cof_B_id]) - kf_CD*pop[cof_C_id]*(1-pop[cof_D_id])

    return J_C

def get_RateEquation_D(net, A: Cofactor, B: Cofactor, C: Cofactor, D: Cofactor, pop: np.array):
    cof_C_id = net.cofactor2id[C]
    cof_D_id = net.cofactor2id[D]
    ### Rate constants ###
    kf_CD = get_Rate_CD(net, A, B, C, D, pop)[0]  # C -> D
    kb_CD = get_Rate_CD(net, A, B, C, D, pop)[1]  # D -> C
    kf_Dres = get_Rate_Dres(net, A, B, C, D, pop)[0]  # D -> Dres
    kb_Dres = get_Rate_Dres(net, A, B, C, D, pop)[1]  # Dres -> D
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

def getReservoirFlux(net, name: str, pop: list, eps: np.array):
    reservoir_id = net.reservoir2id[name]
    name, cofactor, redox_state, num_electron, deltaG, rate = net.reservoirInfo[reservoir_id]
    kout = net.reservoirInfo[reservoir_id][5]
    #kin = net.getRate(kout, net.reservoirInfo[reservoir_id][4])
    kin = kout * np.exp(net.beta*net.reservoirInfo[reservoir_id][4])
    num_electron = net.reservoirInfo[reservoir_id][3]

    A = net.id2cofactor[0]
    B = net.id2cofactor[1]
    C = net.id2cofactor[2]
    D = net.id2cofactor[3]
    
    flux = 0
    if name == 'Ares':
        p_A = pop[net.cofactor2id[cofactor]]
        kout = get_Rate_Ares(net, A, B, C, D, pop)[0]
        kin = get_Rate_Ares(net, A, B, C, D, pop)[1]
        flux = kout*p_A - kin*(1-p_A)
    if name == 'Dres':
        p_D = pop[net.cofactor2id[cofactor]]
        kout = get_Rate_Dres(net, A, B, C, D, pop)[0]
        kin = get_Rate_Dres(net, A, B, C, D, pop)[1]
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

def getpop(net, RateEqns: list, pop_init):
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
