import numpy as np
from sympy import *
from scipy.optimize import *

from MEK_protonpump import *

def get_Rate_H1H2(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, pop: np.array):
    #pop = [p_H1, p_H2, p_E3]
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    cof_E3_id = net.cofactor2id[E3]
    # Search for the initial and final microstate, associated with a proton transfer from H1 -> H2
    rate_H1H2 = 0
    rate_H2H1 = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_H1_id] == 1 and net.idx2state(i)[cof_H2_id] == 0:   #Initially, H1:1 H2:0
                if net.idx2state(f)[cof_H1_id] == 0 and net.idx2state(f)[cof_H2_id] == 1:   #Finally, H1:0 H2:1
                    if net.idx2state(i)[cof_E3_id] == net.idx2state(f)[cof_E3_id]:    #The occupation state of E3 is unchanged
                        kf = net.modK[f][i]
                        kb = net.modK[i][f]
                        initial.append(net.idx2state(i))
                        final.append(net.idx2state(f))
                        kfs.append(kf)
                        kbs.append(kb)
                        if net.idx2state(i)[cof_E3_id] == 0:
                            rate_H1H2 += kf*(1-pop[cof_E3_id])
                            rate_H2H1 += kb*(1-pop[cof_E3_id])
                        if net.idx2state(i)[cof_E3_id] == 1:
                            rate_H1H2 += kf*pop[cof_E3_id]
                            rate_H2H1 += kb*pop[cof_E3_id]
    # print('initial:', initial)
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_H1H2, rate_H2H1]
    
def get_Rate_H1N(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, pop: np.array):
    #pop = [p_H1, p_H2, p_E3]
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    cof_E3_id = net.cofactor2id[E3]
    # Search for the initial and final microstate, associated with a proton transfer from H1 -> N-side(reservoir)
    rate_H1N = 0
    rate_NH1 = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_H1_id] == 1 and net.idx2state(f)[cof_H1_id] == 0:   #Initially, H1:1  #Finally, H1:0 
                if net.idx2state(i)[cof_H2_id]==net.idx2state(f)[cof_H2_id] and net.idx2state(i)[cof_E3_id]==net.idx2state(f)[cof_E3_id]:    #The occupation state of H2 and E3 are unchanged
                    kf = net.modK[f][i]
                    kb = net.modK[i][f]
                    initial.append(net.idx2state(i))
                    final.append(net.idx2state(f))
                    kfs.append(kf)
                    kbs.append(kb)
                    if net.idx2state(i)[cof_H2_id]==0 and net.idx2state(i)[cof_E3_id]==0:
                        rate_H1N += kf*(1-pop[cof_H2_id])*(1-pop[cof_E3_id])
                        rate_NH1 += kb*(1-pop[cof_H2_id])*(1-pop[cof_E3_id])
                    if net.idx2state(i)[cof_H2_id]==1 and net.idx2state(i)[cof_E3_id]==0:
                        rate_H1N += kf*pop[cof_H2_id]*(1-pop[cof_E3_id])
                        rate_NH1 += kb*pop[cof_H2_id]*(1-pop[cof_E3_id])
                    if net.idx2state(i)[cof_H2_id]==0 and net.idx2state(i)[cof_E3_id]==1:
                        rate_H1N += kf*(1-pop[cof_H2_id])*pop[cof_E3_id]
                        rate_NH1 += kb*(1-pop[cof_H2_id])*pop[cof_E3_id]
                    if net.idx2state(i)[cof_H2_id]==1 and net.idx2state(i)[cof_E3_id]==1:
                        rate_H1N += kf*pop[cof_H2_id]*pop[cof_E3_id]
                        rate_NH1 += kb*pop[cof_H2_id]*pop[cof_E3_id]
    # print('initial:', initial)
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_H1N, rate_NH1]

def get_Rate_H2P(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, pop: np.array):
    #pop = [p_H1, p_H2, p_E3]
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    cof_E3_id = net.cofactor2id[E3]
    # Search for the initial and final microstate, associated with a proton transfer from H2 -> P-side(reservoir)
    rate_H2P = 0
    rate_PH2 = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_H2_id] == 1 and net.idx2state(f)[cof_H2_id] == 0:   #Initially, H2:1  #Finally, H2:0 
                if net.idx2state(i)[cof_H1_id]==net.idx2state(f)[cof_H1_id] and net.idx2state(i)[cof_E3_id]==net.idx2state(f)[cof_E3_id]:    #The occupation state of H1 and E3 are unchanged
                    kf = net.modK[f][i]
                    kb = net.modK[i][f]
                    initial.append(net.idx2state(i))
                    final.append(net.idx2state(f))
                    kfs.append(kf)
                    kbs.append(kb)
                    if net.idx2state(i)[cof_H1_id]==0 and net.idx2state(i)[cof_E3_id]==0:
                        rate_H2P += kf*(1-pop[cof_H1_id])*(1-pop[cof_E3_id])
                        rate_PH2 += kb*(1-pop[cof_H1_id])*(1-pop[cof_E3_id])
                    if net.idx2state(i)[cof_H1_id]==1 and net.idx2state(i)[cof_E3_id]==0:
                        rate_H2P += kf*pop[cof_H1_id]*(1-pop[cof_E3_id])
                        rate_PH2 += kb*pop[cof_H1_id]*(1-pop[cof_E3_id])
                    if net.idx2state(i)[cof_H1_id]==0 and net.idx2state(i)[cof_E3_id]==1:
                        rate_H2P += kf*(1-pop[cof_H1_id])*pop[cof_E3_id]
                        rate_PH2 += kb*(1-pop[cof_H1_id])*pop[cof_E3_id]
                    if net.idx2state(i)[cof_H1_id]==1 and net.idx2state(i)[cof_E3_id]==1:
                        rate_H2P += kf*pop[cof_H1_id]*pop[cof_E3_id]
                        rate_PH2 += kb*pop[cof_H1_id]*pop[cof_E3_id]
    # print('initial:', initial)
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_H2P, rate_PH2]

def get_Rate_E3Cyt(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, pop: np.array):
    #pop = [p_H1, p_H2, p_E3]
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    cof_E3_id = net.cofactor2id[E3]
    # Search for the initial and final microstate, associated with a proton transfer from E3 -> Cytochrome c (reservoir)
    rate_E3Cyt = 0
    rate_CytE3 = 0
    initial = []
    final = []
    kfs = []
    kbs = []
    for i in range(net.adj_num_state):
        for f in range(net.adj_num_state):
            if net.idx2state(i)[cof_E3_id] == 1 and net.idx2state(f)[cof_E3_id] == 0:   #Initially, E3:1  #Finally, E3:0 
                if net.idx2state(i)[cof_H1_id]==net.idx2state(f)[cof_H1_id] and net.idx2state(i)[cof_H2_id]==net.idx2state(f)[cof_H2_id]:    #The occupation state of H1 and H2 are unchanged
                    kf = net.modK[f][i]
                    kb = net.modK[i][f]
                    initial.append(net.idx2state(i))
                    final.append(net.idx2state(f))
                    kfs.append(kf)
                    kbs.append(kb)
                    if net.idx2state(i)[cof_H1_id]==0 and net.idx2state(i)[cof_H2_id]==0:
                        rate_E3Cyt += kf*(1-pop[cof_H1_id])*(1-pop[cof_H2_id])
                        rate_CytE3 += kb*(1-pop[cof_H1_id])*(1-pop[cof_H2_id])
                    if net.idx2state(i)[cof_H1_id]==1 and net.idx2state(i)[cof_H2_id]==0:
                        rate_E3Cyt += kf*pop[cof_H1_id]*(1-pop[cof_H2_id])
                        rate_CytE3 += kb*pop[cof_H1_id]*(1-pop[cof_H2_id])
                    if net.idx2state(i)[cof_H1_id]==0 and net.idx2state(i)[cof_H2_id]==1:
                        rate_E3Cyt += kf*(1-pop[cof_H1_id])*pop[cof_H2_id]
                        rate_CytE3 += kb*(1-pop[cof_H1_id])*pop[cof_H2_id]
                    if net.idx2state(i)[cof_H1_id]==1 and net.idx2state(i)[cof_H2_id]==1:
                        rate_E3Cyt += kf*pop[cof_H1_id]*pop[cof_H2_id]
                        rate_CytE3 += kb*pop[cof_H1_id]*pop[cof_H2_id]
    # print('initial:', initial)
    # print('final:', final)
    # print('forward:', kfs)
    # print('backward:', kbs)

    return [rate_E3Cyt, rate_CytE3]

def get_RateEquation_H1(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, kp: float, pop: np.array):
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    cof_E3_id = net.cofactor2id[E3]
    ### Rate constants ###
    k_H1H2 = get_Rate_H1H2(net, H1, H2, E3, pop)[0]  # H1 -> H2
    k_H2H1 = get_Rate_H1H2(net, H1, H2, E3, pop)[1]  # H2 -> H1
    k_H1N = get_Rate_H1N(net, H1, H2, E3, pop)[0]  # H1 -> N-side
    k_NH1 = get_Rate_H1N(net, H1, H2, E3, pop)[1]  # N-side -> H1
    #kp: rate of H2O production    # H1, E3 -> oxygen  # Assume irreversible
    ### Rate equation of H1: dH1/dt ###
    J_H1 = k_H2H1*pop[cof_H2_id]*(1-pop[cof_H1_id]) + k_NH1*(1-pop[cof_H1_id]) - k_H1H2*pop[cof_H1_id]*(1-pop[cof_H2_id]) - k_H1N*pop[cof_H1_id] - kp*pop[cof_H1_id]*pop[cof_E3_id]

    return J_H1

def get_RateEquation_H2(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, kp: float, pop: np.array):
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    cof_E3_id = net.cofactor2id[E3]
    ### Rate constants ###
    k_H1H2 = get_Rate_H1H2(net, H1, H2, E3, pop)[0]  # H1 -> H2
    k_H2H1 = get_Rate_H1H2(net, H1, H2, E3, pop)[1]  # H2 -> H1
    k_H2P = get_Rate_H2P(net, H1, H2, E3, pop)[0]  # H2 -> P-side
    k_PH2 = get_Rate_H2P(net, H1, H2, E3, pop)[1]  # P-side -> H2
    ### Rate equation of H2: dH2/dt ###
    J_H2 = k_H1H2*pop[cof_H1_id]*(1-pop[cof_H2_id]) + k_PH2*(1-pop[cof_H2_id]) - k_H2H1*pop[cof_H2_id]*(1-pop[cof_H1_id]) - k_H2P*pop[cof_H2_id]

    return J_H2

def get_RateEquation_E3(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, kp: float, pop: np.array):
    cof_H1_id = net.cofactor2id[H1]
    cof_H2_id = net.cofactor2id[H2]
    cof_E3_id = net.cofactor2id[E3]
    ### Rate constants ###
    k_E3Cyt = get_Rate_E3Cyt(net, H1, H2, E3, pop)[0]  # E3 -> CytC
    k_CytE3 = get_Rate_E3Cyt(net, H1, H2, E3, pop)[1]  # CytC -> E3
    #kp: rate of H2O production    # H1, E3 -> oxygen  # Assume irreversible
    ### Rate equation of E3: dE3/dt ###
    J_E3 = k_CytE3*(1-pop[cof_E3_id]) - k_E3Cyt*pop[cof_E3_id] - kp*pop[cof_H1_id]*pop[cof_E3_id]
    # J_E3 = k_CytE3*(1-p_E3) - k_E3Cyt*p_E3 - kp*p_H1*p_E3

    return J_E3

def getPumpFlux(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, kp: float, pop: np.array):
    cof_H2_id = net.cofactor2id[H2]
    p_H2 = pop[cof_H2_id]
    kout = get_Rate_H2P(net, H1, H2, E3, pop)[0]     # H2 -> P-side
    kin = get_Rate_H2P(net, H1, H2, E3, pop)[1]     # P-side -> H2
    #print('kout=', kout, 'kin=', kin, 'p_A=', p_A)
    flux = kout*p_H2 - kin*(1-p_H2)

    return flux  

def getElectronFlux(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, kp: float, pop: np.array):
    cof_E3_id = net.cofactor2id[E3]
    p_E3 = pop[cof_E3_id]
    kout = get_Rate_E3Cyt(net, H1, H2, E3, pop)[0]    # E3 -> CytC
    kin = get_Rate_E3Cyt(net, H1, H2, E3, pop)[1]     # CytC -> E3
    flux = kin*(1-p_E3) - kout*p_E3

    return flux  

def getUpFlux(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, kp: float, pop: np.array):
    cof_H1_id = net.cofactor2id[H1]
    p_H1 = pop[cof_H1_id]
    kout = get_Rate_H1N(net, H1, H2, E3, pop)[0]    # H1 -> N-side
    kin = get_Rate_H1N(net, H1, H2, E3, pop)[1]     # N-side -> H1
    flux = kin*(1-p_H1) - kout*p_H1

    return flux  

def getProductFlux(net, H1: Cofactor, H2: Cofactor, E3: Cofactor, kp: float, pop: np.array):
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
                if 0 < pop[0] < 1 and 0 < pop[1] < 1-10**(-3) and 0 < pop[2] < 1:
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

