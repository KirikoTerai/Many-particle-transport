from MEK_public import *

import numpy as np
from scipy import linalg
from scipy.optimize import fsolve
import math
import sympy as sy
from sympy import *

#packages for plotting
import matplotlib
import matplotlib.pyplot as plt

net = Network()

def getHoppingRates(net, cof_i: Cofactor, cof_f: Cofactor, cof_i_initial_redox: int):
    """
    Extract the correct forward (cof_i -> cof_f) rate constant from the rate matrix constructed in MEK.
    {Input}
    cof_i: initial cofactor (cof_i, donor)
    cof_f: final cofactor (cof_f, acceptor)
    cof_i_initial_redox: redox state of cof_i before electron transfer
    {Output}
    ks: list [kf, kb]
    """
    ks = [0, 0]
    cof_i_id = net.cofactor2id[cof_i]
    cof_f_id = net.cofactor2id[cof_f]
    for state_i_id in range(net.adj_num_state):
        for state_f_id in range(net.adj_num_state):
            if state_i_id != state_f_id:
                if net.K[state_f_id][state_i_id] != 0:  # There is transition from microstate state_i_id -> state_f_id
                # Is cof_i and cof_j involved in ET ?
                # cof_i: donor, cof_f: acceptor
                    # Case 1: Redox state of cof_i before ET is 1
                    if cof_i_initial_redox == 1:
                        if net.idx2state(net.allow[state_i_id])[cof_i_id] == 1 and net.idx2state(net.allow[state_i_id])[cof_f_id] == 0:  # state_i_id microstate: [..,1,0,...]
                            if net.idx2state(net.allow[state_f_id])[cof_i_id] == 0 and net.idx2state(net.allow[state_f_id])[cof_f_id] == 1:   # state_f_id microstate [...,0,1,...]
                                cof_f_initial_redox = 0
                                # cof_i and cof_f found!
                                # Check conservation of cofactors not involved in ET
                                I = np.delete(net.idx2state(net.allow[state_i_id]), [cof_i_id, cof_f_id])
                                J = np.delete(net.idx2state(net.allow[state_f_id]), [cof_i_id, cof_f_id])
                                if np.array_equal(I, J):
                                    # Forward ET rate
                                    kf = net.K[state_f_id][state_i_id]
                                    # Backward ET rate
                                    deltaG = cof_i.redox[cof_i_initial_redox-1] - cof_f.redox[cof_f_initial_redox]
                                    kb = kf * np.exp(net.beta*deltaG)
                                    ks[0] = kf
                                    ks[1] = kb
                    # Case 2: Redox state of cof_i before ET is 2
                    if cof_i_initial_redox == 2:
                        if net.idx2state(net.allow[state_i_id])[cof_i_id] == 2 and net.idx2state(net.allow[state_i_id])[cof_f_id] == 0:  # state_i_id microstate: [..,2,0,...]
                            if net.idx2state(net.allow[state_f_id])[cof_i_id] == 1 and net.idx2state(net.allow[state_f_id])[cof_f_id] == 1:   # state_f_id microstate [...,1,1,...]
                                cof_f_initial_redox = 0
                                # cof_i and cof_f found!
                                # Check conservation of cofactors not involved in ET
                                I = np.delete(net.idx2state(net.allow[state_i_id]), [cof_i_id, cof_f_id])
                                J = np.delete(net.idx2state(net.allow[state_f_id]), [cof_i_id, cof_f_id])
                                if np.array_equal(I, J):
                                    # Forward ET rate
                                    kf = net.K[state_f_id][state_i_id]
                                    # Backward ET rate
                                    deltaG = cof_i.redox[cof_i_initial_redox-1] - cof_f.redox[cof_f_initial_redox]
                                    kb = kf * np.exp(net.beta*deltaG)
                                    ks[0] = kf
                                    ks[1] = kb

    return ks

def getInitialProbabilityVector(net):
    """
    Initialize dictionary of probabilities that has to be solved for
    i.e. In the EB case, [p_ox, p_sq, p_L1, p_H1, p_L2,　p_H2]
    """
    num_unknowns = 6
    init_prob_list = []
    N = 0
    while N < num_unknowns:
        N += 1
        q = np.random.rand()
        #init_prob_list.append(0)
        init_prob_list.append(q)
    # print(init_prob_list)
    return init_prob_list

    #     cof = net.id2cofactor[cof_id]
    #     redox = []
    #     redox_state = 0
    #     while redox_state < cof.capacity + 1:
    #         redox_state +=1
    #         redox.append(0)     #### Need to add unknown variables! Not numerical values!! ####
    #         probability[cof] =  redox
    # return probability

def getRateEquations_L1(net, donor: Cofactor, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2):   # Obtain rate eqn expression for dP_L1-/dt
    # Search for cofactors that have physical connections with each other
    cof_f_list = []
    cof_f_name = []

    cof_i_id = net.cofactor2id[donor]

    for cof_f_id in range(net.num_cofactor):
        if net.D[cof_i_id][cof_f_id] != 0:
            cof_f = net.id2cofactor[cof_f_id]
            cof_f_list.append(cof_f)
            cof_f_name.append(cof_f.name)
            J1 = 0
            for acceptor in cof_f_list:
                if acceptor.name == "L2":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # L1 -> L2
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # L2 -> L1
                    J1 += kb*(1 - p_L2)*p_L1 - kf*(1 - p_L1)*p_L2
                if acceptor.name == "H1":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # L1 -> H1
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # H1 -> L1
                    J1 += kb*(1 - p_H1)*p_L1 - kf*(1 - p_L1)*p_H1
                if acceptor.name == "H2":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # L1 -> H2
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # H2 -> L1
                    J1 += kb*(1 - p_H2)*p_L1 - kf*(1 - p_L1)*p_H2
                if acceptor.name == "D":
                    kf_2 = getHoppingRates(net, acceptor, donor, 2)[1]   # L1 -> D-
                    kb_2 = getHoppingRates(net, acceptor, donor, 2)[0]   # D= -> L1
                    kf_1 = getHoppingRates(net, donor, acceptor, 1)[0]   # L1 -> D
                    kb_1 = getHoppingRates(net, donor, acceptor, 1)[1]   # D- -> L1
                    J1 += kb_2*(1 - (p_ox + p_sq))*p_L1 - kf_2*(1 - p_L1)*p_sq + kb_1*p_sq*p_L1 - kf_1*(1 - p_L1)*p_ox
    #print(cof_f_name)
    return J1

def getRateEquations_H1(net, donor: Cofactor, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2):   # Obtain rate eqn expression for dP_H1-/dt
    # Search for cofactors that have physical connections with each other
    cof_f_list = []
    cof_f_name = []

    cof_i_id = net.cofactor2id[donor]

    for cof_f_id in range(net.num_cofactor):
        if net.D[cof_i_id][cof_f_id] != 0:
            cof_f = net.id2cofactor[cof_f_id]
            cof_f_list.append(cof_f)
            cof_f_name.append(cof_f.name)
            J2 = 0
            for acceptor in cof_f_list:
                if acceptor.name == "L1":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # H1 -> L1
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # L1 -> H1
                    J2 += kb*(1 - p_L1)*p_H1 - kf*(1 - p_H1)*p_L1
                if acceptor.name == "L2":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # H1 -> L2
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # L2 -> H1
                    J2 += kb*(1 - p_L2)*p_H1 - kf*(1 - p_H1)*p_L2
                if acceptor.name == "H2":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # H1 -> H2
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # H2 -> H1
                    J2 += kb*(1 - p_H2)*p_H1 - kf*(1 - p_H1)*p_H2
                if acceptor.name == "D":
                    kf_2 = getHoppingRates(net, acceptor, donor, 2)[1]   # H1 -> D-
                    kb_2 = getHoppingRates(net, acceptor, donor, 2)[0]   # D= -> H1
                    kf_1 = getHoppingRates(net, donor, acceptor, 1)[0]   # H1 -> D
                    kb_1 = getHoppingRates(net, donor, acceptor, 1)[1]   # D- -> H1
                    J2 += kb_2*(1 - (p_ox + p_sq))*p_H1 - kf_2*(1 - p_H1)*p_sq + kb_1*p_sq*p_H1 - kf_1*(1 - p_H1)*p_ox
    #print(cof_f_name)
    return J2

def getRateEquations_L2(net, donor: Cofactor, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2):   # Obtain rate eqn expression for dP_L2-/dt
    # Extract info of reservoirs
    """
    reservoirInfo[cofactorID][n]
    n = 0 :reservoir name, 1 :cofactor object coupled to the reservoir, 2 :redox state of the cofactor,
    3 :number of electrons, 4: deltaG for electron exchange, 5: cofactor -> reservoir ET rate
    """
    res_cof_list = []
    for reservoir_id, info in net.reservoirInfo.items():
        name, cofactor, redox_state, num_electron, deltaG, rate = info
        res_cof_list.append(cofactor)

    # Search for cofactors that have physical connections with each other
    cof_f_list = []
    cof_f_name = []

    cof_i_id = net.cofactor2id[donor]

    for cof_f_id in range(net.num_cofactor):
        if net.D[cof_i_id][cof_f_id] != 0:
            cof_f = net.id2cofactor[cof_f_id]
            cof_f_list.append(cof_f)
            cof_f_name.append(cof_f.name)
            J3 = 0
            for acceptor in cof_f_list:
                if acceptor.name == "L1":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # L2 -> L1
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # L1 -> L2
                    J3 += kb*(1 - p_L1)*p_L2 - kf*(1 - p_L2)*p_L1
                if acceptor.name == "H1":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # L2 -> H1
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # H1 -> L2
                    J3 += kb*(1 - p_H1)*p_L2 - kf*(1 - p_L2)*p_H1
                if acceptor.name == "H2":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # L2 -> H2
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # H2 -> L2
                    J3 += kb*(1 - p_H2)*p_L2 - kf*(1 - p_L2)*p_H2
                if acceptor.name == "D":
                    kf_2 = getHoppingRates(net, acceptor, donor, 2)[1]   # L2 -> D-
                    kb_2 = getHoppingRates(net, acceptor, donor, 2)[0]   # D= -> L2
                    kf_1 = getHoppingRates(net, donor, acceptor, 1)[0]   # L2 -> D
                    kb_1 = getHoppingRates(net, donor, acceptor, 1)[1]   # D- -> L2
                    J3 += kb_2*(1 - (p_ox + p_sq))*p_L2 - kf_2*(1 - p_L2)*p_sq + kb_1*p_sq*p_L2 - kf_1*(1 - p_L2)*p_ox
            for res_id in range(net.num_reservoir):
                if donor.name == res_cof_list[res_id].name:
                    kout = net.reservoirInfo[res_id][5]
                    kin = kout * np.exp(net.beta*net.reservoirInfo[res_id][4])
                    #kin = net.getRate(net.reservoirInfo[res_id][5], net.reservoirInfo[res_id][4])
                    J3 += kin*p_L2 - kout*(1 - p_L2)
    #print(cof_f_name)
    return J3

def getRateEquations_H2(net, donor: Cofactor, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2):   # Obtain rate eqn expression for dP_H2-/dt
    # Extract info of reservoirs
    """
    reservoirInfo[cofactorID][n]
    n = 0 :reservoir name, 1 :cofactor object coupled to the reservoir, 2 :redox state of the cofactor,
    3 :number of electrons, 4: deltaG for electron exchange, 5: cofactor -> reservoir ET rate
    """
    res_cof_list = []
    for reservoir_id, info in net.reservoirInfo.items():
        name, cofactor, redox_state, num_electron, deltaG, rate = info
        res_cof_list.append(cofactor)

    # Search for cofactors that have physical connections with each other
    cof_f_list = []
    cof_f_name = []

    cof_i_id = net.cofactor2id[donor]

    for cof_f_id in range(net.num_cofactor):
        if net.D[cof_i_id][cof_f_id] != 0:
            cof_f = net.id2cofactor[cof_f_id]
            cof_f_list.append(cof_f)
            cof_f_name.append(cof_f.name)
            J4 = 0
            for acceptor in cof_f_list:
                if acceptor.name == "L1":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # H2 -> L1
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # L1 -> H2
                    J4 += kb*(1 - p_L1)*p_H2 - kf*(1 - p_H2)*p_L1
                if acceptor.name == "L2":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # H2 -> L2
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # L2 -> H2
                    J4 += kb*(1 - p_L2)*p_H2 - kf*(1 - p_H2)*p_L2
                if acceptor.name == "H1":
                    kf = getHoppingRates(net, donor, acceptor, 1)[0]   # H2 -> H1
                    kb = getHoppingRates(net, donor, acceptor, 1)[1]   # H1 -> H2
                    J4 += kb*(1 - p_H1)*p_H2 - kf*(1 - p_H2)*p_H1
                if acceptor.name == "D":
                    kf_2 = getHoppingRates(net, acceptor, donor, 2)[1]   # H2 -> D-
                    kb_2 = getHoppingRates(net, acceptor, donor, 2)[0]   # D= -> H2
                    kf_1 = getHoppingRates(net, donor, acceptor, 1)[0]   # H2 -> D
                    kb_1 = getHoppingRates(net, donor, acceptor, 1)[1]   # D- -> H2
                    J4 += kb_2*(1 - (p_ox + p_sq))*p_H2 - kf_2*(1 - p_H2)*p_sq + kb_1*p_sq*p_H2 - kf_1*(1 - p_H2)*p_ox
            for res_id in range(net.num_reservoir):
                if donor.name == res_cof_list[res_id].name:
                    kout = net.reservoirInfo[res_id][5]
                    kin = kout * np.exp(net.beta*net.reservoirInfo[res_id][4])
                    #kin = net.getRate(net.reservoirInfo[res_id][5], net.reservoirInfo[res_id][4])
                    J4 += kin*p_H2 - kout*(1 - p_H2)
    #print(cof_f_name)
    return J4

def getRateEquations_D(net, donor: Cofactor, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2):   # Obtain rate eqn expression for dP_D=/dt and dP_D-/dt
    # Extract info of reservoirs
    """
    reservoirInfo[cofactorID][n]
    n = 0 :reservoir name, 1 :cofactor object coupled to the reservoir, 2 :redox state of the cofactor,
    3 :number of electrons, 4: deltaG for electron exchange, 5: cofactor -> reservoir ET rate
    """
    res_cof_list = []
    for reservoir_id, info in net.reservoirInfo.items():
        name, cofactor, redox_state, num_electron, deltaG, rate = info
        res_cof_list.append(cofactor)

    # Search for cofactors that have physical connections with each other
    cof_f_list = []
    cof_f_name = []

    cof_i_id = net.cofactor2id[donor]

    for cof_f_id in range(net.num_cofactor):
        if net.D[cof_i_id][cof_f_id] != 0:
            cof_f = net.id2cofactor[cof_f_id]
            cof_f_list.append(cof_f)
            cof_f_name.append(cof_f.name)
            J5 = 0
            J6 = 0
            for acceptor in cof_f_list:
                if acceptor.name == "L1":
                    kf_2 = getHoppingRates(net, donor, acceptor, 2)[0]   # D= -> L1
                    kb_2 = getHoppingRates(net, donor, acceptor, 2)[1]   # L1 -> D-
                    kf_1 = getHoppingRates(net, donor, acceptor, 1)[0]   # D- -> L1
                    kb_1 = getHoppingRates(net, donor, acceptor, 1)[1]   # L1 -> D
                    J5 += kb_2*(1 - p_L1)*p_sq - kf_2*(1 - (p_sq + p_ox))*p_L1              
                    J6 += kb_1*(1 - p_L1)*p_ox - kf_1*p_sq*p_L1 + kf_2*(1 - (p_sq + p_ox))*p_L1 - kb_2*(1-p_L1)*p_sq
                if acceptor.name == "L2":
                    kf_2 = getHoppingRates(net, donor, acceptor, 2)[0]   # D= -> L2
                    kb_2 = getHoppingRates(net, donor, acceptor, 2)[1]   # L2 -> D-
                    kf_1 = getHoppingRates(net, donor, acceptor, 1)[0]   # D- -> L2
                    kb_1 = getHoppingRates(net, donor, acceptor, 1)[1]   # L2 -> D
                    J5 += kb_2*(1 - p_L2)*p_sq - kf_2*(1 - (p_sq + p_ox))*p_L2
                    J6 += kb_1*(1 - p_L2)*p_ox - kf_1*p_sq*p_L2 + kf_2*(1 - (p_sq + p_ox))*p_L2 - kb_2*(1-p_L2)*p_sq
                if acceptor.name == "H1":
                    kf_2 = getHoppingRates(net, donor, acceptor, 2)[0]   # D= -> H1
                    kb_2 = getHoppingRates(net, donor, acceptor, 2)[1]   # H1 -> D-
                    kf_1 = getHoppingRates(net, donor, acceptor, 1)[0]   # D- -> H1
                    kb_1 = getHoppingRates(net, donor, acceptor, 1)[1]   # H1 -> D
                    J5 += kb_2*(1 - p_H1)*p_sq - kf_2*(1 - (p_sq + p_ox))*p_H1
                    J6 += kb_1*(1 - p_H1)*p_ox - kf_1*p_sq*p_H1 + kf_2*(1 - (p_sq + p_ox))*p_H1 - kb_2*(1-p_H1)*p_sq
                if acceptor.name == "H2":
                    kf_2 = getHoppingRates(net, donor, acceptor, 2)[0]   # D= -> H2
                    kb_2 = getHoppingRates(net, donor, acceptor, 2)[1]   # H2 -> D-
                    kf_1 = getHoppingRates(net, donor, acceptor, 1)[0]   # D- -> H2
                    kb_1 = getHoppingRates(net, donor, acceptor, 1)[1]   # H2 -> D
                    J5 += kb_2*(1 - p_H2)*p_sq - kf_2*(1 - (p_sq + p_ox))*p_H2
                    J6 += kb_1*(1 - p_H2)*p_ox - kf_1*p_sq*p_H2 + kf_2*(1 - (p_sq + p_ox))*p_H2 - kb_2*(1-p_H2)*p_sq
            for res_id in range(net.num_reservoir):
                if donor.name == res_cof_list[res_id].name:
                    kout = net.reservoirInfo[res_id][5]
                    kin = kout * np.exp(net.beta*net.reservoirInfo[res_id][4])
                    #kin = net.getRate(net.reservoirInfo[res_id][5], net.reservoirInfo[res_id][4])
                    J5 += kin*p_ox - kout*(1 - (p_sq + p_ox))
    # print(cof_f_name)
    return J5, J6   # J5: dP_D=/dt and J6: dP_D-/dt

def getReservoirFlux(net, name: str, pop: list):
    """
    Calculate the instantaneous net electron flux into the reservoir.
    pop = [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
    """
    reservoir_id = net.reservoir2id[name]
    name, cofactor, redox_state, num_electron, deltaG, rate = net.reservoirInfo[reservoir_id]
    kout = net.reservoirInfo[reservoir_id][5]
    #kin = net.getRate(kout, net.reservoirInfo[reservoir_id][4])
    kin = kout * np.exp(net.beta*net.reservoirInfo[reservoir_id][4])
    num_electron = net.reservoirInfo[reservoir_id][3]

    p_ox = pop[0]
    p_sq = pop[1]
    p_red = 1 - (p_ox + p_sq)
    p_L1 = pop[2]
    p_L1_red = 1 - p_L1
    p_H1 = pop[3] 
    p_H1_red = 1 - p_H1
    p_L2 = pop[4]
    p_L2_red = 1 - p_L2
    p_H2 = pop[5]
    p_H2_red = 1 - p_H2
    
    flux = 0
    if cofactor.name == "D":
        flux = (p_red*kout - p_ox*kin)*num_electron
        #flux = (p_red*kout - p_ox*kin)*1
    if cofactor.name == "L2":
        flux = (p_L2_red*kout - p_L2*kin)*num_electron
    if cofactor.name == "H2":
        flux = (p_H2_red*kout - p_H2*kin)*num_electron
    
    return flux

def getBoltzmann(net, name: str, slope):
    data = [0, 0, 0]
    if name == "Low-potential branch":
        # E_sq = slope*2
        # E_L1 = slope*1
        # E_L2 = 0
        μ = -(-0.4 + slope*2)
        E_sq = 0.4
        E_L1 = -(-0.4 + slope*1)
        E_L2 = -(-0.4 + slope*2)
        partition_function = np.exp(net.beta*(μ-E_sq)) + np.exp(net.beta*(μ-E_L1)) + np.exp(net.beta*(μ-E_L2))
        data[0] = np.exp(net.beta*(μ-E_sq))/(1 + np.exp(net.beta*(μ-E_sq)))   # p_sq
        data[1] = np.exp(net.beta*(μ-E_L1))/(1 + np.exp(net.beta*(μ-E_L1)))     # p_L1
        data[2] = np.exp(net.beta*(μ-E_L2))/(1 + np.exp(net.beta*(μ-E_L2)))     # p_L2
    if name == "High-potential branch":
        # E_red = -slope*2
        # E_H1 = -slope*1
        # E_H2 = 0
        μ = -(0.4 - slope*2)
        E_red = -0.4
        E_H1 = -(0.4 - slope*1)
        E_H2 = -(0.4 - slope*2)
        partition_function = np.exp(net.beta*(μ-E_red)) + np.exp(net.beta*(μ-E_H1)) + np.exp(net.beta*(μ-E_H2))
        data[0] = np.exp(net.beta*(μ-E_red))/(1 + np.exp(net.beta*(μ-E_red)))    # p_red
        data[1] = np.exp(net.beta*(μ-E_H1))/(1 + np.exp(net.beta*(μ-E_H1)))    # p_H1
        data[2] = np.exp(net.beta*(μ-E_H2))/(1 + np.exp(net.beta*(μ-E_H2)))    # p_H2

    return data

def getSCflux(net, cof_i: Cofactor, cof_f1: Cofactor, cof_f2: Cofactor, cof_f3: Cofactor, pop: list):
    """
    Determine the short-circuiting flux from cof_i.
    cof_i: Donor cofactor in the low-potential branch (D, L1, L2)
    cof_f1: D, cof_f2: H1, cof_f3: H2
    """

    p_ox = pop[0]
    p_sq = pop[1]
    p_red = 1 - (p_ox + p_sq)
    p_L1 = pop[2]
    p_L1_red = 1 - p_L1
    p_H1 = pop[3] 
    p_H1_red = 1 - p_H1
    p_L2 = pop[4]
    p_L2_red = 1 - p_L2
    p_H2 = pop[5]
    p_H2_red = 1 - p_H2

    cof_f_list = [cof_f1, cof_f2, cof_f3]
    # cof_i: D-
    if cof_i.name == "D":
        acceptor_list = []
        for cof_f in cof_f_list:
            if cof_f.name != cof_i.name:
                acceptor_list.append(cof_f)
                SCflux = 0
                for acceptor in acceptor_list:
                    if acceptor.name == "H1":
                        kf = getHoppingRates(net, cof_i, acceptor, 1)[0]   # D- -> H1
                        kb = getHoppingRates(net, cof_i, acceptor, 1)[1]   # H1 -> D
                        SCflux += p_sq*p_H1*kf - p_ox*p_H1_red*kb
                        #SCflux += p_sq*p_H1*kf
                    if acceptor.name == "H2":
                        kf = getHoppingRates(net, cof_i, acceptor, 1)[0]   # D- -> H2
                        kb = getHoppingRates(net, cof_i, acceptor, 1)[1]   # H2 -> D
                        SCflux += p_sq*p_H2*kf - p_ox*p_H2_red*kb
                        #SCflux += p_sq*p_H2*kf
    # cof_i: L1
    if cof_i.name == "L1":
        acceptor_list = []
        for cof_f in cof_f_list:
            if cof_f.name != cof_i.name:
                acceptor_list.append(cof_f)
                SCflux = 0
                for acceptor in acceptor_list:
                    if acceptor.name == "D":
                        kf = getHoppingRates(net, acceptor, cof_i, 2)[1]   # L1 -> D-
                        kb = getHoppingRates(net, acceptor, cof_i, 2)[0]   # D= -> L1
                        SCflux += p_L1_red*p_sq*kf - p_L1*p_red*kb
                        #SCflux += p_L1_red*p_sq*kf
                    if acceptor.name == "H1":
                        kf = getHoppingRates(net, cof_i, acceptor, 1)[0]   # L1 -> H1
                        kb = getHoppingRates(net, cof_i, acceptor, 1)[1]   # H1 -> L1
                        SCflux += p_L1_red*p_H1*kf - p_L1*p_H1_red*kb
                        #SCflux += p_L1_red*p_H1*kf
                    if acceptor.name == "H2":
                        kf = getHoppingRates(net, cof_i, acceptor, 1)[0]   # L1 -> H2
                        kb = getHoppingRates(net, cof_i, acceptor, 1)[1]   # H2 -> L1
                        SCflux += p_L1_red*p_H2*kf - p_L1*p_H2_red*kb
                        #SCflux += p_L1_red*p_H2*kf

    # cof_i: L2
    if cof_i.name == "L2":
        acceptor_list = []
        for cof_f in cof_f_list:
            if cof_f.name != cof_i.name:
                acceptor_list.append(cof_f)
                SCflux = 0
                for acceptor in acceptor_list:
                    if acceptor.name == "D":
                        kf = getHoppingRates(net, acceptor, cof_i, 2)[1]   # L2 -> D-
                        kb = getHoppingRates(net, acceptor, cof_i, 2)[0]   # D= -> L2
                        SCflux += p_L2_red*p_sq*kf - p_L2*p_red*kb
                        #SCflux += p_L2_red*p_sq*kf
                    if acceptor.name == "H1":
                        kf = getHoppingRates(net, cof_i, acceptor, 1)[0]   # L2 -> H1
                        kb = getHoppingRates(net, cof_i, acceptor, 1)[1]   # H1 -> L2
                        SCflux += p_L2_red*p_H1*kf - p_L2*p_H1_red*kb
                        #SCflux += p_L2_red*p_H1*kf
                    if acceptor.name == "H2":
                        kf = getHoppingRates(net, cof_i, acceptor, 1)[0]   # L2 -> H2
                        kb = getHoppingRates(net, cof_i, acceptor, 1)[1]   # H2 -> L2
                        SCflux += p_L2_red*p_H2*kf - p_L2*p_H2_red*kb
                        #SCflux += p_L2_red*p_H2*kf

    return SCflux

def getpop(net, RateEqns: list, p_ox, p_sq, p_L1, p_H1, p_L2, p_H2):
    # Numerically solve set of non-linear equations
    # Core part of the mean-field model calculation!!
    pop = []
    success = False
    while success == False:
        try:
            pop = nsolve(RateEqns, [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2], [0, 0, 0, 0.8, 0.5, 0.5])
            for i in range(len(pop)):
                #if 0 < pop[i] < 1:
                if 0 < pop[0] < 1 and 0 < pop[1] < 1 and 0 < pop[2] < 1 and 0 < pop[3] < 1 and 0 < pop[4] < 1 and 0 < pop[5] < 1:
                    success = True
                    print('My initial guess succeeded!')
                else:
                    success = False
                    print('The solutions given by your very first initial guess were unphysical. Try again...')

                    success2 = False
                    while success2 == False:
                        try:
                            probability_init = getInitialProbabilityVector(net)       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
                            print('Generated new intial guess:', probability_init)
                            pop = nsolve(RateEqns, [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2], probability_init)
                            for i in range(len(pop)):
                                #if 0 < pop[i] < 1:
                                if 0 < pop[0] < 1 and 0 < pop[1] < 1 and 0 < pop[2] < 1 and 0 < pop[3] < 1 and 0 < pop[4] < 1 and 0 < pop[5] < 1:
                                    success2 = True    # Terminate the while success2 == False loop
                                    success = True     # Terminate the while success == False loop
                                    print('Successfully found physical solutions!')
                                else:
                                    success2 = False
                                    print('The solutions were unphysical again. Try another initial guess...')
                        except ValueError:
                            success2 = False
                            print('Oops! Have to try another initial guess...')
        except ValueError:
            print('Oops! Could not find root within given tolerance using given initial guess...')
            success3 = False
            while success3 == False:
                try:
                    probability_init = getInitialProbabilityVector(net)       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
                    print('Generated new intial guess:', probability_init)
                    pop = nsolve(RateEqns, [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2], probability_init)
                    for i in range(len(pop)):
                        #if 0 < pop[i] < 1:
                        if 0 < pop[0] < 1 and 0 < pop[1] < 1 and 0 < pop[2] < 1 and 0 < pop[3] < 1 and 0 < pop[4] < 1 and 0 < pop[5] < 1:
                            success3 = True    # Terminate the while success3 == False loop
                            success = True     # Terminate the while success == False loop
                            print('Successfully found physical solutions!')
                except ValueError:
                    success3 = False
                    print('Oops! Have to try another initial guess...')
    return pop

