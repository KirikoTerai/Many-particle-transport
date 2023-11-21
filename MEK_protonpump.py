import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
from collections import defaultdict as Dict


class Cofactor():
    def __init__(self, name: str, energy: list):
        """
        Initialize this cofactor object, with property: name, and redox potentials
        Arguments:
            name {str} -- Name of the hopping site
            redox {list} -- List of intrinsic free energies of the site
        """
        self.name = name
        self.energy = energy   
        self.capacity = len(energy)    # The number of particles (electron or proton) the site can occupy

    def __str__(self) -> str:         #__str__:a built-in function that computes the "informal" string representations of an object
        """
        Return a string representation of the cofactor
        Returns:
            str -- String representation of the cofactor
        """
        s = ""
        # Initialize with site name
        s += "Site Name: {}\n".format(self.name)     #\n:new line in string
        s += "------------ \n"     #Draw a line between site info (looks cuter!)
        # Print site info, with state_id and relative free energy
        for i in range(len(self.energy)):
            s += "Occupation: {}, Intrinsic free energy: {}\n".format(i, self.energy[i])

        return s


class Network():
    def __init__(self):
        """
        Initialize the whole system
        NOTICE: the initialized Network instance has nothing in it, use other functions to insert information
        """
        # system-specific data structure and parameters
        self.num_cofactor = 0
        self.num_state = 1
        self.adj_num_state = 1 # adjusted number of states with max particle ceiling
        self.allow = [] # list of allowed states with max particle ceiling
        self.id2cofactor = dict()  # key-value mapping is id-cofactor
        self.cofactor2id = dict()  # key-value mapping is cofactor-id
        self.adjacencyList = list()
        self.D = None  # not defined
        self.K = None  # not defined
        self.modK = None
        self.siteCapacity = []  # indexing is id-site_capacity
        self.num_reservoir = 0
        self.reservoirInfo = dict()    # key-value mapping is id-reservoir name, cofactor, redox_state, num_electron, deltaG, rate
        self.id2reservoir=dict()    # key-value mapping is id-reservoir name
        self.reservoir2id=dict()    # key-value mapping is reservoir name-id
        self.max_electrons = None
        self.beta = 39.06  # unit: 1/kT in 1/eV
        self.alpha = 0.5

    def __str__(self) -> str:
        """
        Return a string representation of the defined Network
        Returns:
            str -- String representation of the Network
        """
        s = ""
        # 1. print Cofactors information
        s += "Total number of sites in the Network: {}\n".format(self.num_cofactor)
        if self.num_cofactor == 0:
            s += "There are no sites in the Network, please add sites first!\n"
            return s
        for idx, cofactor in self.id2cofactor.items():
            s += "ID: {}\n".format(idx)
            s += cofactor.__str__()
        # 2. print Adjacency matrix information
        if isinstance(self.D, type(None)):
            s += "------------\n"
            s += "The adjacency matrix has not been calculated yet!\n"
            s += "------------\n"
            return s
        s += "------------\n"
        s += "Adjacency matrix for the Network\n"
        s += "------------\n"
        s += self.D.__str__()
        # 3. print Reservoir information
        if self.num_reservoir == 0:
            s += "------------\n"
            s += "There are no reservoir defined in this system!\n"
            s += "------------\n"
        else:
            s += "------------\n"
            s += "There are {} reservoirs in this system.\n"
            s += "------------\n"
            for res_id, info in self.reservoirInfo.items():
                name, cofactor, redox_state, num_electron, deltaG, rate = info
                s += "------------\n"
                s += "Reservoir ID: {}, Reservoir Name: {}, connects with Site ID {} with occupation state {}\n".format(res_id, name, self.cofactor2id[cofactor], redox_state)
                s += "Number of particles it exchanges at a time: {}\n".format(num_electron)
                s += "Delta G for transfering electron/proton: {}\n".format(deltaG)
                s += "ET or PT rate: {}\n".format(rate)
                s += "------------\n"

        return s

    def addCofactor(self, cofactor: Cofactor):
        """
        Add cofactor into this Network
        Arguments:
            cofactor {Cofactor} -- Cofactor object
        """
        self.num_state *= (cofactor.capacity +1)        # The total number of possible states is equal to the product of sitecapacities+1 of each site.
                                             # (ex.) "Cofactor_1":0,1, "Cofactor_2":0,1,2 -> num_states=(cap_1+1)*(cap_2+1)=(1+1)*(2+1)=2*3=6
        self.id2cofactor[self.num_cofactor] = cofactor   #Starts with self.num_cofactor=0, Gives an ID to cofactors that are added one by one
        self.cofactor2id[cofactor] = self.num_cofactor   #ID of the cofactor just added is basically equal to how many cofactors present in the network
        # self.siteCapacity[self.num_cofactor] = cofactor.capacity
        self.siteCapacity.append(cofactor.capacity)    #Trajectory of cofactor -> id -> capacity of cofactor
        self.num_cofactor += 1    #The number of cofactor counts up

    def addConnection(self, cof1: Cofactor, cof2: Cofactor, distance: float):
        """
        "Physically" connect two cofactors in the network, allow electron to flow
        Arguments:
            cof1 {Cofactor} -- Cofactor instance
            cof2 {Cofactor} -- Cofactor instance
            distance {float} -- Distance between two cofactors, unit in angstrom
        """
        self.adjacencyList.append((self.cofactor2id[cof1], self.cofactor2id[cof2], distance))  #Append ID of cof1, ID of cof2 and distance between cof1 and cof2 to adjacency list 

    def addReservoir(self, name: str, cofactor: Cofactor, redox: int, num_electron: int, deltaG: float, rate: float):
        """
        Add an electron reservoir to the network: which cofactor it exchanges electrons with, how many electrons are exchanged at a time, the deltaG of the exchange, and the rate
        Arguments:
            name {str} -- Name of the reservoir
            cofactor {Cofactor} -- Cofactor the reservoir exchanges electrons
            redox {int} -- Redox state of the cofactor that exchanges electrons with 
            num_electron {int} -- Number of electrons exchanged at a time
            deltaG {float} -- DeltaG of the exchange
            rate {float} -- In rate
        """
        # key: (reservoir_id, cofactor_id)
        # value: list of six variables, [name, cofactor, redox_state, num_electron, deltaG, rate]
        self.id2reservoir[self.num_reservoir] = name
        self.reservoir2id[name] = self.num_reservoir
        self.reservoirInfo[self.num_reservoir] = [name, cofactor, redox, num_electron, deltaG, rate]
        self.num_reservoir += 1

    def set_Max_Electrons(self, max_electrons: int):
            self.max_electrons = max_electrons

    def energy_microstate(self, eps: np.array, Vm: float):
        """
        Assign energy to each microstate.
        {Input}
        eps[i][j]: electro static interaction between site i and j
        """
        L = 30    #distance between N-side and P-side

        energy_list = []   #energy_list = [energy of cof1, energy of cof2, ...]
        for n in range(self.num_cofactor):
            cofactor = self.id2cofactor[n]
            energy_list.append(cofactor.energy[0])

        energy_microstate_list = []   #energy_microstate_list = [energy of microstate 1, energy of microstate 2, ...]
        for microstate_No in range(self.adj_num_state):   # loop thru all the microstates
            #print(self.idx2state(microstate_No), microstate_No)

            state_energy_firstterm = 0   #calculate the first term  (contributions of intrinsic free energies of the sites)
            for cofactor_energy in energy_list:
                index = energy_list.index(cofactor_energy)
                state_energy_firstterm += cofactor_energy*self.idx2state(microstate_No)[index]
            
            state_energy_secondterm = 0   #calculate the second term (contributions of electrostatics)
            for index1 in range(self.num_cofactor):
                for index2 in range(index1+1, self.num_cofactor):
                    state_energy_secondterm += eps[index1][index2]*self.idx2state(microstate_No)[index1]*self.idx2state(microstate_No)[index2]

            state_energy_thirdterm = 0   #calculate the third term (contributions of membrane potential)
            for index3 in range(self.num_cofactor):
                if self.idx2state(microstate_No)[index3] == 1:
                    if index3 == 0:    #If index3 is site 1: q1 = 1, Z1 = 15
                        q1 = 1
                        Z1 = 15
                        state_energy_thirdterm += (q1*Z1*Vm)/L
                    if index3 == 1:    #If index3 is site 2: q2 = 1, Z2 = 25
                        q2 = 1
                        Z2 = 25
                        state_energy_thirdterm += (q2*Z2*Vm)/L
                    if index3 == 2:    #If index3 is site 3: q3 = -1, Z3 = 20
                        q3 = -1
                        Z3 = 20
                        state_energy_thirdterm += (q3*Z3*Vm)/L

            energy_microstate_list.append(state_energy_firstterm + state_energy_secondterm + state_energy_thirdterm)

            # print(energy_microstate_list)

        return energy_microstate_list


    def evolve(self, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.adj_num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return linalg.expm(self.K * t) @ pop_init

    def evolve_mod(self, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.adj_num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return linalg.expm(self.modK * t) @ pop_init

    #def stateEnergy(self, ):

    def forwardrate(self, kappa, alpha, deltaG: float, beta) -> float:
        """
        Calculate the forward transition rate between microstates
        -> eq (3) of "Proton-pumping mechanism of cytochrome c oxidase: A kinetic master-equation approach"
        """
        rate = kappa * np.exp(-alpha * deltaG * beta)
        return rate

    def backwardrate(self, kappa, alpha, deltaG: float, beta) -> float:
        """
        Calculate the backward transition rate between microstates
        -> eq (4) of "Proton-pumping mechanism of cytochrome c oxidase: A kinetic master-equation approach"
        """
        rate = kappa * np.exp((1-alpha) * deltaG * beta)
        return rate

    # def ET(self, deltaG: float, R: float, reorgE, beta, V) -> float:
    #     """
    #     Calculate the nonadiabatic ET rate according to Marcus theory
    #     Arguments:
    #         deltaG {float} -- reaction free energy, unit: eV
    #         R {float} -- distance for decay factor, unit: angstrom
    #         reorgE {float} -- reorganization energy, unit: eV
    #         beta {float} -- inverse of kT, unit: 1/eV
    #         V {float} -- electronic coupling, unit: eV
    #     Returns:
    #         float -- nonadiabatic ET rate, unit: 1/s
    #     """
    #     return (2*math.pi/self.hbar)*(self.V**2)*np.exp(-R)*(1/(math.sqrt(4*math.pi*(1/beta)*reorgE)))*np.exp(-beta*(deltaG + reorgE)**2/(4*reorgE))

    def constructAdjacencyMatrix(self):
        """
        Build adjacency matrix from the adjacency list
        """
        # obtain the dimension of this matrix
        dim = self.num_cofactor
        self.D = np.zeros((dim, dim), dtype=float)
        for item in self.adjacencyList:
            id1, id2, distance = item
            # we allow electron to flow back and forth between cofactors, thus D matrix is symmetric
            self.D[id1][id2] = self.D[id2][id1] = distance

    ####################################################
    ####  Core Functions for Building Rate Matrix   ####
    ####################################################

    # For the following functions, we make use of the internal labelling of the
    # states which uses one index which maps to the occupation number
    # representation [n1, n2, n3, ..., nN] and convert to idx in the rate
    # back and forth with state2idx() and idx2state() functions.

    def state2idx(self, state: list) -> int:
        """
        Given the list representation of the state, return index number in the main rate matrix
        Arguments:
            state {list} -- List representation of the state
        Returns:
            int -- Index number of the state in the main rate matrix
        """
        idx = 0
        N = 1
        for i in range(self.num_cofactor):
            idx += state[i] * N
            N *= (self.siteCapacity[i] + 1)

        return idx

    def idx2state(self, idx: int) -> list:
        """
        Given the index number of the state in the main rate matrix, return the list representation of the state
        Arguments:
            idx {int} -- Index number of the state in the main rate matrix
        Returns:
            list -- List representation of the state
        """
        state = []
        for i in range(self.num_cofactor):
            div = self.siteCapacity[i] + 1
            idx, num = divmod(idx, div)
            state.append(num)

        return state

    def constructStateList(self) -> list:
        """
        Go through all the possible states and make list of the subset with the allowed number of particles
        """
        self.allow = []
        if self.max_electrons == None:
            self.max_electrons = sum([site for site in self.siteCapacity])
        for i in range(self.num_state):
            if sum(self.idx2state(i)) <= self.max_electrons:
                self.allow.append(i)
        self.adj_num_state = len(self.allow)

    def getRate(self, kb: float, deltaG: float):
        #rate is the rate you will input in the addReservoir
        #kb is the rate of cofactor -> reservoir
        rate = kb * np.exp(-self.beta*deltaG)
        return rate

    def connectStateRate(self, cof_i: int, red_i: int, cof_f: int, red_f: int, kf: float, kb: float, deltaG: float, num_electrons: int):
        """
        Add rate constant k between electron donor (cof_i) and acceptor (cof_f) with initial redox state and final redox state stated (red_i, red_f)
        ADDITION: this function combine with detailed balancing feature, helps to save initialization time.
        Arguments:
            cof_i {int} -- Donor cofactor ID
            red_i {int} -- Redox state for donor ID
            cof_f {int} -- Acceptor cofactor ID
            red_f {int} -- Redox state for acceptor ID
            k {float} -- forward state
            deltaG {float} -- deltaG between initial state and final state
        """
        for i in range(self.adj_num_state):
            # loop through all allowed states, to look for initial (donor) state
            if self.idx2state(self.allow[i])[cof_i] == red_i and self.idx2state(self.allow[i])[cof_f] == red_f:
                """
                ex. idx:some (allowed) number -> state:[0 1 1 0 2 3 ...]
                    idx2state(allow[i])[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                Basically, this "if" statement means: 
                "If cof_ith element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cof_i" and also 
                "If cof_fth element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cof_f"
                """
                for j in range(self.adj_num_state):
                    # loop through all allowed states, to look for final (acceptor) state
                    if self.idx2state(self.allow[j])[cof_i] == red_i - num_electrons and self.idx2state(self.allow[j])[cof_f] == red_f + num_electrons:
                        """
                        ex. idx:some allowed number -> state:[0 1 1 0 2 3 ...]
                            idx2state(allow[i])[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                        Basically, this "if" statement means: 
                        "If cof_ith element of the state:[0 1 1 0 2 3...] is equal to the (redox state - 1) (donates electron so this cofactor is oxidized) of the cof_i" and also 
                        "If cof_fth element of the state:[0 1 1 0 2 3...] is equal to the (redox state + 1) (accepts electron so this cofactor is reduced) of the cof_f"
                           """
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(self.allow[i]), [cof_i, cof_f])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(self.idx2state(self.allow[j]), [cof_i, cof_f])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):   # Checking that sites other than cof_i and cof_f hasn't changed
                            # i and j state found!
                            self.K[j][i] += kf  # add population of final state, forward process
                            self.K[i][i] -= kf  # remove population of initial state, forward process   #Diagonal elements are the negative sum of the other elements in the same column
                            self.K[i][j] += kb  # add population of initial state, backward process
                            self.K[j][j] -= kb  # remove population of final sate, backward process

    def connectReservoirRate(self, cof_id: int, red_i: int, red_f: int, k: float, deltaG: float):
        """
        Add rate constant k between red_i and red_f of a cofactor, which is connected to a reservoir
        ADDITION: this function combine with detailed balancing feature, helps to save initialization time.
        Arguments:
            cof_id {int} -- Cofactor ID
            red_i {int} -- Redox state for cofactor
            red_f {int} -- Redox state for cofactor
            k {float} -- forward rate (cofactor -> reservoir)
            deltaG {float} -- deltaG between initial state and final state
        """
        #if self.max_electrons == None:
        self.max_electrons = sum([site for site in self.siteCapacity])
        for i in range(self.adj_num_state):
            # loop through all allowed states, to look for initial (donor) state
            #if sum(self.idx2state(self.allow[i])) <= self.max_electrons:
             if self.idx2state(self.allow[i])[cof_id] == red_i:
                    for j in range(self.adj_num_state):
                    # loop through all allowed states, to look for final (acceptor) state
                        if self.idx2state(self.allow[j])[cof_id] == red_f:
                            # initial, final state found! check other electron conservation
                            I = np.delete(self.idx2state(self.allow[i]), [cof_id])
                            J = np.delete(self.idx2state(self.allow[j]), [cof_id])
                            if np.array_equal(I, J):
                                # i and j state found!
                                kf = k  # forward rate
                                kb = k * np.exp(self.beta*deltaG)
                                self.K[j][i] += kf  # add population of final state, forward process
                                self.K[i][i] -= kf  # remove population of initial state, forward process
                                self.K[i][j] += kb  # add population of initial state, backward process
                                self.K[j][j] -= kb  # remove population of final state, backward process

    def H2O_productionRate(self, cof_H1_id: int, red_H1_i: int, red_H1_f: int, cof_E3_id: int, red_E3_i: int, red_E3_f: int, k: float, deltaG: float):
        for i in range(self.adj_num_state):
            if self.idx2state(self.allow[i])[cof_H1_id] == red_H1_i and self.idx2state(self.allow[i])[cof_E3_id] == red_E3_i:
                for j in range(self.adj_num_state):
                    if self.idx2state(self.allow[j])[cof_H1_id] == red_H1_f and self.idx2state(self.allow[j])[cof_E3_id] == red_E3_f:
                        # initial(i) and final(f) state found! Check conservation of other sites
                        I = np.delete(self.idx2state(self.allow[i]), [cof_H1_id, cof_E3_id])
                        J = np.delete(self.idx2state(self.allow[j]), [cof_H1_id, cof_E3_id])
                        if np.array_equal(I, J):
                            kf = k  # forward rate
                            #kb = k * np.exp(self.beta*deltaG)
                            kb = 0
                            self.K[j][i] += kf  # add population of final state, forward process
                            self.K[i][i] -= kf  # remove population of initial state, forward process
                            self.K[i][j] += kb  # add population of initial state, backward process
                            self.K[j][j] -= kb  # remove population of final state, backward process

    def doNotAddRate(self):
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                # Do not add Rate. This is to avoid double counting the rate constant for H2O production
                self.K[j][i] += 0 
                self.K[i][i] -= 0 
                self.K[i][j] += 0 
                self.K[j][j] -= 0 

    def addMultiElectronConnection(self, cof_i, cof_f, donor_state: int, acceptor_state: int, num_electrons, k):
        i = self.cofactor2id[cof_i]
        f = self.cofactor2id[cof_f]   # Finding the name of cofactor of the ijth of the adjacency matrix
        deltaG = sum([cof_i.redox[donor_state-num_electrons + n] - cof_f.redox[acceptor_state+n] for n in range(0, num_electrons)])
        self.connectStateRate(i, donor_state, f, acceptor_state, k, deltaG, num_electrons)   #Adding the rate constant to rate matrix

    def constructRateMatrix(self, H1: Cofactor, H2: Cofactor, E3: Cofactor, kappa_11: float, kappa_22: float, kappa_33: float, kappa_12:float, kp: float):
        """
        Build rate matrix
        """
        # initialize the rate matrix with proper dimension
        self.K = np.zeros((self.adj_num_state, self.adj_num_state), dtype=float)      #The dimension of the rate matrix is basically equal to the total number of states
        # loop through cofactor_id in adjacency matrix
        """
        Take the adjacency matrix which is weighted by the distance to construct the full rate matrix
        """
        for i in range(self.num_cofactor):
            for j in range(i+1, self.num_cofactor):   # These two "for" loops take care of (upper triangular - diagonal) part of the adjacency matrix
                if self.D[i][j] != 0:  # cofactor i and j are connected!  !=:not equal to
                    cof_i = self.id2cofactor[i]
                    cof_f = self.id2cofactor[j]
                    if cof_i.name == "H1" and cof_f.name == "H2":
                        for donor_state in range(1, cof_i.capacity+1):    #This is correct!!!! Python is weird      #cof.capacity=maximum number of electrons the cofactor can occupy
                            for acceptor_state in range(0, cof_f.capacity):    #This is correct!!!! Python is weird
                                deltaG = cof_f.energy[acceptor_state] - cof_i.energy[donor_state-1]  #This is correct!!!! Python is weird
                                # forward is H1 -> H2, kappa_12 is the H1 -> H2 rate
                                #print('H1 -> H2 process')
                                kf = self.forwardrate(kappa_12, self.alpha, deltaG, self.beta)
                                kb = self.backwardrate(kappa_12, self.alpha, deltaG, self.beta)
                                self.connectStateRate(i, donor_state, j, acceptor_state, kf, kb, deltaG, 1)   #Adding the rate constant to rate matrix. The last parameter is 1 because these are all 1-electron transfers!
        for ii in range(self.adj_num_state):
            for jj in range(self.adj_num_state):        
                # loop through reservoirInfo to add reservoir-related rate
                for reservoir_id, info in self.reservoirInfo.items():
                    name, cofactor, redox_state, num_electron, deltaG, rate = info
                    final_redox_state = redox_state - num_electron
                    if name == "oxygen1":
                        cof_H1_id = self.cofactor2id[H1]
                        cof_E3_id = self.cofactor2id[E3]
                        if self.idx2state(self.allow[ii])[cof_H1_id] == 1 and self.idx2state(self.allow[ii])[cof_E3_id] == 1:
                            if self.idx2state(self.allow[jj])[cof_H1_id] == 0 and self.idx2state(self.allow[jj])[cof_E3_id] == 0:
                                I = np.delete(self.idx2state(self.allow[ii]), [cof_H1_id, cof_E3_id])
                                J = np.delete(self.idx2state(self.allow[jj]), [cof_H1_id, cof_E3_id])
                                if np.array_equal(I, J):
                                    #print('kp process')
                                    kf = kp
                                    kb = 0
                                    self.K[jj][ii] += kf
                                    self.K[ii][ii] -= kf
                                    self.K[ii][jj] += kb
                                    self.K[jj][jj] -= kb                
                    if name == "oxygen2":
                        # Do not add Rate. This is to avoid double counting the rate constant for H2O production
                        self.doNotAddRate()
                    if name == "N-side":
                        cof_H1_id = self.cofactor2id[H1]
                        if self.idx2state(self.allow[ii])[cof_H1_id] == 0 and self.idx2state(self.allow[jj])[cof_H1_id] == 1:
                            I = np.delete(self.idx2state(self.allow[ii]), [cof_H1_id])
                            J = np.delete(self.idx2state(self.allow[jj]), [cof_H1_id])
                            if np.array_equal(I, J):
                                #print('N-side process')
                                # forward is reservoir -> cofactor, deltaG is cofactor -> reservoir rate, kappa_11 is the rate (0, x, x) -> (1, x, x)
                                kf = self.forwardrate(kappa_11, self.alpha, -deltaG, self.beta)
                                kb = self.backwardrate(kappa_11, self.alpha, -deltaG, self.beta)
                                self.K[jj][ii] += kf
                                self.K[ii][ii] -= kf
                                self.K[ii][jj] += kb
                                self.K[jj][jj] -= kb
                    if name == "P-side":
                        cof_H2_id = self.cofactor2id[H2]
                        if self.idx2state(self.allow[ii])[cof_H2_id] == 0 and self.idx2state(self.allow[jj])[cof_H2_id] == 1:
                            I = np.delete(self.idx2state(self.allow[ii]), [cof_H2_id])
                            J = np.delete(self.idx2state(self.allow[jj]), [cof_H2_id])
                            if np.array_equal(I, J):
                                #print('P-side process')
                                # forward is reservoir -> cofactor, deltaG is cofactor -> reservoir rate, kappa_22 is the rate (x, 0, x) -> (x, 1, x)                      
                                kf = self.forwardrate(kappa_22, self.alpha, -deltaG, self.beta)
                                kb = self.backwardrate(kappa_22, self.alpha, -deltaG, self.beta)
                                self.K[jj][ii] += kf
                                self.K[ii][ii] -= kf
                                self.K[ii][jj] += kb
                                self.K[jj][jj] -= kb
                    if name == "CytC":
                        cof_E3_id = self.cofactor2id[E3]
                        if self.idx2state(self.allow[ii])[cof_E3_id] == 0 and self.idx2state(self.allow[jj])[cof_E3_id] == 1:
                            I = np.delete(self.idx2state(self.allow[ii]), [cof_E3_id])
                            J = np.delete(self.idx2state(self.allow[jj]), [cof_E3_id])
                            if np.array_equal(I, J):
                                #print('CytC process')
                                # forward is reservoir -> cofactor, deltaG is cofactor -> reservoir rate, kappa_33 is the rate (x, x, 0) -> (x, x, 1)
                                kf = self.forwardrate(kappa_33, self.alpha, -deltaG, self.beta)
                                kb = self.backwardrate(kappa_33, self.alpha, -deltaG, self.beta)
                                self.K[jj][ii] += kf
                                self.K[ii][ii] -= kf
                                self.K[ii][jj] += kb
                                self.K[jj][jj] -= kb
                    else:
                        if name == "oxygen1":
                            # Do not add Rate. This is to avoid double counting the rate constant for H2O production
                            self.doNotAddRate()

    def constructRateMatrix_Mod(self, H1:Cofactor, H2:Cofactor, E3: Cofactor, eps: np.array, kappa_11, kappa_22, kappa_33, kappa_12, kp, Vm):
        energy_microstate = self.energy_microstate(eps, Vm)

        cof_list = []
        for n in range(self.num_cofactor):
            cof_list.append(self.id2cofactor[n])

        self.modK = np.zeros((self.adj_num_state, self. adj_num_state), dtype = float)

        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if i != j:
                    if self.K[j][i] != 0:
                        for donorID in range(self.num_cofactor):
                            for acceptorID in range(donorID+1, self.num_cofactor):
                                if self.idx2state(self.allow[i])[donorID] == 1 and self.idx2state(self.allow[i])[acceptorID] == 0:
                                    if self.idx2state(self.allow[j])[donorID] == 0 and self.idx2state(self.allow[j])[acceptorID] == 1:
                                        I = np.delete(self.idx2state(self.allow[i]), [donorID, acceptorID])
                                        J = np.delete(self.idx2state(self.allow[j]), [donorID, acceptorID])
                                        if np.array_equal(I, J):
                                            deltaG = energy_microstate[j] - energy_microstate[i]
                                            # print(deltaG)
                                            kf = self.forwardrate(kappa_12, self.alpha, deltaG, self.beta)
                                            kb = self.backwardrate(kappa_12, self.alpha, deltaG, self.beta)
                                            self.modK[j][i] += kf
                                            self.modK[i][i] -= kf
                                            self.modK[i][j] += kb
                                            self.modK[j][j] -= kb
                        for reservoir_id, info in self.reservoirInfo.items():
                            name, cofactor, redox_state, num_electron, deltaG, rate = info
                            if name == "oxygen1":
                                cof_H1_id = self.cofactor2id[H1]
                                cof_E3_id = self.cofactor2id[E3]
                                if self.idx2state(self.allow[i])[cof_H1_id] == 1 and self.idx2state(self.allow[i])[cof_E3_id] == 1:
                                    if self.idx2state(self.allow[j])[cof_H1_id] == 0 and self.idx2state(self.allow[j])[cof_E3_id] == 0:
                                        I = np.delete(self.idx2state(self.allow[i]), [cof_H1_id, cof_E3_id])
                                        J = np.delete(self.idx2state(self.allow[j]), [cof_H1_id, cof_E3_id])
                                        if np.array_equal(I, J):
                                            #print("######kp process######")
                                            #print("final:",self.idx2state(self.allow[j]), "initial:", self.idx2state(self.allow[i]))
                                            deltaG_mod = energy_microstate[j] - energy_microstate[i]
                                            #print("energy of final=", energy_microstate[j], "energy of initial=", energy_microstate[i])
                                            #print("Gj - Gi=", deltaG_mod)
                                            kf = kp
                                            kb = 0
                                            self.modK[j][i] += kf
                                            self.modK[i][i] -= kf
                                            self.modK[i][j] += kb
                                            self.modK[j][j] -= kb
                                #self.H2O_productionRate_Mod(cof_H1_id, redox_state, final_redox_state, cof_E3_id, redox_state, final_redox_state, rate, deltaG)
                            if name == "oxygen2":
                                # Do not add Rate. This is to avoid double counting the rate constant for H2O production
                                self.modK[j][i] += 0 
                                self.modK[i][i] -= 0 
                                self.modK[i][j] += 0 
                                self.modK[j][j] -= 0 
                                #self.doNotAddRate_Mod()
                            if name == "N-side":
                                cof_H1_id = self.cofactor2id[H1]
                                if self.idx2state(self.allow[i])[cof_H1_id] == 0 and self.idx2state(self.allow[j])[cof_H1_id] == 1:
                                    I = np.delete(self.idx2state(self.allow[i]), [cof_H1_id])
                                    J = np.delete(self.idx2state(self.allow[j]), [cof_H1_id])
                                    if np.array_equal(I, J):
                                        #print("######kappa_11 process######")
                                        #print("final:",self.idx2state(self.allow[j]), "initial:", self.idx2state(self.allow[i]))
                                        deltaG_mod = energy_microstate[j] - energy_microstate[i]
                                        #print("energy of final=", energy_microstate[j], "energy of initial=", energy_microstate[i])
                                        #print("Gj - Gi=", deltaG_mod)
                                        kf = self.forwardrate(kappa_11, self.alpha, deltaG_mod, self.beta)   #kappa_11 is the rate (0, x, x) -> (1, x, x)
                                        kb = self.backwardrate(kappa_11, self.alpha, deltaG_mod, self.beta)
                                        self.modK[j][i] += kf
                                        self.modK[i][i] -= kf
                                        self.modK[i][j] += kb
                                        self.modK[j][j] -= kb
                            if name == "P-side":
                                cof_H2_id = self.cofactor2id[H2]
                                if self.idx2state(self.allow[i])[cof_H2_id] == 0 and self.idx2state(self.allow[j])[cof_H2_id] == 1:
                                    I = np.delete(self.idx2state(self.allow[i]), [cof_H2_id])
                                    J = np.delete(self.idx2state(self.allow[j]), [cof_H2_id])
                                    if np.array_equal(I, J):
                                        #print("######kappa_22 process######")
                                        #print("final:",self.idx2state(self.allow[j]), "initial:", self.idx2state(self.allow[i]))
                                        deltaG_mod = energy_microstate[j] - energy_microstate[i]
                                        #print("energy of final=", energy_microstate[j], "energy of initial=", energy_microstate[i])
                                        #print("Gj - Gi=", deltaG_mod)
                                        kf = self.forwardrate(kappa_22, self.alpha, deltaG_mod, self.beta)   #kappa_22 is the rate (x, 0, x) -> (x, 1, x)
                                        kb = self.backwardrate(kappa_22, self.alpha, deltaG_mod, self.beta)
                                        self.modK[j][i] += kf
                                        self.modK[i][i] -= kf
                                        self.modK[i][j] += kb
                                        self.modK[j][j] -= kb
                            if name == "CytC":
                                cof_E3_id = self.cofactor2id[E3]
                                if self.idx2state(self.allow[i])[cof_E3_id] == 0 and self.idx2state(self.allow[j])[cof_E3_id] == 1:
                                    I = np.delete(self.idx2state(self.allow[i]), [cof_E3_id])
                                    J = np.delete(self.idx2state(self.allow[j]), [cof_E3_id])
                                    if np.array_equal(I, J):
                                        #print("######kappa_33 process######")
                                        #print("final:",self.idx2state(self.allow[j]), "initial:", self.idx2state(self.allow[i]))
                                        deltaG_mod = energy_microstate[j] - energy_microstate[i]
                                        #print("energy of final=", energy_microstate[j], "energy of initial=", energy_microstate[i])
                                        #print("Gj - Gi=", deltaG_mod)
                                        kf = self.forwardrate(kappa_33, self.alpha, deltaG_mod, self.beta)   #kappa_33 is the rate (x, x, 0) -> (x, x, 1)
                                        kb = self.backwardrate(kappa_33, self.alpha, deltaG_mod, self.beta)
                                        self.modK[j][i] += kf
                                        self.modK[i][i] -= kf
                                        self.modK[i][j] += kb
                                        self.modK[j][j] -= kb
                            else:
                                if name == "oxygen1":
                                    # Do not add Rate. This is to avoid double counting the rate constant for H2O production
                                    self.modK[j][i] += 0 
                                    self.modK[i][i] -= 0 
                                    self.modK[i][j] += 0 
                                    self.modK[j][j] -= 0 
                                    #self.doNotAddRate_Mod()


    def simple_propensity(self, rateconstants, population, t, x: int):
#     Updates an array of propensities given a set of parameters
#     and an array of populations

        # Unpack population
        pop = population

        # Update propensities
        for i in range(self.adj_num_state):
            rateconstants[i]=self.K[i][x]    #x is the constant: this is where the microstate is!!
                                             #x changes over timestep because population changes and which transition happens depends on where the microstate is and the random number

    def sample_discrete(self, probs, x):      #Align probability and give a random number
        #Randomly sample an index with probability given by probs

        # Generate random number
        q = np.random.rand()

        # Find index     #Find next microstate
        i = 0
        p_sum = 0.0
        for i in range(self.adj_num_state):
            if x!=i:
                    p_sum += probs[i]
            if p_sum >= q:
                break

        return i

    def simple_update(self):
        updatearray = np.identity(self.adj_num_state, dtype=float)
        return updatearray

    def gillespie_draw(self, propensity_func, rateconstants, population, t, x):     #rateconstant/sum of column elements
#     Draws a reaction and the time it took to do that reaction.

#     Parameters
#     ----------
#     propensity_func : function
#         Function with call signature propensity_func(population, t, *args)
#         used for computing propensities. This function must return
#         an array of propensities.
#     population : ndarray
#         Current population of particles
#     t : float
#         Value of the current time.
#     args : tuple, default ()
#         Arguments to be passed to `propensity_func`.

#     Returns
#     -------
#     rxn : int
#         Index of reaction that occured.
#     time : float
#         Time it took for the reaction to occur.

        # Compute propensities
        propensity_func(rateconstants, population, t, x)
        #print(rateconstants)
        # Sum of propensities
        gamma = 0
        for i in range(len(rateconstants)):
            if x!=i:
                gamma += rateconstants[i]
        #print(gamma)
        # Find next time interval
        time = np.random.exponential(1.0 / gamma)
        # Compute discrete probabilities of each reaction
        rxn_probs = rateconstants / gamma        #props_sum is the gamma(=sum of rate constants)
        #print(rxn_probs)
        # Draw reaction from this distribution
        rxn = self.sample_discrete(rxn_probs, x)
        #print(rxn)
        #print(time)

        return rxn, time

    def gillespie_ssa(self, propensity_func, update, population_0, time_points, x):
#     Uses the Gillespie stochastic simulation algorithm to sample
#     from probability distribution of particle counts over time.

#     Parameters
#     ----------
#     propensity_func : function
#         Function of the form f(params, t, population) that takes the current
#         population of particle counts and return an array of propensities
#         for each reaction.
#     update : ndarray, shape (num_reactions, num_chemical_species)
#         Entry i, j gives the change in particle counts of species j
#         for chemical reaction i.
#     population_0 : array_like, shape (num_chemical_species)
#         Array of initial populations of all chemical species.
#     time_points : array_like, shape (num_time_points,)
#         Array of points in time for which to sample the probability
#         distribution.
#     args : tuple, default ()
#         The set of parameters to be passed to propensity_func.

#     Returns
#     -------
#     sample : ndarray, shape (num_time_points, num_chemical_species)
#         Entry i, j is the count of chemical species j at time
#         time_points[i].

    # Initialize output
        pop_out = np.empty((len(time_points), self.adj_num_state), dtype=np.int)

    # Initialize and perform simulation
        i_time = 1
        i = 0
        t = time_points[0]
        population = population_0.copy()
        pop_out[0,:] = population
        rateconstants = np.zeros(self.adj_num_state)
        while i < len(time_points):
            while t < time_points[i_time]:   #The timestep defined by time_points (in this case 0.2 s) do not proceed unless the sum of dt's does not reach the timestep.
            # draw the event and time step
                event, dt = self.gillespie_draw(propensity_func, rateconstants, population, t, x)    #event: state that it jumps to,    dt: time interval
                x = event                                                                            #new x depend on event and this x needs to be iterated into the next Gillespie loop
            # Update the population
                population_previous = population.copy()
                population += update[:,event]      #state a -> state b transition in time interval dt. In this time interval, pop of state a: 1->0 and pop of state_b: 0 -> 1
                     #If population = update[:,event], we can see transitions per time interval (like instantaneous transition in a time interval t)
                     #If population = update[:,event], we can see transitions per time interval (like average transition in a time interval t)
            # Increment time
                t += dt
        # Update the index
            i = np.searchsorted(time_points > t, True)
        # Update the population
            pop_out[i_time:min(i,len(time_points))] = population_previous
        # Increment index
            i_time = i

        return pop_out         #pop_out: population of each state

    def listConnectedStates(self) -> list:
        """
        List the states that are connected
        Returns:
            list -- [i (row number of K), j (column number of K), [ith state], [jth state]]
        """
        #search for rate matrix elements that are nonzero -> search for states that are connected
        connectedstates=[]
        for i in range(self.adj_num_state):     #loop through all the possible states
            for j in range(self.adj_num_state):   #looks through all the possible states
                if (self.K[i][j]!=0):
                    connectedstates.append([i,j, self.idx2state(self.allow[i]), self.idx2state(self.allow[j]), self.K[i][j]])   #list up states that are connected

        return connectedstates

    def totalnumelectron(self) -> list:
        """
        Calculate the total number of particles in the system.
        Returns:
            list of total number of particles in all the possible states
        """
        numelectrons=[]
        for i in range(self.adj_num_state):    #loop through all the possible states
            #print(self.idx2state(i))   #printing out all the possible states
            sum=0
            for j in range(self.num_cofactor):
                sum+=self.idx2state(self.allow[i])[j]    #sum the numbers included in the state list to get the total number of electrons in that state
            numelectrons.append(sum)
            #print(self.idx2state(i), sum)

        return numelectrons

    def listAllStates(self):
        """
        List up all the possible states of the model
        """
        allstates=[]
        for i in range(self.adj_num_state):
            allstates.append([self.idx2state(self.allow[i]), i])  #list: allstates=[state, ith]
        #print(len(allstates))     #len(allstates)=self.num_states
        # print(allstates)
        return allstates

    ########################################
    ####    Data Analysis Functions     ####
    ########################################

    def population(self, pop: np.array, cofactor: Cofactor, redox: int) -> float:
        """
        Calculate the population of a cofactor in a given redox state
         -> (ex.)pop=[1 0 0 2 5 ...]:len(pop)=num_state, pop is the population vector of the states. (pop[0]=population of state[0], pop[1]=population of state[1]...)

        Arguments:
            pop {numpy.array} -- Population vector     This is the population vector of the states. len(pop)=self.adj_num_state
            cofactor {Cofactor} -- Cofactor object
            redox {int} -- Redox state of the cofactor
        Returns:
            float -- Population of the cofactor at specific redox state
        """
        cof_id = self.cofactor2id[cofactor]
        ans = 0
        for i in range(len(pop)):
            #Loop through all the possible states
            if self.idx2state(self.allow[i])[cof_id] == redox:   #For every state, the number of electrons on each site is known, (ex.)state[0]=[1 2 0 3 2...], state[1]=[0 2 3 1 ...]
                # It loops through all the states to find where the cof th element of (ex.)state:[0 1 1 0 2 3...] is equal to the given redox state
                # Population of electron at each cofactor = redox state of that cofactor
                ans += pop[i]

        return ans

    def gillespie_pop2hopping_pop(self, gillespie_pop, t):
        """
        Probability of the states over at a given time
        Choosing time is choosing the row of gillespie_pop. This is the parameter "t"
        """
        sum=0
        for i in range(self.adj_num_state):
            sum+=gillespie_pop[t][i]     #one of the rows in gillespie's pop_out

        hopping_pop=[]
        for i in range(self.adj_num_state):
            hopping_pop.append(gillespie_pop[t][i]/sum)

        return hopping_pop

    def getCofactorRate(self, cof_i: Cofactor, red_i: int, cof_f: Cofactor, red_f: int, pop: np.array) -> float:
        """
        Calculate the instantaneous forward rate from cof_i to cof_f
        Arguments:
            cof_i {Cofactor} -- Cofactor object for initial cofactor
            red_i {int} -- Redox state for initial cofactor
            cof_f {Cofactor} -- Cofactor object for final cofactor
            red_f {int} -- Redox state for final cofactor
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous forward rate
        """
        cof_i_id = self.cofactor2id[cof_i]
        cof_f_id = self.cofactor2id[cof_f]
        flux = 0
        for i in range(self.adj_num_state):
            # loop through all states, to find initial state
            if self.idx2state(self.allow[i])[cof_i_id] == red_i and self.idx2state(self.allow[i])[cof_f_id] == red_f - 1:
                """
                This "if" statement means: 
                "If the element that corresponds to cof_i in the state:[0 1 1 0 2 3...] is equal to the redox state of cof_i (prior to donating an electron)" and
                "If the element that corresponds to cof_f in the state:[0 1 1 0 2 3...] is equal to the (redox state of cof_f -1) (prior to accepting an electron)"
                """
                for j in range(self.adj_num_state):
                    # loop through all states, to find final state
                    if self.idx2state(self.allow[j])[cof_i_id] == red_i - 1 and self.idx2state(self.allow[j])[cof_f_id] == red_f:
                        """
                        This "if" statement means: 
                        "If the element that corresponds to cof_i in the state:[0 1 1 0 2 3...] is equal to the (redox state of cof_i -1) (donated an electron)" and
                        "If the element that corresponds to cof_f in the state:[0 1 1 0 2 3...] is equal to the redox state of cof_f (accepted an electron)"
                        """
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(self.allow[i]), [cof_i_id, cof_f_id])
                        J = np.delete(self.idx2state(self.allow[j]), [cof_i_id, cof_f_id])
                        if np.array_equal(I, J):
                            # i and j state found!)
                            flux += self.K[j][i] * pop[i]      #K is rate matrix, so len(K)=self.num_state

        return flux

    def getCofactorFlux(self, cof_i: Cofactor, red_i: int, cof_f: Cofactor, red_f: int, pop: np.array) -> float:
        """
        Calculate the instantaneous NET flux from initial cofactor(state) to final cofactor(state), by calling getCofactorRate() twice
        Arguments:
            cof_i {Cofactor} -- Cofactor object for initial cofactor
            red_i {int} -- Redox state for initial cofactor before ET
            cof_f {Cofactor} -- Cofactor object for final cofactor
            red_f {int} -- Redox state for final cofactor after ET
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous net flux
        """
        return self.getCofactorRate(cof_i, red_i, cof_f, red_f, pop) - self.getCofactorRate(cof_f, red_f, cof_i, red_i, pop)

    def getNewReservoirFlux(self, name: str, pop: np.array) -> float:
        """
        Calculate the instantaneous net flux into the reservoir
        Arguments:
            reservoir_id {int} -- Reservoir ID
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous net flux connected to the reservoir
        """
        reservoir_id = self.reservoir2id[name]
        name, cofactor, redox_state, num_electron, deltaG, rate=self.reservoirInfo[reservoir_id]
        reverse_rate = rate * np.exp(self.beta*deltaG)
        final_redox=redox_state-num_electron      #redox_state is basically the initial redox state of the cofactor, which is info stored in reservoirInfo dict()

        cof_id = self.cofactor2id[cofactor]
        ans = 0
        for i in range(len(pop)):
            #Loop through all the possible states
            if self.idx2state(self.allow[i])[cof_id] == redox_state: # populated * rate out
                ans += pop[i] * rate
            if self.idx2state(self.allow[i])[cof_id] == final_redox and sum(self.idx2state(self.allow[i])) < self.max_electrons: # unpopulated * rate in (if permitted by max particle limit)
                ans -= pop[i] * reverse_rate
        return ans

    def getReservoirFlux(self, name: str, pop: np.array) -> float:
        """
        Calculate the instantaneous net flux into the reservoir connected to the reservoir
        Arguments:
            reservoir_id {int} -- Reservoir ID
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous net flux connected to the reservoir
        """
        reservoir_id = self.reservoir2id[name]
        name, cofactor, redox_state, num_electron, deltaG, rate=self.reservoirInfo[reservoir_id]
        reverse_rate = rate * np.exp(self.beta*deltaG)
        final_redox=redox_state-num_electron      #redox_state is basically the initial redox state of the cofactor, which is info stored in reservoirInfo dict()

        return (self.population(pop, cofactor, redox_state) * rate - self.population(pop, cofactor, final_redox) * reverse_rate) * num_electron

    def getProductFlux(self, H1: Cofactor, E3: Cofactor, pop: np.array):
        """
        Obtain and sum the fluxes of transitions (1, 0, 1) -> (0, 0, 0) and (1, 1, 1) -> (0, 1, 0)
        """
        cof_H1_id = self.cofactor2id[H1]
        cof_E3_id = self.cofactor2id[E3]

        productflux = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if self.idx2state(self.allow[i])[cof_H1_id] == 1 and self.idx2state(self.allow[i])[cof_E3_id] == 1:
                    if self.idx2state(self.allow[j])[cof_H1_id] == 0 and self.idx2state(self.allow[j])[cof_E3_id] == 0:
                        I = np.delete(self.idx2state(self.allow[i]), [cof_H1_id, cof_E3_id])
                        J = np.delete(self.idx2state(self.allow[j]), [cof_H1_id, cof_E3_id])
                        if np.array_equal(I, J):
                            #print("initial=", self.idx2state(self.allow[i]), "final=", self.idx2state(self.allow[j]))
                            productflux += self.K[j][i]*pop[i] - self.K[i][j]*pop[j]
                            #print("kf=", self.modK[j][i], "pop[i]=", pop[i], "kb=", self.modK[i][j], "pop[f]=", pop[j])
                            #productflux.append(self.modK[j][i]*pop[i] - self.modK[i][j]*pop[j])
        return productflux

    def getProductFlux_Mod(self, H1: Cofactor, E3: Cofactor, pop: np.array):
        """
        Obtain and sum the fluxes of transitions (1, 0, 1) -> (0, 0, 0) and (1, 1, 1) -> (0, 1, 0)
        """
        cof_H1_id = self.cofactor2id[H1]
        cof_E3_id = self.cofactor2id[E3]

        productflux = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if self.idx2state(self.allow[i])[cof_H1_id] == 1 and self.idx2state(self.allow[i])[cof_E3_id] == 1:
                    if self.idx2state(self.allow[j])[cof_H1_id] == 0 and self.idx2state(self.allow[j])[cof_E3_id] == 0:
                        I = np.delete(self.idx2state(self.allow[i]), [cof_H1_id, cof_E3_id])
                        J = np.delete(self.idx2state(self.allow[j]), [cof_H1_id, cof_E3_id])
                        if np.array_equal(I, J):
                            #print("initial=", self.idx2state(self.allow[i]), "final=", self.idx2state(self.allow[j]))
                            productflux += self.modK[j][i]*pop[i] - self.modK[i][j]*pop[j]
                            #print("kf=", self.modK[j][i], "pop[i]=", pop[i], "kb=", self.modK[i][j], "pop[f]=", pop[j])
                            #productflux.append(self.modK[j][i]*pop[i] - self.modK[i][j]*pop[j])
        return productflux

    def getPumpFlux(self, H2: Cofactor, pop: np.array):
        """
        Obtain and sum the fluxes of transitions (0, 1, 0) -> (0, 0, 0) and (1, 1, 0) -> (1, 0, 0) and (0, 1, 1) -> (0, 0, 1) and (1, 1, 1) -> (1, 0, 1)
        """
        cof_H2_id = self.cofactor2id[H2]

        pumpflux = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if self.idx2state(self.allow[i])[cof_H2_id] == 1 and self.idx2state(self.allow[j])[cof_H2_id] == 0:
                    I = np.delete(self.idx2state(self.allow[i]), [cof_H2_id])
                    J = np.delete(self.idx2state(self.allow[j]), [cof_H2_id])
                    if np.array_equal(I, J):
                        #print("initial=", self.idx2state(self.allow[i]), "final=", self.idx2state(self.allow[j]))
                        pumpflux += self.K[j][i]*pop[i] - self.K[i][j]*pop[j]
        return pumpflux

    def getPumpFlux_Mod(self, H2: Cofactor, pop: np.array):
        """
        Obtain and sum the fluxes of transitions (0, 1, 0) -> (0, 0, 0) and (1, 1, 0) -> (1, 0, 0) and (0, 1, 1) -> (0, 0, 1) and (1, 1, 1) -> (1, 0, 1)
        """
        cof_H2_id = self.cofactor2id[H2]

        pumpflux = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if self.idx2state(self.allow[i])[cof_H2_id] == 1 and self.idx2state(self.allow[j])[cof_H2_id] == 0:
                    I = np.delete(self.idx2state(self.allow[i]), [cof_H2_id])
                    J = np.delete(self.idx2state(self.allow[j]), [cof_H2_id])
                    if np.array_equal(I, J):
                        #print("initial=", self.idx2state(self.allow[i]), "final=", self.idx2state(self.allow[j]))
                        pumpflux += self.modK[j][i]*pop[i] - self.modK[i][j]*pop[j]
        return pumpflux

    def getElectronFlux(self, E3: Cofactor, pop: np.array):
        """
        Obtain and sum the fluxes of transitions (0, 0, 0) -> (0, 0, 1) and (1, 0, 0) -> (1, 0, 1) and (0, 1, 0) -> (0, 1, 1) and (1, 1, 0) -> (1, 1, 1)
        """
        cof_E3_id = self.cofactor2id[E3]

        electronflux = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if self.idx2state(self.allow[i])[cof_E3_id] == 0 and self.idx2state(self.allow[j])[cof_E3_id] == 1:
                    I = np.delete(self.idx2state(self.allow[i]), [cof_E3_id])
                    J = np.delete(self.idx2state(self.allow[j]), [cof_E3_id])
                    if np.array_equal(I, J):
                        #print("initial=", self.idx2state(self.allow[i]), "final=", self.idx2state(self.allow[j]))
                        electronflux += self.K[j][i]*pop[i] - self.K[i][j]*pop[j]
        return electronflux

    def getElectronFlux_Mod(self, E3: Cofactor, pop: np.array):
        """
        Obtain and sum the fluxes of transitions (0, 0, 0) -> (0, 0, 1) and (1, 0, 0) -> (1, 0, 1) and (0, 1, 0) -> (0, 1, 1) and (1, 1, 0) -> (1, 1, 1)
        """
        cof_E3_id = self.cofactor2id[E3]

        electronflux = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if self.idx2state(self.allow[i])[cof_E3_id] == 0 and self.idx2state(self.allow[j])[cof_E3_id] == 1:
                    I = np.delete(self.idx2state(self.allow[i]), [cof_E3_id])
                    J = np.delete(self.idx2state(self.allow[j]), [cof_E3_id])
                    if np.array_equal(I, J):
                        #print("initial=", self.idx2state(self.allow[i]), "final=", self.idx2state(self.allow[j]))
                        electronflux += self.modK[j][i]*pop[i] - self.modK[i][j]*pop[j]
        return electronflux

    def getUpFlux(self, H1: Cofactor, pop: np.array):
        """
        Obtain and sum the fluxes of transitions (0, 0, 0) -> (1, 0, 0) and (0, 1, 0) -> (1, 1, 0) and (0, 0, 1) -> (1, 0, 1) and (0, 1, 1) -> (1, 1, 1)
        """
        cof_H1_id = self.cofactor2id[H1]

        upflux = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if self.idx2state(self.allow[i])[cof_H1_id] == 0 and self.idx2state(self.allow[j])[cof_H1_id] == 1:
                    I = np.delete(self.idx2state(self.allow[i]), [cof_H1_id])
                    J = np.delete(self.idx2state(self.allow[j]), [cof_H1_id])
                    if np.array_equal(I, J):
                        #print("initial=", self.idx2state(self.allow[i]), "final=", self.idx2state(self.allow[j]))
                        upflux += self.K[j][i]*pop[i] - self.K[i][j]*pop[j]
        return upflux  

    def getUpFlux_Mod(self, H1: Cofactor, pop: np.array):
        """
        Obtain and sum the fluxes of transitions (0, 0, 0) -> (1, 0, 0) and (0, 1, 0) -> (1, 1, 0) and (0, 0, 1) -> (1, 0, 1) and (0, 1, 1) -> (1, 1, 1)
        """
        cof_H1_id = self.cofactor2id[H1]

        upflux = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                if self.idx2state(self.allow[i])[cof_H1_id] == 0 and self.idx2state(self.allow[j])[cof_H1_id] == 1:
                    I = np.delete(self.idx2state(self.allow[i]), [cof_H1_id])
                    J = np.delete(self.idx2state(self.allow[j]), [cof_H1_id])
                    if np.array_equal(I, J):
                        #print("initial=", self.idx2state(self.allow[i]), "final=", self.idx2state(self.allow[j]))
                        upflux += self.modK[j][i]*pop[i] - self.modK[i][j]*pop[j]
        return upflux    

    def reservoirFluxPlot(self, pop_init: np.array, t: float) -> list:
        """
        Calculate the net flux into a reservoir given its id versus time
        Arguments:
            t {float} -- Final time
            pop_init {np.array} -- Initial population vector (default: None)
            reservoir_id {int} -- Reservoir id
        Returns:
            list -- Net flux into the reservoir along a period of time
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        res_list=[]     #list of reservoir names
        for reservoir_id in range(self.num_reservoir):
            name, cofactor, redox_state, num_electron, deltaG, rate=self.reservoirInfo[reservoir_id]
            res_list.append(name)
        #print(res_list)
        for res in res_list:
            T = np.linspace(0, t, 1000)  # default spacing number: 1000
            fluxes=[]
            for time in T:
                pop=self.evolve(time, pop_init)
                flux=self.getReservoirFlux(res, pop)
                fluxes.append(flux)
            #print(fluxes)
            plt.plot(T, fluxes, label=res)

        plt.legend(loc="upper right")
        ax.set_xlabel('time (sec)',size='x-large')
        ax.set_ylabel('Flux',size='x-large')

    def popPlot(self, cof_list, pop_init: np.array, t: float) -> list:
        """
        Calculate the population of a given cofactor at specific redox state along a period of time
        Arguments:
            t {float} -- Final time
            pop_init {numpy.array} -- Initial population vector (default: None)
            cof_list {array} -- a list containing lists [cof, redox] where cof is a cofactor whose population is to be plotted, and redox is the desired redox state to be plotted
        Returns:
            list -- Population of the cofactor along a period of time
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for cof in cof_list:
            T = np.linspace(0, t, 100)  # default spacing number: 100
            A=[]
            pops=[] #population at site i
            for time in T:
                pop=self.evolve(time, pop_init)
                A.append(pop)
                pops.append(self.population(A[-1], cof[0], cof[1]))

            plt.plot(T, pops, label = cof[0].name + " (redox state = " + str(cof[1]) +")")

        plt.legend(loc="upper right")
        ax.set_xlabel('time (sec)',size='x-large')
        ax.set_ylabel('Probability',size='x-large')
        plt.show()

    def getExptvalue(self, pop: np.array, cofactor: Cofactor) -> float:
        """
        Calculate the expectation value of the number of particles at a given cofactor at a given time
        Arguments:
            cofactor {Cofactor} -- Cofactor object
            pop {cp.array} -- Population vector of the states
        """
        cof_id = self.cofactor2id[cofactor]
        expt=0
        #loop through all the possible states
        for i in range(self.adj_num_state):
            expt+=self.idx2state(self.allow[i])[cof_id]*pop[i]   #sum((number of particle)*(probability))

        return expt

    def popState(self, pop_init: np.array, t: float) -> list:
        """
        Visualize the population of the microstates at a given time
        Arguments:
            t {float} -- given time
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            list -- [[population, microstate that corresponds to that population]]
        """
        popstate=[]
        pop=self.evolve(t, pop_init)
        for i in range(self.adj_num_state):
            popstate.append([pop[i], self.idx2state(self.allow[i])])

        return popstate

    def getJointExptvalue(self, pop: np.array, cofactor_1: Cofactor, red_1: int, cofactor_2: Cofactor, red_2: int) -> float:
        """
        Calculate the joint probability of cofactor_1 being in redox state (red_1) and cofactor_2 being in redox state (red_2)
        Arguments:
            cofactor {Cofactor} -- Cofactor object
            pop {cp.array} -- Population vector of the states
        """
        cof1_id = self.cofactor2id[cofactor_1]
        cof2_id = self.cofactor2id[cofactor_2]
        expt=0
        for i in range(self.adj_num_state):
            if self.idx2state(self.allow[i])[cof1_id] == red_1 and self.idx2state(self.allow[i])[cof2_id] == red_2:
                expt += pop[i]
        return expt        
    
    def getMicroStateEnergyList(self):
        # List of microstate energies
        # [energy of 1st microstate, energy of 2nd microstate, ...]
        microstate_energy = []
        for i in range(self.adj_num_state):
            # Calculate energy of each microstate
            energy = 0
            for j in range(self.num_cofactor):
                cof = self.id2cofactor[j]
                if self.idx2state(self.allow[i])[j] == 0:
                    energy += 0
                if self.idx2state(self.allow[i])[j] == 1:
                    energy += 1*(-cof.redox[0])
                if self.idx2state(self.allow[i])[j] == 2:
                    energy += 1*(-cof.redox[0]) + 1*(-cof.redox[1])
                if self.idx2state(self.allow[i])[j] == 3:
                    energy += 1*(-cof.redox[0]) + 1*(-cof.redox[1]) + 1*(-cof.redox[2])
            microstate_energy.append(energy)

        return microstate_energy

    def getSlippage(self, cof1: Cofactor, cof2: Cofactor, pop: np.array) -> float:
        # cof1: H1, cof2: E3
        # There is slippage when H2O is produced before a proton is pumped
        # So the slippage microstate transitions are (1,0,1)->(0,0,0) and (1,1,1)->(0,1,0)
        slippage = 0
        for i in range(self.adj_num_state):
            for j in range(self.adj_num_state):
                cof1_id = self.cofactor2id[cof1]
                cof2_id = self.cofactor2id[cof2]
                if self.idx2state(self.allow[i])[cof1_id] == 1 and self.idx2state(self.allow[i])[cof2_id] == 1:
                    if self.idx2state(self.allow[j])[cof1_id] == 0 and self.idx2state(self.allow[j])[cof2_id] == 0:
                        I = np.delete(self.idx2state(self.allow[i]), [cof1_id, cof2_id])
                        J = np.delete(self.idx2state(self.allow[j]), [cof1_id, cof2_id])
                        if np.array_equal(I, J):
                            # print(self.idx2state(self.allow[i]), self.idx2state(self.allow[j]))
                            kf = self.K[j][i]
                            kb = self.K[i][j]
                            slippage += kf*pop[j]*(1-pop[i]) - kb*pop[i]*(1-pop[j])
                            #print(slippage)
        return slippage