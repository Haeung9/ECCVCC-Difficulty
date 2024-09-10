import logging
import math
import time
import numpy as np
import copy

from . import constants, utils, ldpc

class SILDPC(ldpc.LDPC):
    def __init__(self, blockLength = 16, rowDegree = 4, colDegree = 3, seed = 0) -> None:
        super().__init__(blockLength, rowDegree, colDegree, seed)
        self.syndrome = np.zeros(shape=(self.redundancy), dtype=int)
    def LDPC_Decoding(self) -> bool:
        self.Make_Parity_Check_Matrix_Sys()
        self.generateQ()
        self.decodingInitialize()
        # matrix = copy.deepcopy(self.H)
        # (swapmapRow, swapmapCol, lastnonzerow) = utils.computeBinaryRREF(matrix)
        # self.informativeRows = swapmapRow[:lastnonzerow+1]
        return self.SILDPC_Decoding()
    def isParitySatisfied(self) -> bool:
        for i in range(0, self.redundancy):
            if not np.isin(self.dependentParityRows, i).any():
                sum = 0
                for j in range(0, self.row_deg):
                    sum = sum + self.output_word[self.col_in_row[j][i]]
                if not sum % 2 == self.syndrome[i]:
                    return False
        return True

    def SILDPC_Decoding(self) -> bool:
    # iterative decoder for nonzero check condition (i.e., syndrome decoding)
        self.output_word = np.zeros(shape=(self.block_length), dtype=int) # initialized by zero
        if self.isParitySatisfied():
            return True
        self.LRft = np.ones(shape=(self.block_length), dtype=float)*math.log((1.0 - constants.cross_over_probability)/constants.cross_over_probability)
        self.LRpt = np.zeros(shape=(self.block_length), dtype=float)

        logging.debug("prior = " + np.array2string(self.LRft))
        logging.debug("for inputword = " + np.array2string(self.input_word))
        logging.debug("for syndrome = " + np.array2string(self.syndrome))
        logging.debug("Decoding starts")
        iterstart = time.time()
        for index in range(1, constants.ITERATIONS + 1):
            logging.debug("For iteration %d:\n", index)
            V2Cstart = time.time()
            # Bit to Check Node Messages --> LRqtl
            for t in range(self.block_length):
                C2VmessageSum = 0.0
                connectedChecks = self.row_in_col[:,t] # List of C nodes who are connected with V(t) node 
                C2VmessageToVt = self.LRrtl[t, :] #
                for mp in range(self.col_deg): # Sum of all C2V forwarded to V(t)
                    C2VmessageSum = utils.infinityTest(C2VmessageSum + C2VmessageToVt[connectedChecks[mp]]) 
                for m in range(self.col_deg):
                    C2VmessageSelf = self.LRrtl[t, connectedChecks[m]]
                    C2VmessageSumExceptSelf = utils.infinityTest(C2VmessageSum - C2VmessageSelf)
                    self.LRqtl[t, connectedChecks[m]] = utils.infinityTest(self.LRft[t] + C2VmessageSumExceptSelf)
            
            # Check to Bit Node Messages --> LRrtl
            logging.debug("V2C computed (%f s).", time.time()-V2Cstart) 
            logging.debug("V2C = \n" + np.array2string(self.LRqtl))   
            C2Vstart = time.time()  
            # ######################################## k
            for k in range(self.redundancy):
                if not np.isin(self.dependentParityRows, k).any():
                # if np.isin(self.informativeRows, k).any():
                    connectedVars = self.col_in_row[:,k] # List of V nodes who are connected with C(k) node
                    for l in range(self.row_deg):
                        V2CmessageLogSum = 0.0
                        sign = (self.syndrome[k] * (-2)) + 1
                        for m in range(self.row_deg):
                            if not m == l: # log sum of V2C messages from all connected V nodes to C(k), except V2C from V(connectedVars(l))
                                V2CmessageLogSum += utils.func_f( abs(self.LRqtl.item((connectedVars[m], k))) )
                                if not self.LRqtl.item((connectedVars[m],k)) > 0.0: # determine the sign: Odd number of negative value -> negative
                                    sign = sign * (-1.0)
                        magnitude = utils.func_f(V2CmessageLogSum) # exp compensation
                    # C2V message from C(k) to V(connectedVars(l)) is the product of V2C messages, except V2C from V(connectedVars(l))
                        self.LRrtl[connectedVars[l],k] = utils.infinityTest(sign*magnitude)
            
            # Last iteration get LR (pi)
            # LRpt => L_j^Total, LRft = L_j, LRrtl = L_i->
            logging.debug("C2V computed (%f s).", time.time()-C2Vstart)
            logging.debug("C2V = \n" + np.array2string(self.LRrtl))
            poststart = time.time()
            for t in range(self.block_length):
                self.LRpt[t] = utils.infinityTest(self.LRft[t])
                for k in range(self.col_deg):
                    self.LRpt[t] += self.LRrtl[t, self.row_in_col.item((k, t))]
                    self.LRpt[t] = utils.infinityTest(self.LRpt[t])
            logging.debug("posterior computed (%f s).", time.time()-poststart)
            logging.debug("posterior = \n" + np.array2string(self.LRpt))
            # Decision
            for i in range(self.block_length):
                if self.LRpt[i] < 0.0:
                    self.output_word[i] = 1
                else:
                    self.output_word[i] = 0
            logging.debug("outputword at %d-th iteration = " + np.array2string(self.output_word), index)
            # if self.isCodeword():
            #     self.Iter = index
            #     logging.info("Decoding finished at %d-th iteration", index)
            #     return True
            # logging.debug("Iteration %d ends (%f s).", index, time.time()-iterstart)
            if self.isParitySatisfied():
                logging.info("SI Decoding success")
                return True
        logging.info("Decoder reached maximum iteration")
        return False