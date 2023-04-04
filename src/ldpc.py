import numpy as np
import math
from typing import Tuple
from . import constants
from . import utils

import time # for estimate runtime

class LDPC:
    def __init__(self, blockLength = 16, rowDegree = 4, colDegree = 3, seed = 0 ):
        self.block_length = blockLength; self.row_deg = rowDegree; self.col_deg = colDegree
        self.redundancy = int (self.block_length * self.col_deg / self.row_deg)
        if not all(self.dimensionCheck()):
            raise ValueError("Invalied code parameters")
        self.message_length = self.block_length - self.redundancy
        self.code_rate = float(self.col_deg) / float(self.row_deg)
        self.seed = seed; 
        # self.q = 0
        self.lastnonzerow = 0; self.Iter = 0
        self.H = np.zeros(shape=(self.redundancy, self.block_length), dtype=int )
        self.H_ORI = None; self.H_SYS = None; self.H_NEW = None; self.G_SYS = None
        self.row_in_col = np.zeros(shape=(self.col_deg, self.block_length), dtype=int) # row_in_col : col_deg * block_lenegth
        self.col_in_row = np.zeros(shape=(self.row_deg, self.redundancy), dtype=int) # col_in_row : row_deg * redundancy						
        self.LRft = np.zeros(shape=(self.block_length), dtype=float) # prior LR
        self.LRpt = np.zeros(shape=(self.block_length), dtype=float) # posterior LR
        self.LRrtl = np.zeros(shape=(self.block_length, self.redundancy), dtype=float) # C2V message
        self.LRqtl = np.zeros(shape=(self.block_length, self.redundancy), dtype=float) # V2C message
        self.input_word = np.zeros(shape=(self.block_length), dtype=int) 
        self.output_word = np.zeros(shape=(self.block_length), dtype=int) 
    
    def dimensionCheck(self) -> Tuple[bool, bool]:
        rateCheck = (self.block_length * self.col_deg == self.redundancy * self.row_deg)
        rowDegreeCheck = (self.block_length % self.row_deg == 0)
        return (rateCheck, rowDegreeCheck)
    
    def Make_Gallager_Parity_Check_Matrix(self, seed: int) -> bool:
        self.seed = seed
        
        if not self.dimensionCheck:
            print("Make_Gallager_Parity_Check_Matrix: invalid dimension.")
            return False
        
        self.H = np.zeros(shape=(self.redundancy, self.block_length),dtype=int) 
        H0 = np.zeros(shape =(int(self.redundancy/self.col_deg), self.block_length), dtype=int)

        for i in range(0, int(self.redundancy/self.col_deg)): # generate H(0)
            for j in range(i*self.row_deg, (i+1)*self.row_deg):
                H0[i,j] = 1
        self.H = H0

        for i in range(1,self.col_deg): # generate H(1), H(2), ..., H(col_deg - 1), who are column-permutations of H(0) 
            rng = np.random.default_rng(seed + i - 1)
            col_order = rng.permutation(self.block_length)
            Hperm = H0[:, col_order]
            self.H = np.concatenate([self.H, Hperm], axis=0)

        return True
    
    def generateQ(self) -> bool:
        self.col_in_row = np.zeros(shape = (self.row_deg, self.redundancy), dtype=int )
        self.row_in_col = np.zeros(shape = (self.col_deg, self.block_length), dtype=int )
        colInd = 0; rowInd = 0
        for i in range(self.redundancy):
            for j in range(self.block_length):
                if self.H[i,j] == 1 :
                    self.col_in_row[colInd % self.row_deg, i] = j; colInd += 1
                    self.row_in_col[int(rowInd / self.block_length), j] = i; rowInd += 1
        return True
    
    def isCodeword(self):
        syndrome = np.matmul(self.H, self.output_word.reshape(self.block_length,1)) % 2
        if (syndrome==0).all():
            return True
        else:
            return False
        
    def LDPC_Decoding(self, verbose = False, useOriginal = False) -> bool:
        if useOriginal:
            return self.LDPC_Decoding_Original(verbose=verbose)
        else:
            return self.LDPC_Decoding_ETHECC(verbose=verbose)

    # unused                        
    def Make_Parity_Check_Matrix_Sys(self) -> bool:
        self.H_SYS = self.H.copy()
        (swapmapRow, swapmapCol, lastnonzerow) = utils.computeBinaryRREF(self.H_SYS)

        self.H_SYS = self.H_SYS[0:lastnonzerow, :]; swapmapRow = swapmapRow[0:lastnonzerow] # drop zero-rows
        self.H_SYS = np.concatenate([ self.H_SYS[:,lastnonzerow:], self.H_SYS[:,0:lastnonzerow] ], axis=1);  # swap eye part and remainders 
        swapmapCol = np.concatenate([ swapmapCol[lastnonzerow:], swapmapCol[0:lastnonzerow] ])

        parity_length = self.block_length - lastnonzerow
        syndrome_length = lastnonzerow
        self.G_SYS = np.concatenate([np.eye(parity_length, dtype=int), self.H_SYS[:,0:parity_length].transpose()],axis=1)
        return True
    
    # implementation of ETH-ECC; only explore C2V for k in range(self.row_deg)
    def LDPC_Decoding_ETHECC(self, verbose = False) -> bool:
        self.output_word = self.input_word.copy()
        if self.isCodeword():
            return True
        self.LRft = math.log((1.0 - constants.cross_over_probability)/(constants.cross_over_probability))*(self.input_word.astype(float) * 2 - 1)
        self.LRpt = np.zeros(shape=(self.block_length), dtype=float)
        if verbose:
            print("prior = ", self.LRft)
            print("for inputword = ", self.input_word)
            print("Decoding starts")
            iterstart = time.time()
        for index in range(1, constants.ITERATIONS + 1):
            if verbose: 
                print("For iteration ", index, ":\n")
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
            if verbose: 
                print("V2C computed (", time.time()-V2Cstart, " s)."); C2Vstart = time.time()   
                print("V2C = ", self.LRqtl)   
            # ######################################## k
            for k in range(self.row_deg):
                connectedVars = self.col_in_row[:,k] # List of V nodes who are connected with C(k) node
                for l in range(self.row_deg):
                    V2CmessageLogSum = 0.0
                    sign = 1.0
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
            if verbose:
                print("C2V computed (", time.time()-C2Vstart, " s)."); poststart = time.time()
                print("C2V = ", self.LRrtl)
            for t in range(self.block_length):
                self.LRpt[t] = utils.infinityTest(self.LRft[t])
                connectedChecks = self.row_in_col[:, t]
                for k in range(self.col_deg):
                    self.LRpt[t] += self.LRrtl[t, connectedChecks[k]]
                    self.LRpt[t] = utils.infinityTest(self.LRpt[t])
            if verbose: 
                print("posterior computed (", time.time()-poststart, " s).")
                print("posterior = ", self.LRpt)
            # Decision
            for i in range(self.block_length):
                if self.LRpt[i] >= 0.0:
                    self.output_word[i] = 1
                else:
                    self.output_word[i] = 0
            if verbose: print("outputword at ", index, "-th iteration = ", self.output_word)
            if self.isCodeword():
                self.Iter = index
                if verbose: print("output word is a codeword.")
                return True
            if verbose: print("Iteration ", index, " ends (", time.time()-iterstart, " s).")
        return False
    
    # original implementation in cpp; iterate all C2V, out of row_deg bound
    def LDPC_Decoding_Original(self, verbose = False) -> bool:
        self.output_word = self.input_word.copy()
        if self.isCodeword():
            return True
        self.LRft = math.log((1.0 - constants.cross_over_probability)/(constants.cross_over_probability))*(self.input_word.astype(float) * 2 - 1)
        self.LRpt = np.zeros(shape=(self.block_length), dtype=float)
        if verbose:
            print("prior = ", self.LRft)
            print("for inputword = ", self.input_word)
            print("Decoding starts")
            iterstart = time.time()
        for index in range(1, constants.ITERATIONS + 1):
            if verbose: 
                print("For iteration ", index, ":\n")
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
            if verbose: 
                print("V2C computed (", time.time()-V2Cstart, " s)."); C2Vstart = time.time()   
                print("V2C = ", self.LRqtl)   
            # ######################################## k
            for k in range(self.redundancy):
                connectedVars = self.col_in_row[:,k] # List of V nodes who are connected with C(k) node
                for l in range(self.row_deg):
                    V2CmessageLogSum = 0.0
                    sign = 1.0
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
            if verbose:
                print("C2V computed (", time.time()-C2Vstart, " s)."); poststart = time.time()
                print("C2V = ", self.LRrtl)
            for t in range(self.block_length):
                self.LRpt[t] = utils.infinityTest(self.LRft[t])
                for k in range(self.col_deg):
                    self.LRpt[t] += self.LRrtl[t, self.row_in_col.item((k, t))]
                    self.LRpt[t] = utils.infinityTest(self.LRpt[t])
            if verbose: 
                print("posterior computed (", time.time()-poststart, " s).")
                print("posterior = ", self.LRpt)
            # Decision
            for i in range(self.block_length):
                if self.LRpt[i] >= 0.0:
                    self.output_word[i] = 1
                else:
                    self.output_word[i] = 0
            if verbose: print("outputword at ", index, "-th iteration = ", self.output_word)
            if self.isCodeword():
                self.Iter = index
                if verbose: print("output word is a codeword.")
                return True
            if verbose: print("Iteration ", index, " ends (", time.time()-iterstart, " s).")
        return False