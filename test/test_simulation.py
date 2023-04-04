import unittest
import numpy as np
import time
from src import ldpc
from src import simulation
from src import parameters

class TestSimulation(unittest.TestCase):
    def test_weightCheck(self):
        param = parameters.codeParameters()
        word = np.zeros(shape=(param.block_length), dtype=int)
        [eval, _] = simulation.weightCheck(word, param)
        self.assertTrue(eval) # zero word is always a codeword

        param.hammingWeigthLow = 2
        [eval, _] = simulation.weightCheck(word, param)
        self.assertFalse(eval)

    def test_runSingleSimulation_zeroWord(self):
        verbose = False
        param = parameters.codeParameters() # default parameters
        (sol, decodingSuccess, _) = simulation.runSingleSimulation(param, word = np.zeros(shape=(param.block_length), dtype=int),useWordInput=True , verbose=verbose)
        self.assertTrue(decodingSuccess) # zero-word always a codeword
        self.assertTrue(sol)
        print("is solution found? : ", sol, ", is decoding success? : ", decodingSuccess)
    @unittest.skip("")
    def test_runSingleSimulation_singleErrorWord(self):
        verbose = False
        param = parameters.codeParameters(blockLength= 32, rowDegree=4, colDegree=3, hammingWeigtHigh=32) 
        # word = np.zeros(shape=(param.block_length), dtype=int)
        onepart = np.ones(shape=(param.row_deg), dtype=int)
        zeropart = np.zeros(shape=(param.block_length - param.row_deg), dtype=int)
        word = np.concatenate((onepart, zeropart))
        word[-1] = (word[-1] + 1) % 2
        (sol, decodingSuccess, _) = simulation.runSingleSimulation(param, word = word, useWordInput=True, verbose=verbose)
        self.assertTrue(decodingSuccess) # single error should be corrected
        self.assertTrue(sol)
        print("is solution found? : ", sol, ", is decoding success? : ", decodingSuccess)
    
    @unittest.skip("")
    def test_runSingleSimulation_randomWord(self):
        # For randomword
        verbose = True
        param = parameters.codeParameters() # default parameters
        (sol, decodingSuccess, _) = simulation.runSingleSimulation(param, verbose=verbose)
        print("is solution found? : ", sol, ", is decoding success? : ", decodingSuccess)


    def test_runMonteCarlo(self):
        verbose = False
        param = parameters.codeParameters(blockLength=32, rowDegree=4, colDegree=3)
        (errorRate, decodingRate)= simulation.runMonteCarlo(param, saveResult = False, numSim=1000,verbose=verbose)
        print("Error rate = ", errorRate)
        print("Decoding rate = ", decodingRate)

