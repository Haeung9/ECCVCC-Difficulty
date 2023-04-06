import unittest
import time
import numpy as np
from src import ldpc, parameters

class TestLDPC(unittest.TestCase):
    def test_LDPC_generateMatrix(self):
        instance = ldpc.LDPC()
        instance.row_deg = 4; instance.col_deg = 3
        instance.block_length = 16; instance.redundancy = 12; instance.message_length = 4
        instance.Make_Gallager_Parity_Check_Matrix(seed = int(time.time()))
        # print("\noriginal H: \n", instance.H)
        instance.Make_Parity_Check_Matrix_Sys()
        # print("systematic H: \n", instance.H_SYS)
        # print("systematic G: \n",instance.G_SYS)
    def test_isCodeword(self):
        instance = ldpc.LDPC()
        instance.row_deg = 4; instance.col_deg = 3
        instance.block_length = 16; instance.redundancy = 12; instance.message_length = 4
        instance.Make_Gallager_Parity_Check_Matrix(seed = int(time.time()))
        instance.output_word = np.zeros(shape=(instance.block_length), dtype=int)

        eval = instance.isCodeword()
        self.assertTrue(eval)

        instance.output_word[0] = 1
        eval = instance.isCodeword() # word with odd 1s is not a codeword
        self.assertFalse(eval)
    def test_LDPC_generateQ(self):
        instance = ldpc.LDPC()
        instance.row_deg = 4; instance.col_deg = 3
        instance.block_length = 16; instance.redundancy = 12; instance.message_length = 4
        instance.Make_Gallager_Parity_Check_Matrix(seed = int(time.time()))
        instance.generateQ()
        # print("H: \n", instance.H)
        # print("colInRow = \n", instance.col_in_row)
        # print("rowInCol = \n", instance.row_in_col)
    def test_LDPC_decoding(self):
        instance = ldpc.LDPC(blockLength=8)
        instance.block_length = 8; instance.redundancy = 6; instance.message_length = 2
        instance.row_deg = 4; instance.col_deg = 3
        oneMap = np.zeros(shape=(2, 24), dtype=int)
        oneMap[0,0] = 0; oneMap[1,0] = 0; oneMap[0,1] = 0; oneMap[1,1] = 1
        oneMap[0,2] = 0; oneMap[1,2] = 2; oneMap[0,3] = 0; oneMap[1,3] = 3 # row 0
        oneMap[0,4] = 1; oneMap[1,4] = 4; oneMap[0,5] = 1; oneMap[1,5] = 5
        oneMap[0,6] = 1; oneMap[1,6] = 6; oneMap[0,7] = 1; oneMap[1,7] = 7 # row 1
        oneMap[0,8] = 2; oneMap[1,8] = 0; oneMap[0,9] = 2; oneMap[1,9] = 1
        oneMap[0,10] = 2; oneMap[1,10] = 4; oneMap[0,11] = 2; oneMap[1,11] = 5 # row 2
        oneMap[0,12] = 3; oneMap[1,12] = 2; oneMap[0,13] = 3; oneMap[1,13] = 3
        oneMap[0,14] = 3; oneMap[1,14] = 6; oneMap[0,15] = 3; oneMap[1,15] = 7 # row 3
        oneMap[0,16] = 4; oneMap[1,16] = 0; oneMap[0,17] = 4; oneMap[1,17] = 4
        oneMap[0,18] = 4; oneMap[1,18] = 5; oneMap[0,19] = 4; oneMap[1,19] = 6 # row 4
        oneMap[0,20] = 5; oneMap[1,20] = 1; oneMap[0,21] = 5; oneMap[1,21] = 2
        oneMap[0,22] = 5; oneMap[1,22] = 3; oneMap[0,23] = 5; oneMap[1,23] = 7 # row 5
        for i in range(24):
            r = oneMap[0,i]
            c = oneMap[1,i]
            instance.H[r,c] = 1
        instance.generateQ()
        print("Reference matrix: ")
        print(instance.H)

        testVector = np.zeros(shape=(9,8), dtype=int)
        testVector[0,:] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        testVector[1,:] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        testVector[2,:] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        testVector[3,:] = np.array([0, 0, 1, 0, 0, 0, 0, 0])
        testVector[4,:] = np.array([0, 0, 0, 1, 0, 0, 0, 0])
        testVector[5,:] = np.array([0, 0, 0, 0, 1, 0, 0, 0])
        testVector[6,:] = np.array([0, 0, 0, 0, 0, 1, 0, 0])
        testVector[7,:] = np.array([0, 0, 0, 0, 0, 0, 1, 0])
        testVector[8,:] = np.array([0, 0, 0, 0, 0, 0, 0, 1])
   
        for i in range(9):
            instance.input_word = np.reshape(testVector[i,:], newshape=(instance.block_length)).copy()
            decodingFlag = instance.LDPC_Decoding(useOriginal=True)
            print("For test input: ", instance.input_word, ", output word: ", instance.output_word,", ", decodingFlag)

            




        
        

