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
        
        

