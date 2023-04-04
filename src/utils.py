import numpy as np
import math
from typing import Tuple
from . import constants

def computeBinaryRREF(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    if matrix.dtype==int and ((matrix==0)|(matrix==1)).all():
        (redundancy, block_length) = matrix.shape
        lastnonzerow = redundancy
        swapmapRow = np.arange(0,redundancy)
        swapmapCol = np.arange(0,block_length)

        for pivot in range(0,redundancy):
            (target_row, target_col, targetFound) = FindGoodRowCol(matrix, pivot) # find swap target for nonzero pivot
            if not targetFound:
                lastnonzerow = pivot # mark as lastnonzerow if no anymore 1s
                break
            swapmapRow[pivot], swapmapRow[target_row] = swapmapRow[target_row], swapmapRow[pivot]
            swapmapCol[pivot], swapmapCol[target_col] = swapmapCol[target_col], swapmapCol[pivot]
            tempRow = matrix[pivot, :]; matrix[pivot, :] = matrix[target_row, :]; matrix[target_row, :] = tempRow # row swap
            tempCol = matrix[:, pivot]; matrix[:, pivot] = matrix[:, target_col]; matrix[:, target_col] = tempCol # column swap

            if pivot + 1 < redundancy:
                for row in range(pivot + 1,redundancy):
                    if matrix[row,pivot]==1:
                        matrix[row, :] = matrix[row, :]^matrix[pivot, :] # gen REF by forward elimination

        for col in range(lastnonzerow - 1, 0, -1):
            for row in range(0, col):
                if matrix[row, col] == 1:
                    matrix[row, :] =  matrix[row, :]^matrix[col,:] # gen RREF by backward elimination
    else:
        print("computeBinaryRREF: input is not a binary matrix.")
    return (swapmapRow, swapmapCol, lastnonzerow)
   
def FindGoodRowCol(matrix: np.ndarray, pivot: int) -> Tuple[int|None, int|None, bool]:
    for j in range(pivot, matrix.shape[1]):
        for i in range(pivot, matrix.shape[0]):
            if matrix[i,j] == 1:
                target_row = i; target_col = j
                return(target_row, target_col, True) 
    return (pivot, pivot, False)

def infinityTest(number: int|float) -> float:
    number = float(number)
    if number > constants.INFINITY:
        # print("clipped by infinityTest")
        return constants.INFINITY
    elif number < - constants.INFINITY:
        # print("clipped by infinityTest")
        return (- constants.INFINITY)
    else:
        return number

def func_f(number: int|float) -> float:
    input = float(number)
    if input >= constants.BIG_INFINITY:
        # print("clipped by func_f (high). Value = ", input)
        return float(1.0 / constants.BIG_INFINITY)
    elif input <= float(1.0 / constants.BIG_INFINITY):
        # print("clipped by func_f (low). Value = ", input)
        return float(constants.BIG_INFINITY)
    else:
        return math.log( (math.exp(input) + 1)/(math.exp(input) - 1) )
