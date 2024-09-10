import pandas as pd
import numpy as np
import time
from src import ldpc, parameters
from . import network

class Worker:
    def __init__(self, workerId: int, maxTrials = 10):
        self.id = workerId
        self.maxTrials = maxTrials
    def solveProblem(self, problemId: int, problem: pd.DataFrame, param: parameters.codeParameters) -> dict:
        problemSeed = problem.loc[problemId]["seed"]
        myProblem = problemSeed + self.id + int(time.time())
        rng = np.random.default_rng(myProblem)
        instance = ldpc.LDPC(param.block_length, param.row_deg, param.col_deg, seed= problemSeed)
        instance.Make_Gallager_Parity_Check_Matrix(problemSeed) # gen problem. Input should be problemSeed, instead of myProblem
        for i in range(self.maxTrials):
            instance.decodingInitialize()
            inputWord = rng.integers(low=0, high=2, size=(param.block_length, ))
            instance.input_word = inputWord
            result = instance.LDPC_Decoding() # solve problem
            outputWord = instance.output_word
            if result:
                break
        answer = {"problem": problemId, "worker": self.id, "input": inputWord, "output": outputWord, "result": result, "trials": i + 1}
        return answer

