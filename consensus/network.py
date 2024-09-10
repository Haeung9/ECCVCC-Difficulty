import numpy as np
import pandas as pd
import random
import time

from src import parameters
from . import agent

class Network:
    def __init__(self, networkId: int, parameters = parameters.codeParameters(), peers = pd.DataFrame(columns = ["power"])):
        self.networkId = networkId
        self.peers = peers
        self.parameters = parameters
        self.problems = pd.DataFrame(columns = ["seed", "isSolved"])
        self.results = pd.DataFrame(columns = ["winner", "elapsed"])
        self.solutions = []
    def generateNewProblem(self):
        problemId = len(self.problems)
        problemSeed = random.randint(0, 1000000)
        newProblem = pd.DataFrame(data = {"seed": [problemSeed], "isSolved": [False]}, index=[problemId])
        if problemId == 0:
            self.problems = newProblem
        else:
            self.problems = pd.concat([self.problems, newProblem], axis=0) # TODO: except duplicated ids
    def mine(self, blockNumber: int, verbose = False) -> bool:
        if self.problems.loc[blockNumber, "isSolved"] == True:
            print("error: block is alread mined")
            return False
        minElapsed = 99999999
        winner = -1
        solution = {}
        start = time.time()
        for workers in range(len(self.peers)):
            workerId = self.peers.index[workers]
            workerPower = self.peers.loc[workerId, "power"]
            worker = agent.Worker(workerId, maxTrials= 100)
            answer = worker.solveProblem(problemId=blockNumber, problem=self.problems, param=self.parameters)
            if verbose:
                solved = "solved" if answer["result"] else "could'n solve"
                print("For block #", blockNumber, ", worker #", workerId, ", ", solved, "in", answer["trials"], "trials. (", time.time()-start, "s)")
                start = time.time()
            if answer["result"] == True:
                elapsed = answer["trials"] / workerPower
                if elapsed < minElapsed:
                    winner = workerId
                    minElapsed = elapsed
                    solution = answer
        result = pd.DataFrame(data= {"winner": [winner], "elapsed": [minElapsed]}, index= [blockNumber])
        if blockNumber == 0:
            self.results = result
        else:
            self.results = pd.concat([self.results, result], axis=0)
        self.solutions.append(solution)
        return True





        
