import numpy as np
import os
import time
import logging
from typing import Tuple

from . import ldpc
from . import syndromeInputLdpc
from . import parameters

def runSingleSimulation(codeParams: parameters.codeParameters, word = np.zeros(shape=(32), dtype=int), useWordInput = False) -> Tuple[bool, bool, int]:
    blockLength = codeParams.block_length
    syndromeLength = codeParams.redundancy
    rowDegree = codeParams.row_deg
    colDegree = codeParams.col_deg
    randomWord = np.random.randint(np.zeros(shape=(blockLength),dtype=int),2*np.ones(shape=(blockLength), dtype=int))
    if useWordInput:
        randomWord = word
    seed = np.random.randint(1000) + int(time.time())
    instance = ldpc.LDPC(blockLength, rowDegree, colDegree)
    if not instance.Make_Gallager_Parity_Check_Matrix(seed):
        raise Exception("Cannot generate PCM")
    logging.debug("PCM generated." + np.array2string(instance.H))
    if not instance.generateQ():
        raise Exception("Cannot generate sparse of PCM")
    logging.debug("Sparse generated. \n" + np.array2string(instance.row_in_col) + "\n" + np.array2string(instance.col_in_row))
    
    instance.input_word = randomWord
    logging.debug("Decoding ...")
    decodingSuccess = instance.LDPC_Decoding()
    sol = False
    logging.debug("Decoding ends")
    if decodingSuccess:
        (weightCheckPass, foundHammingWeight) = weightCheck(instance.output_word, codeParams)
        if weightCheckPass:
            logging.info("Success!")
            sol = True
    else:
        foundHammingWeight = 0
    snapshotPCM = instance.H.copy()
    snapshotInputWord = instance.input_word.copy()
    snapshotOutputWord = instance.output_word.copy()
    snapshotSol = sol
    snapshotDS = decodingSuccess
    snapshotWeight = foundHammingWeight
    logging.info("PCM: \n" + np.array2string(snapshotPCM))
    logging.info("input word: " + np.array2string(snapshotInputWord))
    logging.info("output word: " + np.array2string(snapshotOutputWord))
    logging.info("decision condition: " + str(codeParams.hammingWeigthLow) + " <= HW <= " + str(codeParams.hammingWeigthHigh))
    logging.info("solution found: " + str(snapshotSol))
    logging.info("decoder flag: " + str(snapshotDS))
    logging.info("Hamming weight of found codeword: %d",foundHammingWeight)
    return (sol, decodingSuccess, foundHammingWeight)

def weightCheck(word: np.ndarray, codeParams: parameters.codeParameters) -> Tuple[bool, int]:
    hammingWeigth = np.count_nonzero(word)
    atStep = (hammingWeigth % codeParams.decisionStep == 0)
    if (hammingWeigth >= codeParams.hammingWeigthLow) and (hammingWeigth <= codeParams.hammingWeigthHigh) and atStep:
        return [True, hammingWeigth]
    else:
        return [False, hammingWeigth]
    
def runMonteCarlo(codeParams: parameters.codeParameters, numSim = 100, saveResult = True, dir = os.getcwd(), printStamp = False) -> Tuple[int, int]:
    solutionFound = 0
    decoderSuccess = 0
    hammingWeigthHistogram = np.zeros(shape=(codeParams.block_length + 1), dtype=int)  
    startTime = time.time()
    for cnt_sim in range(numSim):
        (sol, ds, hw) = runSingleSimulation(codeParams)
        if ds:
            decoderSuccess += 1
            hammingWeigthHistogram[hw] += 1
        if sol:
            solutionFound += 1
        if printStamp and (cnt_sim%2000 == 0):
            print(cnt_sim, "th simulation...(", time.time() - startTime, "s)")
    # solutionRate = float(solutionFound)/float(numSim)
    # decodingRate = float(decoderSuccess)/float(numSim)
    if saveResult:
        simInfo = np.array([solutionFound, decoderSuccess, numSim], dtype=int)
        result = np.concatenate([simInfo, hammingWeigthHistogram], axis=0)
        fileName = os.path.join(dir, "result_compare.csv")
        result.tofile(fileName, ",")
        print("result is saved in: ", dir)
    return (solutionFound, decoderSuccess)

