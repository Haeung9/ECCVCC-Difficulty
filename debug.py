import os
import time
import numpy as np
from src import parameters, ldpc, syndromeInputLdpc

import logging

def main():
    logLevel = logging.DEBUG
    useOriginal= True
    blockLength = 16
    colDegree = 3
    rowDegree = 4 
    hammingWeigthLow = 0
    hammingWeigtHigh = 16
    decisionStep = 2

    dir = os.path.join(os.getcwd(), "data")
    directoryMaker(dir)
    logdir = os.path.join(os.getcwd(), "log")
    directoryMaker(logdir)
    logfile = os.path.join(logdir, "debug.log")
    logging.basicConfig(filename=logfile, format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logLevel)

    # simSetting = np.array([blockLength, colDegree, rowDegree, hammingWeigthLow, hammingWeigtHigh, decisionStep, numSim], dtype=int)
    # simSetting.tofile(os.path.join(dir, "settings.csv"), sep=",")

    param = parameters.codeParameters(blockLength, rowDegree, colDegree, hammingWeigthLow, hammingWeigtHigh, decisionStep)
    instance = ldpc.LDPC(param.block_length, param.row_deg, param.col_deg)
    instance.H = loadReferenceMatrix()
    logging.debug("Reference PCM: \n" + np.array2string(instance.H))
    instance.generateQ()
    logging.debug("Sparse generated. \n" + np.array2string(instance.row_in_col) + "\n" + np.array2string(instance.col_in_row))
    
    instance.input_word = loadReferenceInput()
    logging.debug("Reference input: " + np.array2string(instance.input_word))
    logging.debug("---------- zero-check decoding ----------")
    print("---------- zero-check decoding ----------")
    decodingFlag_regular = instance.LDPC_Decoding(useOriginal)
    print("inputword (imple): ", instance.input_word)
    print("outputword (imple): ", instance.output_word)

    logging.debug("---------- nonzero-check decoding ----------")
    print("---------- nonzero-check decoding ----------")
    syndromeInstance = syndromeInputLdpc.SILDPC(param.block_length, param.row_deg, param.col_deg)
    syndromeInstance.H = loadReferenceMatrix()
    syndromeInstance.generateQ()
    syndromeInstance.syndrome = loadReferenceSyndrome()
    decodingFlag_syndrome = syndromeInstance.LDPC_Decoding()
    outputSyndrome = np.matmul(syndromeInstance.H, syndromeInstance.output_word.reshape(param.block_length)) % 2
    outputSyndrome = outputSyndrome.astype(int)
    outputWord = (syndromeInstance.output_word + instance.input_word) % 2
    print("input syndrome: ", syndromeInstance.syndrome)
    print("output syndrome: ", outputSyndrome)
    print("output error: ", syndromeInstance.output_word)
    print("output word: ", outputWord)

def findInequalResult():
    logLevel = logging.INFO
    useOriginal= True
    blockLength = 16
    colDegree = 3
    rowDegree = 4 
    hammingWeigthLow = 0
    hammingWeigtHigh = 16
    decisionStep = 2
    numSim = 100000
    foundFlag = False

    dir = os.path.join(os.getcwd(), "data")
    directoryMaker(dir)
    logdir = os.path.join(os.getcwd(), "log")
    directoryMaker(logdir)
    logfile = os.path.join(logdir, "findinequal.log")
    logging.basicConfig(filename=logfile, format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logLevel)

    # simSetting = np.array([blockLength, colDegree, rowDegree, hammingWeigthLow, hammingWeigtHigh, decisionStep, numSim], dtype=int)
    # simSetting.tofile(os.path.join(dir, "settings.csv"), sep=",")
    startTime = time.time()
    for i in range(numSim):
        seed = np.random.randint(1000) + int(time.time())
        param = parameters.codeParameters(blockLength, rowDegree, colDegree, hammingWeigthLow, hammingWeigtHigh, decisionStep)
        instance = ldpc.LDPC(param.block_length, param.row_deg, param.col_deg)
        instance.Make_Gallager_Parity_Check_Matrix(seed) 
        instance.input_word = np.random.randint(np.zeros(shape=(blockLength),dtype=int),2*np.ones(shape=(blockLength), dtype=int))
        decodingFlag_regular = instance.LDPC_Decoding(useOriginal)
    # print("inputword (imple): ", instance.input_word)
    # print("outputword (imple): ", instance.output_word)

        syndromeInstance = syndromeInputLdpc.SILDPC(param.block_length, param.row_deg, param.col_deg)
        syndromeInstance.H = instance.H.copy()
        syndromeInstance.generateQ()

        syndrome = np.matmul(syndromeInstance.H, instance.input_word.reshape(param.block_length)) % 2
        syndrome = syndrome.astype(int)
        syndromeInstance.syndrome = syndrome
        decodingFlag_syndrome = syndromeInstance.LDPC_Decoding()
        outputSyndrome = np.matmul(syndromeInstance.H, syndromeInstance.output_word.reshape(param.block_length)) % 2
        outputSyndrome = outputSyndrome.astype(int)
        outputWord = (syndromeInstance.output_word + instance.input_word) % 2
    # print("input syndrome: ", syndromeInstance.syndrome)
    # print("output syndrome: ", outputSyndrome)
    # print("output error: ", syndromeInstance.output_word)
    # print("output word: ", outputWord)
        # if not (outputWord == instance.output_word).all():
        if not (decodingFlag_syndrome == decodingFlag_regular):
            logging.info("Inequal result found (" + str(i) + "th iteration).")
            logging.info("seed = " + str(seed))
            logging.info("Reference PCM: \n" + np.array2string(instance.H))
            logging.info("Reference input: " + np.array2string(instance.input_word))
            logging.info("Reference syndrome: " + np.array2string(syndromeInstance.syndrome))
            logging.info("Regular decoding success? " + str(decodingFlag_regular))
            logging.info("Regular decoding output: " + np.array2string(instance.output_word))
            logging.info("Syndrome decoding success? " + str(decodingFlag_syndrome))
            logging.info("Syndrome decoding ouput: " + np.array2string(outputWord))
            foundFlag = True
            break
        if (i % 100) == 0:
            print("Iteration ", i, ", elape = ", time.time()-startTime)
    if not foundFlag:
        print("Cannot found inequal result")


def loadReferenceMatrix():
    H = np.zeros(shape=(12,16))
    H[0,:] = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=int) # 0
    H[1,:] = np.array([0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0], dtype=int) # 1 
    H[2,:] = np.array([0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0], dtype=int) # 1
    H[3,:] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], dtype=int) # 1
    H[4,:] = np.array([1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1], dtype=int) # 0
    H[5,:] = np.array([0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0], dtype=int) # 0
    H[6,:] = np.array([0,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0], dtype=int) # 0
    H[7,:] = np.array([0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0], dtype=int) # 1
    H[8,:] = np.array([1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0], dtype=int) # 0
    H[9,:] = np.array([0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0], dtype=int) # 1
    H[10,:] = np.array([0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1], dtype=int) # 1
    H[11,:] = np.array([0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0], dtype=int) # 1
    return H

def loadReferenceInput():
    referenceInput = np.zeros(shape=(16), dtype=int)
    referenceInput = np.array([0,1,0,1,1,1,1,0,0,0,0,1,0,0,0,1], dtype=int)
    return referenceInput

def loadReferenceSyndrome():
    syndrome = np.zeros(shape=(12), dtype=int)
    syndrome = np.array([0,1,1,1,0,0,1,0,1,1,0,1], dtype=int)
    return syndrome

def directoryMaker(dir: os.path):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")

if __name__ == "__main__":
    # main()
    findInequalResult()