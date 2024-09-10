import os
import numpy as np
import math
from src import simulation, simulation_compare, parameters

import logging

def main():
    logLevel = logging.INFO
    blockLength = 16
    colDegree = 3
    rowDegree = 4 
    hammingWeigthLow = math.floor(blockLength/4)
    hammingWeigtHigh = math.ceil(blockLength *3 /4)
    decisionStep = 1
    numSim = 4000

    dir = os.path.join(os.getcwd(), "data")
    directoryMaker(dir)
    logdir = os.path.join(os.getcwd(), "log")
    directoryMaker(logdir)
    logfile = os.path.join(logdir, "log_eccvcc_difficulty_debug.log")
    logging.basicConfig(filename=logfile, format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logLevel)

    simSetting = np.array([blockLength, colDegree, rowDegree, hammingWeigthLow, hammingWeigtHigh, decisionStep, numSim], dtype=int)
    simSetting.tofile(os.path.join(dir, "settings.csv"), sep=",")

    param = parameters.codeParameters(blockLength, rowDegree, colDegree, hammingWeigthLow, hammingWeigtHigh, decisionStep)
    (solutionFound_SI, decoderSuccess_SI) = simulation.runMonteCarlo(param, dir=dir, numSim=numSim, printStamp = True)
    print("# of solution found = ", solutionFound_SI, ", Solving probability = ", float(solutionFound_SI)/float(numSim))
    print("# of decoder success = ", decoderSuccess_SI, ", Decoding probability = ", float(decoderSuccess_SI)/float(numSim))

    (solutionFound, decoderSuccess) = simulation_compare.runMonteCarlo(param, dir=dir, numSim=numSim, printStamp = True)
    print("# of solution found = ", solutionFound, ", Solving probability = ", float(solutionFound)/float(numSim))
    print("# of decoder success = ", decoderSuccess, ", Decoding probability = ", float(decoderSuccess)/float(numSim))



def directoryMaker(dir: os.path):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")

if __name__ == "__main__":
    main()