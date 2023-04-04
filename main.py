import os
import numpy as np
from src import simulation, parameters

def main():
    verbose = False
    useOriginal= True
    blockLength = 16
    colDegree = 3
    rowDegree = 4 
    hammingWeigthLow = 4
    hammingWeigtHigh = 14
    decisionStep = 2
    numSim = 10000

    dir = os.path.join(os.getcwd(), "data")
    directoryMaker(dir)
    simSetting = np.array([blockLength, colDegree, rowDegree, hammingWeigthLow, hammingWeigtHigh, decisionStep, numSim], dtype=int)
    simSetting.tofile(os.path.join(dir, "settings.csv"), sep=",")

    param = parameters.codeParameters(blockLength, rowDegree, colDegree, hammingWeigthLow, hammingWeigtHigh, decisionStep)
    (solutionFound, decoderSuccess) = simulation.runMonteCarlo(param, dir=dir, numSim=numSim, verbose=verbose, printStamp = True, useOriginal= useOriginal)
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