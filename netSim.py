import pandas as pd
import math
import os
import time
import matplotlib.pyplot as plt
from consensus import network
from src import parameters

def main(verbose = False):
    codeLength = 36
    param = parameters.codeParameters(blockLength=codeLength, hammingWeigthLow=math.ceil(codeLength/4), hammingWeigtHigh=math.floor(3*codeLength/4))
    powerDistribution = [0.5, 1.0, 1.5]
    peerList = pd.DataFrame(data = {"power": powerDistribution}, index = range(len(powerDistribution)))
    net = network.Network(networkId=0, parameters=param, peers=peerList)
    start = time.time()
    for i in range(2000):
        print("Simulating block #", i, ", (elapsed:", time.time()-start, "s)")
        net.generateNewProblem()
        net.mine(i, verbose)

    fileName = os.path.join(os.getcwd(), "netSimResult.csv")
    net.results.to_csv(fileName)

    data = net.results[net.results.winner != -1] # exclude failed block
    timeElapsed = data["elapsed"]
    winnerHistogram = data.groupby("winner").count().reindex(index = range(len(powerDistribution)), fill_value=0) # pad zero for workers who have never won

    print("result: ") 
    print(net.results.head())
    print("...")
    print("mean time: ", timeElapsed.mean(), "[unit work time]")
    print("best worker: worker#", winnerHistogram.idxmax().item(), " (", winnerHistogram.max().item(), " wins)")
    print("worst worker: worker#", winnerHistogram.idxmin().item(), " (", winnerHistogram.min().item(), " wins)")
    plt.figure(1)
    plt.hist(timeElapsed.apply(round), density=True, bins=int(timeElapsed.max()))
    plt.title("time elapsed distribution")

    plt.figure(2)
    plt.stem(winnerHistogram)
    plt.title("winner histogram")
    plt.show()


if __name__=="__main__":
    verbose = True
    main(verbose)