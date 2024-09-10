import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2, chisquare


def main():
    fileName = os.path.join(os.getcwd(), "netSimResult.csv")
    data = pd.read_csv(fileName)
    data = data[data.winner != -1] # exclude failed blocks
    powerDistribution = [0.5, 1.0, 1.5]
    numberOfWorkers = len(powerDistribution)

    # time
    timeElapsed = data["elapsed"]
    timeElapsedInt = timeElapsed.apply(round)
    sampleMean = timeElapsedInt.mean()
    sampleVariance = timeElapsedInt.var()

    xaxis = np.arange(int(timeElapsedInt.max())) + 1.0
    pdf = (1.0 - (1.0 / sampleMean))**(xaxis-1) / sampleMean
    cdf = 1.0 - (1.0 - (1.0 / sampleMean))**xaxis

    sampleHist = timeElapsedInt.groupby(timeElapsedInt.values).count()
    sampleHist = sampleHist.reindex(xaxis).fillna(value=0) # zero padding

    # Chi-Square Test (block generation time)
    sampleTailPoint = math.ceil( 3.6 * math.sqrt(sampleVariance) )
    xaxis_aggTail = np.arange(sampleTailPoint) + 1
    ## 1. expected frequency
    pdf_aggTail = (1.0 - (1.0 / sampleMean))**(xaxis_aggTail-1) / sampleMean
    pdf_aggTail[-1] = (1.0 - (1.0 / sampleMean))**(sampleTailPoint-1)
    expectedFrequency = pdf_aggTail * len(timeElapsedInt)
    ## 2. observed frequency
    observedData = sampleHist.loc[:sampleTailPoint]
    observedTailWeight = sampleHist.loc[sampleTailPoint:].sum()
    observedData.loc[sampleTailPoint] = observedTailWeight
    observed = observedData.to_numpy(dtype="float")
    print("-------- Contingency Table for BGT --------")
    print(pd.DataFrame([observed, expectedFrequency]))
    print("------------------------------------------")
    ## 3. Test
    degreeOfFreedom = sampleTailPoint - 1
    criticalValue = chi2(degreeOfFreedom).ppf(0.95)
    result = chisquare(f_obs=observed, f_exp=expectedFrequency)
    numberOfBadCells = len(np.where(observed<5))
    print("-------- Chi-Square Test for BGT --------")
    print("    degree of freedom = ", degreeOfFreedom)
    print("    critical value (significance level = 0.05) = ", criticalValue)
    print("    test statistic = ", result.statistic)
    print("    p-value = ", result.pvalue)
    print("    test pass? ", result.pvalue > 0.05)
    print("----------------------------------------")

    winner = data["winner"]
    winnerHistogram = winner.groupby(winner.values).count().reindex(index = range(numberOfWorkers), fill_value=0)
    print("best worker: worker#", winnerHistogram.idxmax(), " (", winnerHistogram.max(), " wins)")
    print("worst worker: worker#", winnerHistogram.idxmin(), " (", winnerHistogram.min(), " wins)")

    # Chi-Square Test (winner distribution)
    ## 1. expected frequency
    expectedFrequency_winner = np.array(powerDistribution, dtype=float) * len(winner) / sum(powerDistribution)
    ## 2. observed frequency
    observed_winner = winnerHistogram.to_numpy(dtype="float")
    print("-------- Contingency Table for Winner --------")
    print(pd.DataFrame([observed_winner, expectedFrequency_winner]))
    print("---------------------------------------------")
    ## 3. Test
    degreeOfFreedom = numberOfWorkers - 1
    criticalValue = chi2(degreeOfFreedom).ppf(0.95)
    result = chisquare(f_obs=observed_winner, f_exp=expectedFrequency_winner)
    print("-------- Chi-Square Test for Winner --------")
    print("    degree of freedom = ", degreeOfFreedom)
    print("    critical value (significance level = 0.05) = ", criticalValue)
    print("    test statistic = ", result.statistic)
    print("    p-value = ", result.pvalue)
    print("    test pass? ", result.pvalue > 0.05)
    print("-------------------------------------------")

    fig1, ax1 = plt.subplots()
    ax1.plot(xaxis, pdf)
    ax1.hist(timeElapsedInt, density=True, bins=int(timeElapsedInt.max()))
    plt.grid(True)
    plt.title("block generation time frequency / pmf")
    plt.xlabel("block generation time (sec)")
    plt.ylabel("probability")
    plt.legend(["geometric model (mean=" + str(sampleMean) + ")", "observed"])

    fig2, ax2 = plt.subplots()
    ax2.plot(xaxis, cdf)
    ax2.hist(timeElapsedInt, density=True, bins=int(timeElapsedInt.max()), cumulative = True)
    plt.grid(True)
    plt.title("block generation time cumulative frequency / cdf")
    plt.xlabel("block generation time (sec)")
    plt.ylabel("probability (cumulative)")
    plt.legend(["geometric model (mean=" + str(sampleMean) + ")", "observed"])

    fig3, ax3 = plt.subplots()
    labels = []
    for i in range(numberOfWorkers):
        labels.append("worker "+str(i))
    bar = ax3.bar(labels, winnerHistogram)
    for rect in bar:
        height = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width()/2.0, 
                 height,
                 "%d" % height,
                 ha="center",
                 va="bottom",
                 size="12")
    plt.title("winner histogram")
    plt.xlabel("worker ID")
    plt.ylabel("number of wins")

    plt.show()


if __name__=="__main__":
    main()