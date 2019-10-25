import numpy as np
import matplotlib.pyplot as plt


def plotIndReward(ind, task, hyp):
    # Get reward
    reward = task.getDistFitnessInd(ind, hyp)
    
    # Plot
    plt.plot(reward, marker='o')
    plt.ylim((0,1))
    plt.title("Individual Reward")
    plt.ylabel("Reward")
    plt.xlabel("Shared Weight Value")
    plt.xticks(np.arange(len(task.wVals)), task.wVals)
    plt.show()
    


def plotPopReward(pop, task, hyp, gen=None):
    # Get population reward
    reward = task.evaluatePop(pop, hyp)
    mean_reward = np.mean(reward, axis=0)
    
    for i in range(reward.shape[0]):
        plt.plot(reward[i,:], color='gray', linewidth=1, alpha=0.6)

    plt.ylim((0,1))
    if gen is None:
        plt.title("Population Reward")
    else:
        plt.title("Population Reward - Generation {}".format(gen))

    plt.plot(mean_reward, marker='o', color='blue')
    plt.ylabel("Reward")
    plt.xlabel("Shared Weight Value")
    plt.xticks(np.arange(len(task.wVals)), task.wVals)
    plt.show()