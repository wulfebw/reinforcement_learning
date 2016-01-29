
import numpy as np
import matplotlib.pyplot as plt

def moving_average(array, over):
    assert len(array) > 0
    averages = []
    for x in xrange(len(array)):
        base_index = x - over
        if base_index < 0:
            base_index = 0
        vals = array[base_index:x]
        averages.append(np.mean(vals))
    return averages

def better_moving_average(array, alpha):
    assert len(array) > 0

    averages = []
    average = array[0]
    for val in array:
        average = (1 - alpha) * average + alpha * int(val)
        averages.append(average)
    return averages


def plot_rewards(rewards):
    avg_rewards = better_moving_average(rewards, .005)
    plt.plot(avg_rewards)
    plt.show()