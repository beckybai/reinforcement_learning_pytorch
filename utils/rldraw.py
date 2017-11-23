import matplotlib.pyplot as plt
import numpy as np

def reward_episode(rewards, env_name='', method_name='', comment=''):
    reward_list = rewards
    total_num = np.shape(reward_list)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(total_num, reward_list)
    fig.suptitle("rewards_episodes_{}_{}_{}".format(env_name, method_name, comment))
    fig.savefig("haha.jpg")