import matplotlib.pyplot as plt
import numpy as np

def reward_episode(rewards, env_name='', method_name='', comment=''):
    reward_list = rewards
    total_num = np.size(reward_list).max()
    plt.plot(total_num, reward_list)
    plt.title("rewards_episodes_{}_{}_{}", env_name, method_name, comment)

    plt.savefig("haha.jpg")