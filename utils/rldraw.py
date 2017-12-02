import matplotlib.pyplot as plt
import numpy as np

def reward_episode(rewards, image_path, env_name='', method_name='', comment=''):
    reward_list = rewards
    total_num = np.shape(reward_list)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(total_num)), reward_list)
    ax.set_xlabel('iteration')
    ax.set_ylabel('rewards')
    fig.suptitle("rewards_episodes_{}_{}_{}".format(env_name, method_name, comment))
    fig.savefig(image_path)