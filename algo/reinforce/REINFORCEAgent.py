import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from model.QNet import QNet_policy
import utils.util as util


class REINFORCEAgent():
    def __init__(self,state_dim, action_dim, learning_rate=0.0001,reward_decay = 0.99, e_greedy=0.9):
# load model or not
# render or not

        torch.cuda.set_device(0)
        print(torch.cuda.current_device())
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = learning_rate
        # self.reward_dacy = reward_decay
        self.gamma = reward_decay  # in according to the parameters in the formulation.
        self.epsilon = e_greedy
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 100000  # this decay is to slow. # TO DO: figure out the relationship between the decay and the totoal step.
        # try to use a good strategy to solve this problem.
        use_cuda = torch.cuda.is_available()
        self.LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.model = QNet_policy(self.state_dim, self.action_dim).cuda() if use_cuda else QNet_policy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5) # the learning rate decrease by a factor gamma every 10000 step_size.
        util.weights_init(self.model)  # The above is copy from SarsaAgent

        self.greedy = False
        self.trajectory = []



    def clear_trajectory(self):
        self.trajectory = []


    def sbc(self, v, volatile=False): # one-dimension
        if(len(np.shape(v))==1):
            return Variable(self.FloatTensor((np.expand_dims(v, 0).tolist())), volatile=volatile)
        elif(len(np.shape(v))==2):
            return Variable(self.FloatTensor(v), volatile=volatile)

    def select_action(self,state):
        if(self.greedy):
            print("doesn't need to be greedy now")
            # TO DO
            return 0
        else:
            p_actions = self.model(self.sbc(state)).data.tolist()[0]
            p_actions = p_actions/np.sum(p_actions)
            return np.random.choice(self.action_dim, 1, p =p_actions)[0] # the return of random.choice is a list, index zero aims to pick a number :)

    def discount_rewards(self, rewards):
        disrew = np.zeros(np.shape(rewards))
        rew = 0
        for i in reversed(range(len(rewards))):
            rew = rewards[i]+self.gamma * rew
            disrew[i] = rew
        return disrew


    def update(self):
        # batch consists of many trajectories. n * total_timestep(not fixed), batch_size is still one?
        discounted_reward = self.discount_rewards([point['reward'] for point in self.trajectory])
        tra_size = len(self.trajectory)
        rewards = np.zeros([tra_size,2])
        states,actions = [],[]
        for (i,point) in enumerate(self.trajectory):
            rewards[i,point['action']]=discounted_reward[i]
            states.append(point['state'])
            actions.append(point['action'].tolist())


        # discounted_reward = self.discount_rewards(rewards)
        # states = [point['state'] for point in self.]
        # can we do many steps together? and add them up....
        q_value = self.model(self.sbc(states))
        # q_value = [q_value[i][actions[i]] for i in range(q_value.data.size()[0])]
        loss = torch.sum(-torch.log((q_value))*self.sbc(rewards))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



