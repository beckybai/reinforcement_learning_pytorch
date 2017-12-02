import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from model.QNet import QNet_policy, QNet
from model.VNet import VNet
import utils.util as util


class ACAgent():
    def __init__(self,state_dim, action_dim,critic = 'TD', learning_rate=0.0001,reward_decay = 0.99, e_greedy=0.9):
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
        self.use_cuda = torch.cuda.is_available()
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        """Model part"""
        # Policy Model $Pi$
        self.model = QNet_policy(self.state_dim, self.action_dim).cuda() if self.use_cuda else QNet_policy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5) # the learning rate decrease by a factor gamma every 10000 step_size.
        util.weights_init(self.model)  # The above is copy from SarsaAgent
        # State_Value Function V
        self.modelV = VNet(self.state_dim, value_dim=1).cuda() if self.use_cuda else VNet(self.state_dim,value_dim=1)
        self.optimizerV = optim.Adam(self.modelV.parameters(), lr=self.lr*3)
        util.weights_init(self.modelV)
        

        self.greedy = False
        self.trajectory = []
        # self.critic = 'Advantage' || 'TD(\labmda)Actor-Critic
        self.critic = critic
        # if(self.critic=='Q'):
        #     self.Critic = QNet_policy.
        

    def clear_trajectory(self):
        self.trajectory = []


    """
    convert numpy format data to variable.
    """
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
        tra_size = len(self.trajectory)
        states = np.zeros([tra_size,self.state_dim])
        if(self.critic==None):
            discounted_reward = self.discount_rewards([point['reward'] for point in self.trajectory])
            
            critics = np.zeros([tra_size, 2])  # 2 means the dimension of actions
            actions = []
            for (i,point) in enumerate(self.trajectory):
                critics[i,point['action']]=discounted_reward[i]
                # states.append(point['state'])
                states[i] = point['state']
                actions.append(point['action'].tolist())
            # critics = self.sbc(critics)

        elif(self.critic=='Q'):
            # TO  DO
            None
            # for (i,point) in enumerate(self.trajectory):
            #     critics[i,point['action']] = self.QC(state) # for the convenience of calculation
        elif(self.critic=='Adv'):
            None
            
        elif(self.critic=='TD'): # only use the value of the current state and the next state, also need a gamma,
            value = np.zeros([tra_size, 2])  # 2 means the dimension of actions
            reward = np.zeros([tra_size,1])
            actions = np.zeros([tra_size,1])
            new_states  = np.zeros([tra_size,4])
            # critics = torch.zeros([tra_size, 2]).cuda() if self.use_cuda else torch.zeros([tra_size,2])# 2 means the dimension of actions
            critics = np.zeros([tra_size,2])
            for (i,point) in enumerate(self.trajectory[:-1]): # this can be simplified...
                # critics[i,point['action']] = point['reward'] + self.gamma*self.modelV(self.sbc(self.trajectory[i+1]['state'])).data\
                #                               - self.modelV(self.sbc(point['state'])).data
                states[i] = point['state']
                actions[i] = point['action'].tolist()
                reward[i] = point['reward']
                new_states[i] = point['new_state']
            # None
            
            new_states[-1] = np.zeros(4)
            states[-1] = np.zeros(4)
            critics_tmp= self.sbc(reward.tolist()) + self.modelV(self.sbc(new_states,volatile=False))*self.gamma - self.modelV(self.sbc(states,volatile=False))
            
            td_error = torch.mean(critics_tmp.pow(2))
            self.optimizerV.zero_grad()
            td_error.backward()
            self.optimizerV.step()

            critics_np = critics_tmp.data.view(-1).tolist()
            for (i,a) in enumerate(actions.flatten().astype('int')):
                critics[i,a] = critics_np[i]
            # critics[:,actions.flatten().astype('int')] = critics_tmp.data.tolist()
            # torch[:,torch.from_numpy(actions)] = critics_tmp.data
            if self.use_cuda:
                critics = Variable(torch.from_numpy(critics.astype('float32'))).cuda()
            else:
                critics = Variable(torch.from_numpy(critics.astype('float32')))
            
        else:
            print('You may use an unimplemented critic crierion')

        # discounted_reward = self.discount_rewards(rewards)
        # states = [point['state'] for point in self.]
        # can we do many steps together? and add them up....
        policy_dis = self.model(self.sbc(states)) # distribution on the policy
        # q_value = [q_value[i][actions[i]] for i in range(q_value.data.size()[0])]
        # critics = critics.detach()
        loss = torch.sum(-torch.log((policy_dis+0.01))*critics)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



