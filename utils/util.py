import numpy as np
import torch.nn as nn
import utils.rldraw as draw
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') !=-1:
        print('hello i am init')
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(1/(fan_in))
        m.weight.data.normal_(0.0, variance)

        # m.weight.data.uniform_(-variance,variance)
        if(fan_out==1):
          m.weight.data.uniform_(-0.001,0.001)
        
        



def adjust_learning_rate(optimizer,lr, epoch,step, lr_decay= 0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = lr * (0.1 ** (epoch // step))
    if(epoch%step==0):
        lr = lr_decay*lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def before_exit(model,reward,model_path):

    torch.save(model, model_path+'model.pt')
    # model.save_model(str(now))
    draw.reward_episode(reward,model_path+'reward.png')

def weight_copy(source, target,tau=1):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data *(1-tau) + param.data * tau)


# def weight_copy(source, target,tau=1):
#     weight_copy(source=source, target= target, tau=1)


