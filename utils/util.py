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
        variance = np.sqrt(1/(fan_in+fan_out))
        m.weight.data.normal_(0.0, variance)


def adjust_learning_rate(optimizer,lr, epoch,step, lr_decay= 0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = lr * (0.1 ** (epoch // step))
    if(epoch%step==0):
        lr = lr_decay*lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def before_exit(model,now,reward):

    torch.save(model, "../../log/"+str(now))
    # model.save_model(str(now))
    draw(reward)
