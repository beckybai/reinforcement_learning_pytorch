import numpy as np
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') !=-1:
        print('hello i am init')
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(1/(fan_in+fan_out))
        m.weight.data.normal_(0.0, variance)

