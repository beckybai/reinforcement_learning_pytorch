import os
import shutil
import numpy as np
import torch
import sys

class Logger(object):
    def __init__(self,path):
        self.terminal = sys.stdout
        self.log = open("{}logfile.log".format(path), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def logger_init(out_dir,name):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        shutil.copyfile(sys.argv[0], out_dir + name)
    sys.stdout = Logger(out_dir)

