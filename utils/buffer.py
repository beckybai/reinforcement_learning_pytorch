from collections import deque
import random
import numpy as np

buffer_cap = 10000
class MemoryBuffer:
	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.size = size
		self.len = 0
		
	def sample(self,sample_n):
		sample_n = min(sample_n,self.len)
		batch = random.sample(self.buffer, sample_n)
		return batch
	
	
	# The input should be
	# a list of different elements we need to preserve
	def append(self,element):
		if(self.len < self.size):
			self.len += 1
			self.buffer.append(element)
		else:
			self.buffer.popleft()
			self.buffer.append(element)
			
			
	