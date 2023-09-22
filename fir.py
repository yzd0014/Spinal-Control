import numpy as np

class FirFilt(object):
  def __init__(self, b):
    self.b = b
    self.n = len(b)
    self.x = np.array([0]*self.n)

  def filter(self,sample):
    self.x = np.concatenate((np.array([sample]),self.x[0:-1]))
    return np.dot(self.x,self.b)
