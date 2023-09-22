import numpy as np

class IirFilt(object):
  def __init__(self, b, a):
    self.b = b
    self.a = a
    self.n = len(b)
    self.x = np.array([0]*self.n)
    self.y = np.array([0]*self.n)

  def filter(self,sample):
    self.x = np.concatenate((np.array([sample]),self.x[0:-1]))
    out = (np.dot(self.x,self.b) - np.dot(self.y[1:],self.a[1:]))/self.a[0]
    self.y = np.concatenate((np.array([out]),self.y[0:-1]))
    return out
