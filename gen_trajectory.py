import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from matplotlib import pyplot as plt

def generate_trajectory(n,min_time=1,max_time=10,min_x=-0.8,max_x=0.8,
                        doplot=False):

  dt = 1/200

  x1 = np.random.uniform(min_x,max_x,n)
  x2 = np.random.uniform(min_x,max_x,n)
  X = np.vstack((np.concatenate((np.array([0]),x1)),
                np.concatenate((np.array([0]),x2))))

  dX = X[:,0:-1] - X[:,1:]
  dXnorm = np.linalg.norm(dX,axis=0)
  Cmax = 10
  cost = lambda dt: np.absolute(Cmax - np.sum(np.dot(dXnorm,dt)))
  x0 = np.ones(n)*1
  bounds = Bounds(np.ones(n)*min_time,np.ones(n)*max_time,keep_feasible=True)
  res = minimize(cost, x0, bounds=bounds, method = 'Nelder-Mead',
                  options={'disp': False,
                          'maxiter': 1e3,
                          'maxfev' : 1e3,
                          'xatol' : 1e-5})

  Dt = res.x
  t = np.zeros((n,1))
  for i in range(len(Dt)-1):
    t[i+1] = t[i] + Dt[i]

  x1 = np.vstack((x1,x1))
  x1 = x1.transpose()
  x1 = x1.reshape(1,2*n)

  x2 = np.vstack((x2,x2))
  x2 = x2.transpose()
  x2 = x2.reshape(1,2*n)

  t = t.squeeze()
  t = np.vstack((np.concatenate((np.array([0]),t[0:-1]+dt)),t))
  t = t.transpose()
  t = t.reshape(1,2*n)


  t = t.squeeze()
  x1 = x1.squeeze()
  x2 = x2.squeeze()

  T = t[-1]

  if doplot:
    plt.figure()
    plt.plot(t,x1)
    plt.plot(t,x2)
    plt.show()

  # Create the trajectory
  trajectory = lambda tq : np.array([np.interp(tq,t,x1),np.interp(tq,t,x2)])

  return T, trajectory
