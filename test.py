import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds

def generate_trajectory(n, min_time=1, max_time=5, min_x=-0.8, max_x=0.8):

  # Generate t unique time points between min_time and max_time
  dt = 1/200

  # Randomly select n values for the trajectory
  x1 = np.random.uniform(min_x,max_x,n)
  x2 = np.random.uniform(min_x,max_x,n)

  Xu= np.vstack((np.concatenate((np.array([0]),x1)),
                np.concatenate((np.array([0]),x2))))

  dX = X[:,0:-1] - X[:,1:]
  dXnorm = np.linalg.norm(dX,axis=0)
  Cmax = 5
  cost = lambda dt: np.absolute(Cmax - np.sum(np.dot(dXnorm,dt)))
  x0 = np.ones(n)*1
  bounds = Bounds(np.ones(n)*1,np.ones(n)*3,keep_feasible=True)
  res = minimize(cost, x0, bounds=bounds, method = 'Nelder-Mead',
                  options={'disp': True,
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

  # Create the trajectory
  trajectory = lambda tq : np.array([np.interp(tq,t,x1),np.interp(tq,t,x2)])

  return trajectory

# Example usage:
n = 5  # Number of values in the trajectory
min_time = 1  # Minimum time
max_time = 5  # Maximum time

trajectory = generate_trajectory(n, min_time, max_time)
