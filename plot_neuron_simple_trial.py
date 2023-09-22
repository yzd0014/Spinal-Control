import sys
import numpy as np
from matplotlib import pyplot as plt

def plot_neuron_simple_trial(filename):

  win = 1000
  d = np.loadtxt(filename,delimiter=',')


  qd1 = d[0:,0]
  qd2 = d[0:,1]
  q1 = d[0:,2]
  q2 = d[0:,3]
  qh1 = d[0:,4]
  qh2 = d[0:,5]
  vh1 = d[0:,6]
  vh2 = d[0:,7]
  l11 = d[0:,8]
  l12 = d[0:,9]
  l21 = d[0:,10]
  l22 = d[0:,11]
  l31 = d[0:,12]
  l32 = d[0:,13]
  u11 = d[0:,14]
  u12 = d[0:,15]
  u21 = d[0:,16]
  u22 = d[0:,17]
  u31 = d[0:,18]
  u32 = d[0:,19]
  l11d = d[0:,20]
  l12d = d[0:,21]
  l21d = d[0:,22]
  l22d = d[0:,23]

  plt.rcParams['lines.linewidth'] = 0.5

  control_effort = np.linalg.norm([u11[0:win], \
                                  u12[0:win], \
                                  u21[0:win], \
                                  u22[0:win]])
  print('\ncontrol effort = ')
  print(control_effort)
  print('\n')


  plt.figure(figsize=(15,8))

  plt.subplot(311)
  plt.title('neuron, ||u|| = ' + str(round(control_effort,4)))
  plt.plot(qd1,label='qd1')
  plt.plot(qd2,label='qd2')
  plt.plot(q1,label='q1')
  plt.plot(q2,label='q2')
  plt.plot(qh1,label='qh1')
  plt.plot(qh2,label='qh2')
  plt.ylabel('position')
  plt.legend(loc='lower right')
  plt.xlim(0, win)

  plt.subplot(312)
  plt.plot(u11,label='u11')
  plt.plot(u12,label='u12')
  plt.plot(u21,label='u21')
  plt.plot(u22,label='u22')
  plt.ylabel('activation')
  plt.legend(loc='lower right')
  plt.xlim(0, win)

  plt.subplot(313)
  plt.plot(l11d,label='l11')
  plt.plot(l12d,label='l12')
  plt.plot(l21d,label='l21')
  plt.plot(l22d,label='l22')
  plt.legend(loc='lower right')
  plt.ylabel('desired')
  plt.xlabel('timestep')
  plt.xlim(0, win)

  plt.savefig('neuron-simple.eps', format='eps', dpi=1200)
  plt.show()

if __name__ == '__main__':
  plot_neuron_simple_trial(*sys.argv[1:])
