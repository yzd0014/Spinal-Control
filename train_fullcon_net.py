import mujoco as mj
from mujoco.glfw import glfw
import os
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
import pickle


from control import *

# Initial params
fs = 200
b = np.array([0.9, 0, 0, 0.9, 0, 0, 0, 0, 0.9, 0, 0, 0.9])
a = np.array([1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 1.0])*0.4691358024691358
controller_params = NeuronFullConParams(alpha=a, \
                                  beta=b, \
                                  gamma=1, \
                                  fc=10, \
                                  fs=fs)

targetMax = 0.85
targetMin = -0.85

xml_path = 'double_links_fast.xml'
global_timer = 0
episode_sec = 4

controller = NeuronFullConnectController(controller_params)
controller.set_action(np.array([0.1,0.5,0.1,0.5]))

dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)
# get framebuffer viewport
viewport_width, viewport_height = glfw.get_framebuffer_size(window)
viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Example on how to set camera configuration
cam.azimuth = 90
cam.elevation = -20
cam.distance = 3
cam.lookat = np.array([0.0, -1, 2])

#set the controller
mj.set_mjcb_control(controller.callback)

global_timer = data.time
t_start = data.time

def getLengths(theta1,theta2,model,data):
  data.qpos[0] = theta1
  data.qpos[1] = theta2
  mj.mj_forward(model,data)
  lens = np.array([data.actuator_length[0],data.actuator_length[1], \
                    data.actuator_length[2],data.actuator_length[3]])
  return lens

target = []
for i in np.linspace(-0.8, 0.8, 4):
  for j in np.linspace(-0.8, 0.8, 4):
    target.append(np.array([i, j]))

def cost(p):
  controller.setAlphaMatrix(p[0:12])
  controller.setBetaMatrix(p[12:])

  c = 0;
  for qtar in target:
    ld = getLengths(qtar[0],qtar[1],model,data)
    controller.set_action(ld)
    data.qpos[0] = 0;
    data.qpos[1] = 0;
    t_start = data.time
    while data.time - t_start < episode_sec:
      mj.mj_step(model, data)
      c += np.linalg.norm(qtar - data.qpos)

  print('\n')
  print(c)
  return c

# -----------------------------------------------------------------------------
# Partial set
# -----------------------------------------------------------------------------
#x0 = np.array([1,0.9,1,0.9])
x0 = pickle.load(open("x0_minimal.p", "rb"))
# map partial set to full set
pfun = lambda x : np.array([x[0], 0, 0, x[0], 0, 0,
                            0, 0, x[2], 0, 0, x[2],
                            x[1], 0, 0, x[1], 0, 0,
                            0, 0, x[3], 0, 0, x[3]])
bounds = Bounds(np.ones(4)*0,
                np.concatenate((np.ones(2)*1, np.ones(2)*0.9)),
                keep_feasible=True)
Fcost = lambda x : cost(pfun(x))
res = minimize(Fcost, x0, bounds=bounds, method = 'Nelder-Mead', \
                options={'disp': True, \
                          'verbose': True, \
                          'maxiter': 1e3, \
                          'maxfev' : 1e3, \
                          'xatol' : 1e-5})
print('\n\n')
print(res)
print('\n\n')
res.x
print(res.x)
p = pfun(res.x)
pickle.dump(res.x,open("./x0_minimal.p", "wb"))
pickle.dump(p,open("./p0_minimal.p", "wb"))

# -----------------------------------------------------------------------------
# Full set
# -----------------------------------------------------------------------------
#x0 = np.concatenate((np.ones(12)*0.1,np.ones(12)*0.1))
#x0 = pickle.load(open("p0.p", "rb"))
#bounds = Bounds(np.concatenate((np.ones(12)*0,np.ones(12)*0)),
#                np.concatenate((np.ones(12)*1,np.ones(12)*0.95)),
#                                keep_feasible=True)
#
#res = minimize(cost, x0, bounds=bounds, method = 'Nelder-Mead',
#                options={'disp': True,
#                         'verbose': True,
#                         'maxiter': 1e3,
#                         'maxfev' : 1e3,
#                         'xatol' : 1e-5})
#print('\n\n')
#print(res)
#print('\n\n')
#res.x
#print(res.x)
#p = res.x
#pickle.dump(p,open("./p0.p", "wb"))



controller.setAlphaMatrix(p[0:12])
controller.setBetaMatrix(p[12:])

while True:
  for qtar in target:
    ld = getLengths(qtar[0],qtar[1],model,data)
    controller.set_action(ld)
    data.qpos[0] = 0;
    data.qpos[1] = 0;
    t_start = data.time
    while data.time - t_start < episode_sec:
      mj.mj_step(model, data)
      # Update scene and render
      mj.mjv_updateScene(model, data, opt, None, cam,
                         mj.mjtCatBit.mjCAT_ALL.value, scene)
      mj.mjr_render(viewport, scene, context)

      # swap OpenGL buffers (blocking call due to v-sync)
      glfw.swap_buffers(window)

      # process pending GUI events, call GLFW callbacks
      glfw.poll_events()

glfw.terminate()



