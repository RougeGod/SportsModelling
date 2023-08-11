import numpy as n
from scipy.integrate import solve_ivp

#constants. These are global to the entire program
ROCK_RAD = 0.14 #rock radius r (metres)
RUNNING_RAD = 0.063 #running band radius R (metres)
ETA = ROCK_RAD**2 / (2*(RUNNING_RAD**2)) #dimensionless
ROCK_MASS = 19 #kilograms
MU1 = 0.004 #ice-ice dimensionless friction coefficient
MU2 = 0.016 #ice-stone dimensionless friction coefficient
GRAV = 9.81

def fphi(angle, phi0):
    GRAV = -9.81
    return (GRAV * ROCK_MASS * MU1) if (angle < phi0) else (GRAV * ROCK_MASS * MU2)

def phi0(w):
  return n.pi #comment out this line for variable phi0
  A = 13.1286615
  B = 0.122522839
  C = 75.00122
  D = 249.260950
  w = abs(w)
  if (w <= 0.03): #experimental data showed a phi0 of 180 degrees at this value
                  #so at this value and below, phi0 is taken as 0. The atan-based
                  #function isn't super accurate in this area anyways
    return n.pi
  elif (w <= 0.92786298936722): #interval ending at the point that the atan-based function 
                                #reaches 360 degrees. 
    return n.deg2rad((C*n.arctan(A*(w - B)) + D)) #converted to radians for the numpy trig functions
  else: #if the atan function results in a value higher than 360 degrees, return 2pi
    return 2*n.pi

#PHI0 = n.deg2rad(185)
def computeIntegrals(xVel, yVel, w): 
  PSI = n.arctan(-xVel/yVel)
  PHI0 = phi0(w)
  fInt =  (PHI0 * MU1 + (((2*n.pi)-PHI0) * MU2)) * ROCK_MASS * GRAV
  fsinInt  = ROCK_MASS * GRAV * (MU2 - MU1) * (n.cos(PHI0 + PSI) - n.cos(PSI))
  fcosInt  = ROCK_MASS * GRAV * (MU2 - MU1) * (n.sin(PSI) - n.sin(PHI0 + PSI))
  fsin2Int = (ROCK_MASS * GRAV * (MU2 - MU1) / 2) * ((n.sin(2*PHI0 + 2*PSI)/2) - (n.sin(2*PSI)/2) - PHI0) 
  return (fsinInt, fcosInt, fsin2Int,fInt)

INITIAL_X = INITIAL_Y = INITIAL_N = 0 #yes, this is legal and there's no reference shenanigans
                                      #INITIAL_N is the x-velocity
INITIAL_W = +0.05 #m/s, positive value indocates anti-clockwise spin
INITIAL_P = 2.196 #m/s, initial y-velocity of the throw
initialParams = [INITIAL_X, INITIAL_Y, INITIAL_N, INITIAL_P, INITIAL_W]


def dX_dt(fx, params):
    return params[2]
def dY_dt(fy, params):
    return params[3]
def dN_dt(fn, params): #direct equation plugging
    trigs = computeIntegrals(params[2], params[3], params[4])
    constPack = 2 * n.pi * params[3] * ROCK_MASS
    return ((params[4] * (trigs[0])) - (params[2] * trigs[3])) / (constPack)
def dP_dt(fP, params): 
    trigs = computeIntegrals(params[2], params[3], params[4])
    return -trigs[3] / (2 * n.pi * ROCK_MASS)
def dW_dt(fW, params):
    constPack = 4 * n.pi * ETA * ROCK_MASS
    trigs = computeIntegrals(params[2], params[3], params[4])
    t1 = params[2] * trigs[0] / params[3]
    t2 = trigs[1]
    t3 = params[4] * trigs[2] / params[3]
    return ((t1 - t2) - t3) / constPack 

def wholeSystem(fparams, params):
    return [dX_dt(fparams, params), dY_dt(fparams, params), 
    dN_dt(fparams, params), dP_dt(fparams, params), dW_dt(fparams, params)]

def yVelStop(fparams, params):
    global TIME_END #tell the rest of the program when the rock stops
    TIME_END = 24
    if (params[3] < 8e-8):
    #for some reason, recalculating phi0 causes
    #a huge number of steps near the end even with no huge derivatives,
    # and stopping slightly prematurely serves to keep both runtime and
    #JSON filesize reasonable. Unnecessary for the constant phi0, but
    #kept in for consistency in that case. 8e-8 was arbitrarily selected
        TIME_END = fparams 
        return 0
    return params[3]
yVelStop.terminal = True
yVelStop.direction = -1

def solveAndPlot(tolArray, clist):
  import matplotlib.pyplot as plt
  figure, axes = plt.subplots()
  for count in range (len(tolArray)): 
    tol = tolArray[count]
    traj = solve_ivp(wholeSystem, (0.0,n.inf), initialParams, atol=tol, rtol=tol, events=yVelStop,dense_output=True) 
    T_ARRAY = n.linspace(0,TIME_END,num=5000)
    X_POS_ARRAY = (traj.sol.__call__(n.linspace(0.0,TIME_END,num=5000)))[0]
    Y_POS_ARRAY = (traj.sol.__call__(n.linspace(0.0,TIME_END,num=5000)))[1]
    X_VEL_ARRAY = (traj.sol.__call__(n.linspace(0.0,TIME_END,num=5000)))[2]
    Y_VEL_ARRAY = (traj.sol.__call__(n.linspace(0.0,TIME_END,num=5000)))[3]
    W_ARRAY = (traj.sol.__call__(n.linspace(0.0,TIME_END,num=5000)))[4]
    ARRAY_END = 5000
    for bob in range(5000):
        if (Y_VEL_ARRAY[bob] < 0): 
        #some of the coarse-toleracnce solutions way overshoot the velocity stop event.
        #this ensures that points after velocity chnages to negative are excluded
            ARRAY_END = bob
            break
    xAxis = T_ARRAY[:ARRAY_END]
    yAxis = W_ARRAY[:ARRAY_END]
    axes.plot(xAxis, yAxis, clist[count], linewidth=1.78)
  axes.set(xlim=(min(xAxis),max(xAxis)*1.01),  ylim=(0,max(yAxis) * 1.01),
            ylabel="x (m)", xlabel="t (s)", title="Spin Rate of a Curling Rock Through Time")
  plt.subplots_adjust(top=0.97,right=0.98,left=0.03,bottom=0.07)
  plt.show()

solveAndPlot([0.03,1e-2,2.222e-14],["#cc0000","#cc6800","#0000cc","#99bb99"])
