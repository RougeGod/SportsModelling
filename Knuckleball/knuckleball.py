import numpy
from scipy.integrate import solve_ivp

#CONSTANTS AND PARAMETER DEFINITIONS
AIR_DENSITY = 1.293 #kg/m^3
BALL_MASS = 0.120 #kg
BALL_RADIUS = 0.100 #m
DRAG_COEFF = 0.4 #unitless
BALL_DENSITY = BALL_MASS / ((4/3) * numpy.pi * BALL_RADIUS**3) #kg/m^3
INITIAL_SHOT_VELOCITY = 12 #m/s
INIITIAL_SHOT_ANGLE = numpy.pi/10 #radians, facing angle is upwards
constantPack = AIR_DENSITY * numpy.pi * (BALL_RADIUS ** 2) / (BALL_MASS * 2)
#constantPack is useful when taking lift forces into account, but is only 
#used in one location without it. 


#"removing lift" turns out to be setting the lift coefficients to zero. 
#this results in a reasonable (~12m/s) terminal velocity for a beach (volley?)ball (this number is also estimated at 12m/s in the paper)
#and a reasonable terminal facing angle of straight down . When the lift coefficients
#are 1, the terminal facing angle is about -pi/9. Additionally, the lift coefficient in the phi 
#direction makes the angle of deviation......crazy and growing without bound



def dU_dt(fu,params):  #change in velocity 
    return (-9.81*numpy.sin(params[1])) - ((params[0] ** 2) * constantPack * DRAG_COEFF)    
def dtheta_dt(ftheta, params): #change in vertical travelling angle
    return (-9.81 * numpy.cos(params[1]) / params[0]) # + (ftheta[0] * constantPack)
#def dphi_dt(params, fphi): #change in horizontal travelling angle
#    return fphi[0] * constantPack
#without lift, there is no horizontal force and the sideways deviation is 0

def wholeSystem(fparams, params):
    return [dU_dt(fparams, params), dtheta_dt(fparams, params)]#, dphi_dt(params, fparams)]

#params in order: speed, theta, phi (phi not included) 
initialParams = [INITIAL_SHOT_VELOCITY, INIITIAL_SHOT_ANGLE]

def terminal(odeSys, TOLERANCE=1e-4, STEP_SIZE = 0.05, INITIAL_TIME = 0, initialSpeed=INITIAL_SHOT_VELOCITY, initialAngle=INIITIAL_SHOT_ANGLE): 
  #find the point in time where the ball travels terminal velocity at terminal angle
  time = INITIAL_TIME
  conds = [initialSpeed, initialAngle]
  while (True):
    solution = solve_ivp(odeSys, (time,time + STEP_SIZE), conds, t_eval=[time, time + STEP_SIZE], rtol=1e-7, atol=1e-7)
    if (abs(solution.y[1][0] - solution.y[1][1]) <= TOLERANCE) and (abs(solution.y[0][0] - solution.y[0][1]) <= TOLERANCE):
        print("Time until full freefall: ", time, " seconds")
        print("Facing angle at freefall: ", solution.y[1][1], " radians")
        print("Ball speed at freefall: ", solution.y[0][1], " m/s")
        break
    conds = [solution.y[0][1], solution.y[1][1]]#, solution.y[2][1]]
    time += STEP_SIZE


#returns the velocity and facing angle at the given time
def instConds(time):
    odeSolution = solve_ivp(wholeSystem, (0, time), initialParams, rtol=1e-7, atol=1e-7)
    return odeSolution.y.flatten() #returns both velocity([len/2 - 1]) and facing angle ([len - 1])
    
#to find the full trajectory and the range before hitting the ground, integrate velocity sin(theta) by
#time from 0 to time to get the x-position, and integrate velocity cos(theta) by time 
#from 0 to time to get the y-position. 
def yInt(time):
    results = instConds(time)
    return results[len(results)//2 - 1] * numpy.sin(results[-1])
def xInt(time): 
    results = instConds(time)
    return results[len(results)//2 - 1] * numpy.cos(results[-1])
def reynolds(time):
    results = instConds(time)
    CL = 2*BALL_RADIUS*BALL_DENSITY/AIR_DENSITY
    return (results[len(results)//2 - 1] * CL/(1.5e-5))
    #ball diameter * ball density / air density is the "characteristic length" D
    #Reynolds number is velocity * characteristic length (defined in the paper as 6m) / kinematic viscosity of air (~1.5e-5)

from scipy.integrate import quad
def yPosition(time):
    return quad(yInt, 0, time) 
def xPosition(time):
    return quad(xInt, 0, time)



import matplotlib.pyplot as plt

def makePlots():
    plotXVals = []
    plotYVals = []
    plotReVals = []
    plotTVals = []

    t = 0
    y = 0 #only here to reference the variable before use
    #the x-y trajectory agrees pretty well with the lift-less model from the paper
    #while the Reynolds number evolution disagrees pretty severely
    while (y >= 0): #until the ball hits the ground
        y = yPosition(t)[0]
        plotXVals.append(xPosition(t)[0])
        plotYVals.append(y)
        plotReVals.append(reynolds(t))
        plotTVals.append(t)
        t += 0.02 #measure 0.02 seconds later
    

    fig, ax = plt.subplots()
    ax.plot(plotXVals, plotYVals, "#5fb391", linewidth=2)
    ax.set(xlim=(min(plotXVals),max(plotXVals)), ylim=(min(plotYVals),max(plotYVals)), title="Partial Recreation of Figure 4a from [3]", 
    #ax.set(xlim=(0,8), ylim=(0,0.8), title="Ball Trajectory", 
    xlabel="Horizontal Displacement (m)", ylabel="Vertical Displacement (m)")
    plt.show() #shows the graph of x vs y (figure 4a). 
    fig, ax = plt.subplots() #python will overwrite the previous variables
    ax.plot(plotXVals, plotReVals[0:len(plotXVals)], "m", linewidth=2)
    ax.set(xlim=(0,max(plotXVals)), ylim=(1.5e6,max(plotReVals)),
    title="Partial Recreation of Figure 4b from [3]", xlabel="Horizontal Displacement (m)", ylabel="Instantaneous Reynolds Number")
    plt.show() #shows the graph of x vs Re (figure 4b). 
    
    fig, ax = plt.subplots() #python will overwrite the previous variables
    ax.plot(plotTVals, [re / 625000 for re in plotReVals], "-r", label="Reynolds Number / 625000")
    ax.plot(plotTVals, plotXVals, "-b", label="x-position")
    ax.plot(plotTVals, [y * 10 for y in plotYVals], "-y", label="y-poition * 10")
    ax.set(xlim=(0,max(plotTVals)), ylim=(min(plotXVals),max(plotXVals)),
    title="Time Evolution of Reynolds Number and Ball Position", xlabel="Time (s)")
    ax.legend()
    plt.show() 
    #shows the graph of t vs Re (which is a constant multiple of ball velocity), x, and y. it's a mess
terminal(wholeSystem, initialSpeed=1300,initialAngle=1.0)
#makePlots()