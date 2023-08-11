'''trying to match the sigmoid graph to the data by making a function of 
w to phi0. Attempted models include the logistic function, the arctan function, 
and the x/sqrt(x^2 + C) function'''
import numpy as n 
from scipy.optimize import curve_fit


'''
#Equation: E(Ax-B) / sqrt((Ax-B)^2 + C)+D
INITIAL_GUESSES = [6.0,1.0,0.74,270]
A = 13.66203513
B = 1.70391344
C = 2.60147291
D = 250.60295657
E = 107.89796253 
def sqrtFunc(x, A, B, C, D, E): 
#x is the whole array of x-values.
#I don't know how it does this, but math operations on the list
#apply to every element instead of appending to the list, so that'scipy
#why the sqrt operation sometimes fails
    numer = E*(A*x - B)
    denom = (((A*x - B) ** 2) + C) ** 0.5
    return (numer/denom) + D'''


#Equation: C*atan(A(x-B)) + D
INITIAL_GUESSES = [16,0.15,60,270]
A = 13.1286615
B = 0.122522839
C = 75.0012200
D = 249.260950

def atanFunc(x, A, B, C, D):
    return ((n.arctan(A*(x-B)) * C) + D)


'''
#Equation: D-(A/(C + e^(x-B)))

INITIAL_GUESSES = [60, 2.4, 0.3, 808]
For some reason, the logistic curve simply REFUSED to fit the function. Maybe that's a sign. 
Parameters are nonsense so no reason to include them

def logistic(x, A, B, C, D):
    frac = A / (C + n.exp(x - B))
    return D - frac

LOWER_BOUNDS = [-n.infty,-n.infty, 0, -n.infty] # C must be positive to avoid discontinuities in the function
UPPER_BOUNDS = [n.infty, n.inf, n.inf, n.inf]
'''

SPIN_RATE = [0,0.03,0.05,0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 1]
PHI0 = [180, 180, 185, 206.25, 233, 253.5, 275, 291, 307, 328.5, 338.5, 360]



bob = curve_fit(atanFunc, SPIN_RATE, PHI0, p0=INITIAL_GUESSES)

print(bob[0])