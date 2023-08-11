import matplotlib.pyplot as plt
import json
timeFile = open("timesteps.txt", "r", encoding="utf-8")
allArrays = json.load(timeFile)

'''structure of allArrays:
0 - 5 (inclusive): phi0 is treated as a constant pi
6 - 11 (inclusive): phi0 is recalculated and changes the trajectories
0,6: time
1,7: x-position
2,8: y-position
3,9: x-velocity
4,10: y-velocity
5,11: w (related to spin rate)

for both scenarios, atol = rtol = 1e-10 '''

maxTime = max(max(allArrays[0]),max(allArrays[2]))
def setupGraphs(paramNum, yLabel, graphTitle):
  figure, axes = plt.subplots()
  axes.plot(allArrays[0],allArrays[paramNum],"-c", label="Constant phi0")
  axes.plot(allArrays[6],allArrays[paramNum + 6],"-m", label="Variable phi0")
  minX = min(min(allArrays[paramNum]),min(allArrays[paramNum + 6]))
  maxX = max(max(allArrays[paramNum]),max(allArrays[paramNum + 6]))
  axes.set(xlabel="time (s)", ylabel=yLabel)
  axes.set(xlim=(0,maxTime), ylim=(minX,maxX), title=graphTitle)
  axes.legend()
  plt.show()
'''
setupGraphs(1,"x-position(m)", "X-position Evolution")
setupGraphs(2,"y-position(m)", "Y-position Evolution")
setupGraphs(3,"x-velocity(m/s)", "X-velocity Evolution")
setupGraphs(4,"y-velocity(m/s)", "Y-velocity Evolution")
setupGraphs(5,"omega * R (m/s)","Spin Rate Evolution")
'''
figure, axes = plt.subplots()
axes.plot(allArrays[0],allArrays[1],"-r", label="Constant phi0")
axes.plot(allArrays[2],allArrays[3],"-k", label="Variable phi0")
minX = min(min(allArrays[0]),min(allArrays[2]))
maxX = max(max(allArrays[0]),max(allArrays[2]))
minY = min(min(allArrays[1]),min(allArrays[3]))
maxY = max(max(allArrays[1]),max(allArrays[3]))
axes.set(xlabel="time (s)", ylabel="w (m/s)")
axes.set(xlim=(minX,maxX), ylim=(minY,maxY), title="Spin Rate Evolution")
axes.legend()
plt.show()
