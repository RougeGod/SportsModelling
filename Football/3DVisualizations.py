'''Make a 3-d graph relating expected points to distance and first down'''

import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

mainModel = pickle.load(open("mainmodel.dat","rb"))
normalModel = pickle.load(open("normalmodel.dat","rb"))
fourthModel = pickle.load(open("4thmodel.dat","rb"))
KOModel = pickle.load(open("KOModel.dat","rb"))
clutchModel = pickle.load(open("final5.dat","rb"))
gtgModel = pickle.load(open("gtgmodel.dat","rb"))
decisionModel = pickle.load(open("4thChoiceModel.dat","rb"))

#this function exists so that the main function can be on top where it belongs
def runThings():
    plotEP()
    #plotEP(down=n.random.randint(1,5),sd=n.random.randint(-17,18),time=n.random.randint(0,1801),half=2,scoreType=-7)
    #plotDecisions(time=900,sd=0)
    
#wasn't expecting this to end up being recursive but it makes sense, 
#i guess, with teams kicking back and forth until time runs out
def EP(probArray, time, sd):
    MTTS = 261 #mean time to score, determined through averaging the time to next score of all events with a next score that half
    kickoffAdjustment = 0 if time <= 0 else EP(KOModel.predict_proba([[35, time - MTTS, sd]])[0], time - MTTS, sd)
    return ((probArray[6] - probArray[0]) * (6.97 - kickoffAdjustment) + #our TD prob minus their TD prob, multiplied by TD value (6.97 accounting for average conversion score)
            (probArray[5] - probArray[1]) * (3 - kickoffAdjustment) + #same but for FGs
            (probArray[4] - probArray[2]) * (2 + kickoffAdjustment)) #you receive the KO after a safety so add the KO adjustment
        
def GTGProbArray(down, dist, time, sd, underThree):
    output = list(gtgModel.predict_proba([[down, dist, time, sd, underThree]])[0])
    output.append(0)
    output[3:7] = output[2:6]
    mainModel.predict_proba([[down, dist, dist, time, sd, 1, underThree]])[0][2] #get safety against probs from main model
    output = n.array(output)
    output /= n.sum(output) #normalize so probs add up to 1.0
    return output

scoreMap = {-7:(0, "TD Against"),-3:(1,"FG Against"),-2:(2,"Safety Conceded"),0:(3,"Scoreless Rest of Half"),+2:(4,"Safety For"),+3:(5,"FG by offense"),+7:(6,"TD by offense")}
colours = ["viridis", "Wistia","cool","spring","summer","gnuplot","plasma"]
        
#selects the correct model and returns the expected point value for a play with the given parameters, or a certain scoring probability
#(does not include kickoff-type plays)
def EPFunc(down, toGoal, toFirst, time, sd, gtg, underThree, half=1, scoreType="EP"):
  if (scoreType == "EP"):
    if ((half == 2) and (time <= 300)):
         return EP(clutchModel.predict_proba([[down, min(toFirst, toGoal), toGoal, time, sd, int(gtg), int(underThree)]])[0],time,sd) 
    elif (gtg): 
         output = GTGProbArray(down, toGoal, time, sd, underThree)
         return EP(output, time, sd)
    elif (down == 4):
       return EP(fourthModel.predict_proba([[toGoal, toFirst, time, int(toGoal <= 37), sd]])[0],time,sd)  
    else: 
       return EP(normalModel.predict_proba([[down,min(toFirst,toGoal),toGoal,time,sd,int(underThree)]])[0],time,sd)
  else: 
    if (scoreType not in scoreMap):
        raise ValueError("Improper Score Type Provided")
    if ((half == 2) and (time <= 300)):
        return clutchModel.predict_proba([[down, min(toFirst, toGoal), toGoal, time, sd, int(gtg), int(underThree)]])[0][scoreMap[scoreType][0]]
    elif (gtg): 
         return GTGProbArray(down, toGoal, time, sd, underThree)[scoreMap[scoreType][0]]
    elif (down == 4):
       return fourthModel.predict_proba([[toGoal, toFirst, time, int(toGoal <= 37), sd]])[0][scoreMap[scoreType][0]]
    else: 
       return normalModel.predict_proba([[down,min(toFirst,toGoal),toGoal,time,sd,int(underThree)]])[0][scoreMap[scoreType][0]]
      
       
def plotEP(down=1, sd=0, time=1200, half=1, scoreType="EP"):
    z = n.zeros((100, 100))  # Initialize z with zeros. the first array axis is yardsToFirst, and the second is yardsToGoal
    x = n.linspace(1,100,num=100)
    y = n.linspace(1,25,num=100)
    x,y = n.meshgrid(x,y)
    for yardsToGoal in range(100):
        for yardsToFirst in range(100):
            index = yardsToFirst
            yardsToFirst = (yardsToFirst * (25/100) + 1)
            z[index][yardsToGoal] = EPFunc(down, yardsToGoal+1, min(yardsToFirst, yardsToGoal+1), time, sd, (yardsToFirst >= yardsToGoal), (time < 180),half,scoreType=scoreType)
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(x,y,z,cmap=colours[n.random.randint(len(colours))],rcount=100,ccount=100)
    axes.set(xlabel="Yards To Opponent's End Zone",ylabel="Yards to First Down",zlabel="Expected Points",
             title=("EP by Field Position and Distance to First Down, Possession team " + ("winning" if sd >= 0 else "losing")
             + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in half " + str(half) +  " on down #" + str(down)),zlim=(-3,7),ylim=(0,25),xlim=(0,100))
    if (scoreType != "EP"):
        axes.set(zlabel=scoreMap[scoreType][1] + "Probability",
        title="Probability of " + scoreMap[scoreType][1] + ", Possession Team " + ("winning" if sd >= 0 else "losing") + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in half " + str(half) + " on down #" + str(down))
        if (scoreType != "EP"):
            biggest = max(max(z[0]),max(z[99])) 
            #highest peak on either extreme end of the graph. 
            #while some points may technically be higher, this is good enough
            axes.set(zlim=(0,biggest * 1.2 if (biggest <= 0.5) else 1))
    plt.subplots_adjust(top=0.97,right=1,left=0.0,bottom=0.0)
    plt.show()
def plotDecisions(sd=0,time=1200):
    goForIt = n.zeros((100, 100))  #the first array axis is yardsToFirst, and the second is yardsToGoal
    kickFG = n.zeros((100,100))
    punt = n.zeros((100,100))
    x = n.linspace(1,100,num=100)
    y = n.linspace(1,25,num=100)
    x,y = n.meshgrid(x,y)
    for yardsToGoal in range(100):
        for yardsToFirst in range(100):
            index = yardsToFirst
            yardsToFirst = (yardsToFirst / 4)
            punt[index][yardsToGoal], kickFG[index,yardsToGoal],goForIt[index][yardsToGoal] = decisionModel.predict_proba([[yardsToGoal, min(yardsToFirst, yardsToGoal),time, int(yardsToGoal <= 40), sd]])[0]
            #sets the value of all three arrays in the same command
            
    #make goForIt graph        
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(x,y,goForIt,cmap="cool",rcount=100,ccount=100,label="Go for it Probability")
    axes.set(xlabel="Yards To Opponent's End Zone",ylabel="Yards to First Down",zlabel="Probabilities",
             title=("Probability of a team Going for it on 4th down, Possession team " + ("winning" if sd >= 0 else "losing")
             + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in the half"),zlim=(0,1),ylim=(0,25),xlim=(0,100))
    plt.subplots_adjust(top=0.97,right=1,left=0,bottom=0)
    plt.show()
    
    #make field goal graph
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(x,y,kickFG,cmap="winter",rcount=100,ccount=100,label="Field Goal Probability")
    axes.set(xlabel="Yards To Opponent's End Zone",ylabel="Yards to First Down",zlabel="Probabilities",
             title=("Probability of a team attempting a FG on 4th down, Possession team " + ("winning" if sd >= 0 else "losing")
             + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in the half"),zlim=(0,1),ylim=(0,25),xlim=(0,100))
    plt.subplots_adjust(top=0.97,right=1,left=0,bottom=0)
    plt.show()

    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(x,y,punt,cmap="Wistia",rcount=100,ccount=100,label="Punt Probability")
    axes.set(xlabel="Yards To Opponent's End Zone",ylabel="Yards to First Down",zlabel="Probabilities",
             title=("Probability of a team punting on 4th down, Possession team " + ("winning" if sd >= 0 else "losing")
             + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in the half"),zlim=(0,1),ylim=(0,25),xlim=(0,100))
    plt.subplots_adjust(top=0.97,right=1,left=0,bottom=0)
    plt.show()
runThings()