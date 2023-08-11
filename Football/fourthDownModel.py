#script for the expected points modelling, using scikit-learn. makes and tests the model
import numpy as n
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

###CONFIGURATION SETTINGS ###
WEIGHTING_ENABLED = False
NEED_NEW_MODEL = False
PRODUCE_PLOTS = True
FG_RANGE = 37

pxp = open("4th.csv","r",encoding="utf-8-sig")

NUM_OF_PLAYS = 19544 #required so that the numpy array doesn't have rows of zeros at the end

plays = n.empty((NUM_OF_PLAYS,5),dtype=n.int16) #time can exceed 255
scores = n.empty((NUM_OF_PLAYS),dtype=n.int8)
choices = n.empty((NUM_OF_PLAYS), dtype=n.int8) #PUNT = 0, FG = 1, RUN/PASS= 2
count = 0
for line in pxp:
    line = line.split(sep=",")
    plays[count][0] = int(line[0]) #distance to end zone
    plays[count][1] = int(line[5]) #distance to first down
    plays[count][2] = int(line[1]) #time remaining in half
    plays[count][3] = (int(line[0]) <= FG_RANGE) #field goal range
    plays[count][4] = (int(line[8]) - int(line[9])) #score differential
    choices[count] = int(line[10]) #what does the offense choose? 
    scores[count] = int(line[13])  #next score: who and how much?
    count += 1 
pxp.close()

if (NEED_NEW_MODEL):
    lr = LogisticRegression(max_iter=30000, multi_class="multinomial", n_jobs=1, C=n.inf,tol=1e-5).fit(plays,scores)
    pickle.dump(lr, open("4thModel.dat","wb"), protocol=4) #opens file and stores the object there
    pt = LogisticRegression(max_iter=10000,multi_class="multinomial", C=n.inf).fit(plays,choices)
    pickle.dump(pt,open("4thChoiceModel.dat","wb"),protocol=4)
    print("done fitting models!")
else: 
    lr = pickle.load(open("4thModel.dat","rb"))
    pt = pickle.load(open("4thChoiceModel.dat","rb"))
    print("opened the pickle jar")


def plotProbsandEP(wantPlots):
  if (not wantPlots):
    return #this function's whole purpose is to make plots. do nothing if plots are unwanted
  oppTDProb = []
  oppFGProb = []
  oppSafetyProb = []
  scorelessProb = []
  safetyProb = []
  FGProb = []
  TDProb = []
  pointsExpectation = []
  GRAPH_X_MAX = 100
  GRAPH_X_MIN = 1
  for yardsToEZ in range(GRAPH_X_MIN,GRAPH_X_MAX):
      #data fed to predict_proba: yardLine, toFirst, time, (in range?), sd?
      predictions = lr.predict_proba([[yardsToEZ,min(yardsToEZ,5),1200,(yardsToEZ <= FG_RANGE),0]])[0] #next score
      oppTDProb.append(predictions[0])
      oppFGProb.append(predictions[1])
      oppSafetyProb.append(predictions[2])
      scorelessProb.append(predictions[3])
      safetyProb.append(predictions[4])
      FGProb.append(predictions[5])
      TDProb.append(predictions[6])
      pointsExpectation.append((predictions[0] * -6.97 + predictions[1] * -3 + predictions[2] * -2 + predictions[4] * 2 + predictions[5] * 3 + predictions[6] * 6.97)) 
  figure, axes = plt.subplots()
  plt.title("Scoring Probabilities, 4th and 5/Goal, 5 Minutes Remaining in the 3rd, Tie Game")
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),oppTDProb,("#dd1317"),label="Defensive Team Touchdown Chance") #red
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),oppFGProb,("#dd7513"),label="Defensive Team Field Goal Chance") #orange
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),FGProb,("#0a3a0c"),label="Possessing Team Field Goal Chance") #dark green
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),TDProb,("#05fd09"),label="Possessing Team Touchdown Chance") #bright green
  axes.set(xlim=(GRAPH_X_MIN,GRAPH_X_MAX), ylim=(0,1),ylabel="Probability of Score", xlabel="Yards To End Zone")
  plt.legend()
  plt.subplots_adjust(top=0.97,right=0.98,left=0.03,bottom=0.07,hspace=0.25,wspace=0.25)
  plt.show()
  figure, axes = plt.subplots()
  plt.title("Safety and No Score Probabilities, 4th and 5/Goal, 5 Minutes Remaining in the 3rd, Tie Game")
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),oppSafetyProb,("#ddda13"),label="Opposing Team Safety Chance") #yellow
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),scorelessProb,("#000000"),label="No Score this Half") #black
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),safetyProb,("#2fdae0"),label="Possession Team Safety Chance") #teal
  plt.subplots_adjust(top=0.97,right=0.98,left=0.03,bottom=0.07,hspace=0.25,wspace=0.25)
  plt.legend()
  plt.show()
  figure, axes = plt.subplots()
  plt.title("Expected Points, 4th and 5/Goal, 5 Minutes Remaining in the 3rd, Tie Game")
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),pointsExpectation,("#5e59f2"))
  axes.plot(range(-1,101),[0 for bob in range(102)],("#d21e40")) #break-even point
  axes.set(xlim=(GRAPH_X_MIN,GRAPH_X_MAX),ylim=(-3,6),ylabel="Expected Points from 4th and 5",xlabel="Yards To Endzone")
  plt.subplots_adjust(top=0.97,right=0.98,left=0.03,bottom=0.07,hspace=0.25,wspace=0.25)
  plt.show()
  
  puntProb = []
  FGTProb = []
  goProb = []

  for yardsToEZ in range(GRAPH_X_MIN,GRAPH_X_MAX):
      #data fed to predict_proba: yardLine, toFirst, time, inRange?,sd, (gtg?),(t < 180)?
      predictions = pt.predict_proba([[yardsToEZ,min(yardsToEZ,5),1200,(yardsToEZ <= FG_RANGE),0,(yardsToEZ <= 5)]])[0] #next play type
      puntProb.append(predictions[0])
      FGTProb.append(predictions[1])
      goProb.append(predictions[2]) 
  figure, axes = plt.subplots()
  plt.title("Offense Choice Probabilities, 4th and 5/Goal, 5 Minutes Remaining in the 3rd, Tie Game")
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),puntProb,("#f16d9a"),label="Teams Punt") #pink
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),FGTProb,("#6e7502"),label="Teams Attempt FG") #forest green
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),goProb,("#5843d1"),label="Teams Go For It") #purple
  plt.subplots_adjust(top=0.97,right=0.98,left=0.03,bottom=0.07,hspace=0.25,wspace=0.25)
  axes.set(xlim=(GRAPH_X_MIN,GRAPH_X_MAX),ylim=(0,1),ylabel="Offense's Choice from 4th and 5",xlabel="Yards To Endzone")
  plt.legend()
  plt.show() 
      
  '''Testing on the excluded 2018 season'''

def testOn2018(wantPlots): 
  season18 = open("4th18.csv","r",encoding="utf-8-sig")
  plays18 = season18.read().splitlines()
  season18.close()
  
  NUMBER_OF_BUCKETS = 15 #increasing this number gives more granular probabilities but smaller samples
  
  ourTD = [{} for count in range(NUMBER_OF_BUCKETS)]
  ourFG = [{} for count in range(NUMBER_OF_BUCKETS)]
  noScore = [{} for count in range(NUMBER_OF_BUCKETS)]
  theirFG = [{} for count in range(NUMBER_OF_BUCKETS)]
  theirTD = [{} for count in range(NUMBER_OF_BUCKETS)]
  
  playCount = 0
  while (playCount < len(plays18)):
          play = plays18[playCount].split(sep=",")
          play[0] = int(play[0]) #distance to goal, distance to first, time remaining, is FG range?, score difference, gtg?
          play[3] = int((play[0] <= FG_RANGE))
          play[2] = int(play[1])
          play[1] = int(play[5])
          play[4] = int(play[8]) - int(play[9])
          play[5] = int(play[13]) #next score. not used for prediction (duh), only for testing
          plays18[playCount] = play[0:6] 
          #will not be written if parsing errored out. total array should be usable though, and elements not relevant to the model are thrown out
          #make it so future uses of the 2018 plays (ie when testing) don't have to re-parse the play text
          predictions = lr.predict_proba([play[0:5]])[0] 
          #give probabilities for all plays in 2018
          '''each of these arrays representing scores has 20 maps as their elements. 
          each one of those maps represents a 5% bucket, with ourTD[0] holding the numbers of all plays 
          with less than a 5% chance of resulting in the possession team scoring a TD, etc. Safeties
          are almost always a <5% chance so they are excluded from probability analysis'''
          ourTD[int(predictions[6] * NUMBER_OF_BUCKETS)][playCount] = predictions[6]
          ourFG[int(predictions[5] * NUMBER_OF_BUCKETS)][playCount] = predictions[5]
          noScore[int(predictions[3] * NUMBER_OF_BUCKETS)][playCount] = predictions[3]
          theirFG[int(predictions[1] * NUMBER_OF_BUCKETS)][playCount] = predictions[1]
          theirTD[int(predictions[0] * NUMBER_OF_BUCKETS)][playCount] = predictions[0]     
          playCount += 1
  
  TDF = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32) 
  #count of times the next score was a Touchdown for, when predicted at certain probabilities
  FGF = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32)
  NS  = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32)
  FGA = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32)
  TDA = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32)
  
  for count in range(len(plays18)): #will probably error.
      for d in range(NUMBER_OF_BUCKETS):
          if ((count in ourTD[d]) and (plays18[count][-1] == +7)):
              TDF[d] += 1
          if ((count in ourFG[d]) and (plays18[count][-1] == +3)):
              FGF[d] += 1
          if ((count in noScore[d]) and (plays18[count][-1] ==  0)):
              NS[d] += 1
          if ((count in theirFG[d]) and (plays18[count][-1] == -3)):
              FGA[d] += 1
          if ((count in theirTD[d]) and (plays18[count][-1] == -7)):
              TDA[d] += 1
  
  #print the results 
  predErr = [0,0,0,0,0]
  
  TDoProbs = []
  FGoProbs = []
  NSoProbs = []
  FGAoProbs = []
  TDAoProbs = []
  
  for count in range(NUMBER_OF_BUCKETS):
      if (len(ourTD[count]) > 0):
        xProb = n.mean(list(ourTD[count].values()))
        oProb = TDF[count]/len(ourTD[count])
        TDoProbs.append(oProb)
        predErr[0] += abs(xProb - oProb) * len(ourTD[count]) * (16010/45693)
      if (len(ourFG[count]) > 0):
        xProb = n.mean(list(ourFG[count].values()))
        oProb = FGF[count]/len(ourFG[count])
        FGoProbs.append(oProb)
        predErr[1] += abs(xProb - oProb) * len(ourFG[count]) * (10984/45693)
      if (len(noScore[count]) > 0):
        xProb = n.mean(list(noScore[count].values()))
        oProb = NS[count]/len(noScore[count])
        NSoProbs.append(oProb)
        predErr[2] += abs(xProb - oProb) * len(noScore[count]) * (7323/45693)
      if (len(theirFG[count]) > 0):
        xProb = n.mean(list(theirFG[count].values()))
        oProb = FGA[count]/len(theirFG[count])
        FGAoProbs.append(oProb)
        predErr[3] += abs(xProb - oProb) * len(theirFG[count]) * (4231/45693)
      if (len(theirTD[count]) > 0):
        xProb = n.mean(list(theirTD[count].values()))
        oProb = TDA[count]/len(theirTD[count])
        TDAoProbs.append(oProb)
        predErr[4] += abs(xProb - oProb) * len(theirTD[count]) * (6985/45693)
  print("Prediction Errors:", predErr)
  print("Total Error:",sum(predErr))
  if (not wantPlots):
    return sum(predErr)
  scoreProbs = [TDoProbs,FGoProbs,NSoProbs,FGAoProbs,TDoProbs]
  scoreTypes = [ourTD,ourFG,noScore,theirFG,theirTD]
  scoreNames = ["Possession Team Touchdown","Possession Team Field Goal","No Score This Half","Defending Team Field Goal","Defending Team Touchdown"]
  for count in range(5): 
    figure, axes = plt.subplots()
    axes.scatter(n.linspace(0.5/NUMBER_OF_BUCKETS,(0.5/NUMBER_OF_BUCKETS + (1/NUMBER_OF_BUCKETS)*len(scoreProbs[count])),num=len(scoreProbs[count])),scoreProbs[count],color="black",
    s=[len(scoreTypes[count][i])/20 for i in range(len(scoreProbs[count]))])
    axes.plot(n.linspace(0.02,0.98),n.linspace(0.02,0.98),color="blue")
    axes.set(xlim=(0,1),ylim=(0,1),xlabel=("Expected Probability of " + scoreNames[count]),ylabel="Observed Probability")
    plt.show()
  return sum(predErr)

plotProbsandEP(PRODUCE_PLOTS)
testOn2018(PRODUCE_PLOTS)

  