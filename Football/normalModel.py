#script for the expected points modelling, using scikit-learn. makes and tests the model
import numpy as n
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

###CONFIGURATION SETTINGS ###
NEED_NEW_MODEL = False
PRODUCE_PLOTS = True

pxp = open("LDNormal.csv","r",encoding="utf-8")

NUM_OF_PLAYS = 181412 #2015 - 2022 excl 2018+2020

plays = n.empty((NUM_OF_PLAYS,6),dtype=n.int16) #time can exceed 255
scores = n.empty((NUM_OF_PLAYS), dtype=n.int8)
weights = n.empty((NUM_OF_PLAYS), dtype=n.uint8)

count = 0
for line in pxp:
    line = line.split(sep=",")
    try:
       plays[count][0] = int(line[3]) #down
       plays[count][1] = (int(line[5])) #distance to first
       plays[count][2] = int(line[0]) #distance to end zone
       plays[count][3] = int(line[1]) #time remaining in half
       sd = (int(line[8]) - int(line[9])) #score differeential
       plays[count][4] = sd #score differential between teams
       plays[count][5] = int(int(line[1]) < 180) #are there less than three minutes remaining in the half?
       scores[count] = int(line[13])#next score type
       weights[count] = int(line[14]) ** 0.55
       count += 1 #if it gets partway through and errors out, this won't trigger so partially written data can be properly overwritten
    except ValueError:
        pass 
        #value errors will happen if data is missing (as it usually is for penalty or eoq)
        #we just ignore these and don't include them in the dataset
#print(count)
pxp.close()

if (NEED_NEW_MODEL):
    lr = LogisticRegression(max_iter=30000, multi_class="multinomial", n_jobs=1, C=n.inf).fit(plays,scores)
    pickle.dump(lr, open("normalmodel.dat","wb"), protocol=4) #opens file and stores the object there
    print("done fitting model!")
else: 
    lr = pickle.load(open("normalmodel.dat","rb"))
    print("opened the pickle jar")

'''Code to plot the various probabilities for first and 10 or first and goal 
all the way down the field based on the fitted model. '''
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
  acceptedPenalty = []
  thirdDown = []
  for yardsToEZ in range(1,100):
      #data fed to predict_proba: down, distance, field position, time in half, how much is offense leading by,is goal-to-go?,is 4th down?
      predictions = lr.predict_proba([[1,min(yardsToEZ,10),yardsToEZ,1200,0,0]])[0]
      oppTDProb.append(predictions[0])
      oppFGProb.append(predictions[1])
      oppSafetyProb.append(predictions[2])
      scorelessProb.append(predictions[3])
      safetyProb.append(predictions[4])
      FGProb.append(predictions[5])
      TDProb.append(predictions[6])
      pointsExpectation.append((predictions[0] * -6.97 + predictions[1] * -3 + predictions[2] * -2 + predictions[4] * 2 + predictions[5] * 3 + predictions[6] * 6.97))
      #penYard = yardsToEZ#70# + 10 if yardsToEZ <= 80 else (yardsToEZ + (100-yardsToEZ)/2)
      #penPred = lr.predict_proba([[2,penYard/5,50,1200,0,0]])[0]
      #acceptedPenalty.append((penPred[0] * -6.97 + penPred[1] * -3 + penPred[2] * -2 + penPred[4] * 2 + penPred[5] * 3 + penPred[6] * 6.97))
      #penPred = lr.predict_proba([[3,penYard/5,50,1200,0,0]])[0]
      #thirdDown.append((penPred[0] * -6.97 + penPred[1] * -3 + penPred[2] * -2 + penPred[4] * 2 + penPred[5] * 3 + penPred[6] * 6.97))
  #plot for frequent scores + no score
  figure, axes = plt.subplots()
  plt.title("Scoring Probabilities, 1st & 10/Goal, Tie Game, 20 Minutes Left In Half")
  axes.plot(range(1,100),oppTDProb,("#dd1317"),label="Defensive Team Touchdown Probability") #red
  axes.plot(range(1,100),oppFGProb,("#dd7513"),label="Defensive Team FG Probability") #orange
  axes.plot(range(1,100),scorelessProb,("#000000"),label="Probability of No Score This Half") #black
  axes.plot(range(1,100),FGProb,("#0a3a0c"),label="Possession Team FG Probability") #dark green
  axes.plot(range(1,100),TDProb,("#05fd09"),label="Possession Team Touchdown Probability") #bright green
  axes.set(xlim=(0,100), ylim=(0,1),ylabel="Probability of Score", xlabel="Yards from Endzone")
  axes.legend()
  plt.show()
  #plot for safeties
  '''figure, axes = plt.subplots()
  plt.title("Safety Probabilities, 1st & 10/Goal, Tie Game, 20 Minutes Left in Half")
  axes.plot(range(1,100),oppSafetyProb,("#ddda13"),label="Defensive Team Safety Probability") #yellow 
  axes.plot(range(1,100),safetyProb,("#2fdae0"),label="Possession Team Safety Probability") #teal
  axes.set(xlim=(0,100),ylim=(0,0.1),ylabel="Probability of Score",xlabel="Yards from Endzone")
  axes.legend()
  plt.show()'''
  #Expected Points plot
  figure, axes = plt.subplots()
  plt.title("Expected Points, 1st & 10/Goal, Tie Game, 20 Minutes Left in Half")
  axes.plot(range(1,100),pointsExpectation,("#5e59f2"),label="Expected Points (1st down)",linewidth=2)
  #axes.plot(range(1,100),acceptedPenalty,("#40f4a8"),label="Expected Points (2nd down)", linewidth=2)
 # axes.plot(range(1,100),thirdDown,("#40f4a8"),label="Expected Points (3rd down)", linewidth=2)
  axes.plot(range(-1,101),[0 for bob in range(102)],("#d21e40")) #break-even point
  axes.set(xlim=(0,100),ylim=(-3,6),ylabel="Expected Points from First Down",xlabel="Yards from Endzone")
  axes.legend()
  plt.show()
  
  '''Testing on the excluded 2018 season'''

def testOn2018(wantPlots): 
  season18 = open("2018Normal.csv","r",encoding="utf-8")
  plays18 = season18.read().splitlines()
  season18.close()
  
  NUMBER_OF_BUCKETS = 25 #increasing this number gives more granular probabilities but smaller samples
  
  ourTD = [{} for count in range(NUMBER_OF_BUCKETS)]
  ourFG = [{} for count in range(NUMBER_OF_BUCKETS)]
  noScore = [{} for count in range(NUMBER_OF_BUCKETS)]
  theirFG = [{} for count in range(NUMBER_OF_BUCKETS)]
  theirTD = [{} for count in range(NUMBER_OF_BUCKETS)]
  
  playCount = 0
  maxTDProb = 0
  while (playCount < len(plays18)):
      try:
          play = plays18[playCount].split(sep=",")
          play[2] = int(play[0]) #distance to goal
          play[0] = int(play[3]) #down
          play[3] = int(play[1]) #time
          play[1] = int(play[5]) #distance
          #must be in this exact order or else it may be overwritten
          play[4] = int(play[8]) - int(play[9]) #score differential (wrt the team with possession)
          #play[5] = 0 #after play[3] gets overwritten with time
          play[5] = int(play[3] < 180)
          play[6] = int(play[13]) #next score. not used for prediction (duh), only for testing
          plays18[playCount] = play[0:7] 
          #will not be written if parsing errored out. total array should be usable though, and elements not relevant to the model are thrown out
          #make it so future uses of the 2018 plays (ie when testing) don't have to re-parse the play text
          predictions = lr.predict_proba([[play[0],play[1],play[2],play[3],play[4],play[5]]])[0] 
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
      except ValueError:
          pass #ignore plays that don't have all required info (usually penalties/game breaks)
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
    s=[len(scoreTypes[count][i])/50 for i in range(len(scoreProbs[count]))])
    axes.plot(n.linspace(0.02,0.98),n.linspace(0.02,0.98),color="blue")
    axes.set(xlim=(0,1),ylim=(0,1),xlabel=("Expected Probability of " + scoreNames[count]),ylabel="Observed Probability")
    plt.show()
plotProbsandEP(PRODUCE_PLOTS)
testOn2018(PRODUCE_PLOTS)
  