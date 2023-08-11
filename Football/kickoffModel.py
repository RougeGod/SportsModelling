#script for the expected points modelling, using scikit-learn. makes and tests the model
import numpy as n
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

###CONFIGURATION SETTINGS ###
WEIGHTING_ENABLED = False
NEED_NEW_MODEL = False
PRODUCE_PLOTS = True

pxp = open("LateDecadeKickoffs.csv","r",encoding="utf-8")

NUM_OF_PLAYS = 16491 #required so that the numpy array doesn't have rows of zeros at the end

plays = n.empty((NUM_OF_PLAYS,3),dtype=n.int16) #time can exceed 255
scores = n.empty((NUM_OF_PLAYS), dtype=n.int8)
count = 0
for line in pxp:
    line = line.split(sep=",")
    try:
       plays[count][0] = int(line[0]) #distance to KO team's endzone
       plays[count][1] = int(line[1]) #time remaining in half
       sd = (int(line[9]) - int(line[8])) #score differential wrt kicking team 
       plays[count][2] = sd #score differential between teams
       scores[count] = int(line[13])#next score type
       count += 1 #if it gets partway through and errors out, this won't trigger so partially written data can be properly overwritten
    except ValueError:
        pass 
        #value errors will happen if data is missing (as it usually is for penalty or eoq)
        #we just ignore these and don't include them in the dataset. should never trigger on kickoffs
#print(count)
pxp.close()

if (NEED_NEW_MODEL):
    lr = LogisticRegression(max_iter=30000, multi_class="multinomial", n_jobs=1, C=100).fit(plays,scores)
    pickle.dump(lr, open("KOModel.dat","wb"), protocol=4) #opens file and stores the object there
    print("done fitting model!")
else: 
    lr = pickle.load(open("KOModel.dat","rb"))
    print("opened the pickle jar")

def EP(probArray, time, sd):
    MTTS = 261 #mean time to score, determined through averaging the time to next score of all events with a next score that half
    kickoffAdjustment = 0 if time <= 0 else EP(lr.predict_proba([[35, time - MTTS, sd]])[0], time - MTTS, sd)
    return ((probArray[6] - probArray[0]) * (6.97 - kickoffAdjustment) + #our TD prob minus their TD prob, multiplied by TD value
            (probArray[5] - probArray[1]) * (3 - kickoffAdjustment) + #same but for FGs
            (probArray[4] - probArray[2]) * (2 + kickoffAdjustment)) #you receive the KO after a safety so add the KO adjustment

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
  GRAPH_X_MAX = 1800
  GRAPH_X_MIN = 0
  for time in range(GRAPH_X_MIN,GRAPH_X_MAX):
      #data fed to predict_proba: yardLine, time, sd
      predictions = lr.predict_proba([[35,time,0]])[0]
      #print(predictions)
      #exit()
      oppTDProb.append(predictions[0])
      oppFGProb.append(predictions[1])
      oppSafetyProb.append(predictions[2])
      scorelessProb.append(predictions[3])
      safetyProb.append(predictions[4])
      FGProb.append(predictions[5])
      TDProb.append(predictions[6])
      pointsExpectation.append(EP(predictions,time,0))
  
  figure, axes = plt.subplots()
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),oppTDProb,("#dd1317")) #red
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),oppFGProb,("#dd7513")) #orange
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),oppSafetyProb,("#ddda13")) #yellow
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),scorelessProb,("#000000")) #black
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),safetyProb,("#2fdae0")) #teal
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),FGProb,("#0a3a0c")) #dark green
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),TDProb,("#05fd09")) #bright green
  axes.set(xlim=(GRAPH_X_MIN,GRAPH_X_MAX), ylim=(0,1),ylabel="Probability of Score", xlabel="Time Remaining")
  plt.show()
  
  figure, axes = plt.subplots()
  axes.plot(range(GRAPH_X_MIN,GRAPH_X_MAX),pointsExpectation,("#5e59f2"))
  axes.set(xlim=(GRAPH_X_MIN,GRAPH_X_MAX),ylim=(0,1),ylabel="Expected Points from Kickoff",xlabel="Time Remaining")
  plt.show()
  
  '''Testing on the excluded 2018 season'''

def testOn2018(wantPlots): 
  season18 = open("2018Kickoffs.csv","r",encoding="utf-8-sig")
  plays18 = season18.read().splitlines()
  season18.close()
  
  NUMBER_OF_BUCKETS = 12 #increasing this number gives more granular probabilities but smaller samples
  
  ourTD = [{} for count in range(NUMBER_OF_BUCKETS)]
  ourFG = [{} for count in range(NUMBER_OF_BUCKETS)]
  noScore = [{} for count in range(NUMBER_OF_BUCKETS)]
  theirFG = [{} for count in range(NUMBER_OF_BUCKETS)]
  theirTD = [{} for count in range(NUMBER_OF_BUCKETS)]
  
  playCount = 0
  maxTDProb = 0
  while (playCount < len(plays18)):
          play = plays18[playCount].split(sep=",")
          play[0] = int(play[0]) #distance to kicking team's endzone
          play[1] = int(play[1]) #time
          play[2] = int(play[8]) - int(play[9]) #score differential (wrt the team with possession)
          play[3] = int(play[13]) #next score. not used for prediction (duh), only for testing
          plays18[playCount] = play[0:4] 
          #will not be written if parsing errored out. total array should be usable though, and elements not relevant to the model are thrown out
          #make it so future uses of the 2018 plays (ie when testing) don't have to re-parse the play text
          predictions = lr.predict_proba([[play[0],play[1],play[2]]])[0] 
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
    return
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

plotProbsandEP(PRODUCE_PLOTS)
testOn2018(PRODUCE_PLOTS)
  