import numpy as n

allPlays = open("2018.csv","r",encoding="utf-8")
plays = allPlays.read().splitlines()
allPlays.close()
count = 0
while (count < len(plays)): 
        play = plays[count].split(sep=",")
        play[0] = n.nan if (play[0] == "nan" or play[0] == "") else int(play[0])
        play[1] = int(play[1])
        play[3] = n.nan if (play[3] == "nan" or play[3] == "") else int(play[3])
        play[5] = n.nan if (play[5] == "nan" or play[5] == "")else int(play[5])
        play[8] = n.nan if (play[8] == "nan" or play[8] == "")else int(play[8])
        play[9] = n.nan if (play[9] == "nan" or play[9] == "") else int(play[9])
        play[11] = int(play[11])
        play[12] = int(play[12])
        plays[count] = play
        print(count, plays[count])
        count += 1
#plays = plays[:-1] #cut the "end of game" play at the end of the final game

def findHalfBoundaries():
    boundaries = []
    for count in range(len(plays)):
        
        if (plays[count][1] == 1800) and (plays[count][10] == "kickoff"):
        #30 mintues left in the half, and no play-type found. gives array locations of all
        #beginning of half boundaries. These reset the scoreboard for the first half. 
        #note: overtimes are excluded from analysis becasue CFL overtime is extremely different
            boundaries.append(count)
    boundaries.append(len(plays))
    return boundaries
    
def absoluteScoreDiff():
    output = n.zeros(73, dtype=n.short)
    for play in plays:
        print(type(play[8]), type(play[9]))
        if not (n.isnan(play[8]) or n.isnan(play[9]) or n.isnan(play[0]) or (len(play[10]) == 0)): 
        #the scores are with the play (some events lead to scores not being displayed)
        #also a play actually occured (not penalty or EoQ)
            output[abs(play[8] - play[9])] += 1
    return output        

def downAndDist():
    output = n.zeros((4,30), dtype=n.short)
    for play in plays:
        if not (n.isnan(play[3]) or n.isnan(play[5]) or n.isnan(play[0]) or play[10] == ""):
            if (play[5] >= 30):
                output[play[3] - 1][29] += 1
            else:
                output[play[3] - 1][play[5]] += 1
    print(output)

def writeListToFile(fileName, list):
    bob = open(fileName, "w",encoding="utf-8")
    bob.write(str(list))
    bob.close()

#find scoring plays, returned in map of {playNum, (homeScored?, numberOfPoints)}
def findScoreBoundaries():
    scores = {}
    count = 1
    while count < len(plays):
        #print(plays[count])
        hsd = plays[count][11] - plays[count - 1][11] #home score difference from last play
        vsd = plays[count][12] - plays[count - 1][12] #road score difference from last play
        if ((hsd == 6) or (vsd == 6)):
            for convert in range(count, len(plays)):
            #since penalties can occur between TD and convert success or fail, must loop until kickoff
                if (plays[convert][10] == "kickoff"):
                    scores[count] = (hsd > 0, 7) 
                    '''all touchdowns count as 7. the way that the model will work is that it determines the 
                    probabilities for each kind of scoring event, as if touchdowns were worth "Q" and 
                    field goals were worth "oingaboing" it would be the same. Converts are independent plays so they
                    can be modelled seperately once touchdown probability has been established'''
                    count = convert #advance the count so that the convert isn't counted as a new scoring event
                    break
        elif (hsd > 0):
            scores[count - 1] = (True, hsd)
        elif (vsd > 0):
            scores[count - 1] = (False, vsd)
        #when scores reset at the start of the game, the scores go down so it's not recorded as
        #a change in score
        count += 1
    return scores
    
def findPossChanges():
    turnovers = [] #yes i know not all CoP are turnovers
    halfStarts = findHalfBoundaries() 
    #sometimes change of possession happens due to the end of half or end of game. 
    #don't count these. actually now that i think about it i don't think it matters
    count = 1
    while count < len(plays):
        if ((plays[count][6] != plays[count - 1][6]) and (count not in halfStarts)):
            turnovers.append(count - 1)
        count += 1
    return turnovers

def nextScoreAndDrives():
    from bisect import bisect_left
    cop = findPossChanges() #should already be ordered so no need to sort
    scores = findScoreBoundaries()
    halfBreaks = findHalfBoundaries()
    scorePlays = list(scores.keys())
    scorePlays.sort() #map keys may not be sorted out of the box so make sure that they are
    nextScoreArray = [] #holds location of scoring events
    drivesToNext = []
    timeToNext = [] #holds the time until the next score
    for count in range(len(plays)):
        if (bisect_left(scorePlays,count) == len(scores)) or (count < halfBreaks[bisect_left(halfBreaks,count)] < scorePlays[bisect_left(scorePlays,count)]): #if the next halfbreak comes before the next score, or the end of data comes before the next score, return a "home scores 0" object. 
            nextScore = (True, 0, n.nan)
        else: 
            nextScore = scores[scorePlays[bisect_left(scorePlays,count)]]
        #nextScore[0] is "did the home team score". 
        #plays[count][6] == plays[count][7] is "are the possessing team and the home team the same" 
        #if these are equivalent, the possessing team scores next, so if they are
        #not equivalent, multiply resulting score by -1 to indicate non-possessing team scores next
        if (nextScore[0] == (plays[count][6] == plays[count][7])):
            nextScore = nextScore[1]
        else: 
            nextScore = -nextScore[1]
        nextScoreArray.append(nextScore)
        try: #if after the last score in the data set, this will error so need a try-except
            drivesToNext.append(bisect_left(cop, scorePlays[bisect_left(scorePlays,count)]) - bisect_left(cop, count)) 
            timeToNext.append(abs((plays[scorePlays[bisect_left(scorePlays,count)]][1] - plays[count][1])))
        except IndexError:
            drivesToNext.append(100) #doesn't matter what number goes here, by definition this is after the last score in the data set
            timeToNext.append(n.nan)
    #writeListToFile("09Drives.csv",drivesToNext)
    #writeListToFile("09NS.csv",nextScoreArray)
    #add the new columns to the file (only run once for a data set)
    for count in range(1,len(plays)):
        plays[count].append(nextScoreArray[count])
        plays[count].append(drivesToNext[count])
        plays[count].append(timeToNext[count] if nextScoreArray[count] != 0 else n.nan)
    import csv    
    bob = open("2018Time.csv","w",encoding="utf-8",newline="")
    writer = csv.writer(bob)
    writer.writerows(plays)
nextScoreAndDrives()