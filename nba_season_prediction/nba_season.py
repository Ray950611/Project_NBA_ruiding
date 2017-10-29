from bs4 import BeautifulSoup
import requests
import csv
import numpy as np
import matplotlib.pyplot as plt
TeamFull = ['San Antonio Spurs', 'Golden State Warriors', 'Oklahoma City Thunder', 'Cleveland Cavaliers', 'Toronto Raptors', 'Los Angeles Clippers', 'Atlanta Hawks', 'Boston Celtics', 'Charlotte Hornets', 'Utah Jazz', 'Indiana Pacers', 'Miami Heat', 'Portland Trail Blazers', 'Detroit Pistons', 'Houston Rockets', 'Dallas Mavericks', 'Washington Wizards', 'Chicago Bulls', 'Orlando Magic', 'Memphis Grizzlies', 'Sacramento Kings', 'Denver Nuggets', 'New York Knicks', 'New Orleans Pelicans', 'Minnesota Timberwolves', 'Milwaukee Bucks', 'Phoenix Suns', 'Brooklyn Nets', 'Los Angeles Lakers', 'Philadelphia 76ers']
Teams = ['SAS','GSW','OKC','CLE','TOR','LAC','ATL','BOS','CHO','UTA','IND','MIA','POR','DET','HOU','DAL','WAS','CHI','ORL','MEM','SAC','DEN','NYK','NOP','MIN','MIL','PHO','BRK','LAL','PHI']
#regular season data scraping
#database
season_input = 2017
season_train = season_input - 1
#database
advanced_train = 'http://www.basketball-reference.com/leagues/NBA_'+str(season_train)+'_advanced.html'


req = requests.get(advanced_train) 

text = BeautifulSoup(req.text, 'html.parser')
stats = text.find('div',{'id': 'all_advanced_stats'}) 
cols = [i.get_text() for i in stats.thead.find_all('th')] 

# convert from unicode to string 
cols = [x.encode('UTF8') for x in cols] 
#print cols
# get rows 
rows=[]
for i in stats.tbody.find_all('tr'):
    cols = [j.get_text() for j in i.find_all('td')] 
    
    row_i = [x.encode('UTF8') for x in cols]
    
    rows.append(row_i)


PERAvg_train = np.zeros(30)
GP_train = np.zeros(30)
Min_train = np.zeros(30)

for row in rows:
    if len(row)==0:
        continue
    if row[3]!='TOT':
        team = row[3]
        mins = row[5]
        gp = row[4]
        index = Teams.index(team)
        if float(mins)/float(gp) > 8.0:
            GP_train[index] += int(gp)
            Min_train[index] += int(mins)
            PERAvg_train[index] += float(row[6]) * int(mins)
PERAvg_train /= Min_train
#y data
team_train = 'http://www.basketball-reference.com/leagues/NBA_'+str(season_train)+'_ratings.html'
req = requests.get(team_train) 

text = BeautifulSoup(req.text, 'html.parser')
stats = text.find('div',{'id': 'all_ratings'}) 
cols = [i.get_text() for i in stats.thead.find_all('th')] 

# convert from unicode to string 
cols = [x.encode('UTF8') for x in cols] 
#print cols
# get rows 
teams=[]
for i in stats.tbody.find_all('tr'):
    cols = [j.get_text() for j in i.find_all('td')] 
    
    row_i = [x.encode('UTF8') for x in cols]
    
    teams.append(row_i)
Wins_train = np.zeros(30)
Conf = np.zeros(30)
for team in teams:
    
    index = TeamFull.index(team[0])
    Wins_train[index] = float(team[5])
    Conf[index] = int(team[1] == 'W')    
PERAvg_train = np.array(PERAvg_train).reshape((30,1))
Wins_train = np.array(Wins_train).reshape((30,1))
#new inquiry
#regular season data scraping
advanced_test = 'http://www.basketball-reference.com/leagues/NBA_'+str(season_input)+'_advanced.html'

req = requests.get(advanced_test) 

text = BeautifulSoup(req.text, 'html.parser')
stats = text.find('div',{'id': 'all_advanced_stats'}) 
cols = [i.get_text() for i in stats.thead.find_all('th')] 

# convert from unicode to string 
cols = [x.encode('UTF8') for x in cols] 
#print cols
# get rows 
rows=[]
for i in stats.tbody.find_all('tr'):
    cols = [j.get_text() for j in i.find_all('td')] 
    
    row_i = [x.encode('UTF8') for x in cols]
    
    rows.append(row_i)


# find the schema 
team_test = 'http://www.basketball-reference.com/leagues/NBA_'+str(season_input)+'_ratings.html'
req = requests.get(team_test) 

text = BeautifulSoup(req.text, 'html.parser')
stats = text.find('div',{'id': 'all_ratings'}) 
cols = [i.get_text() for i in stats.thead.find_all('th')] 

# convert from unicode to string 
cols = [x.encode('UTF8') for x in cols] 
#print cols
# get rows 
teams=[]
for i in stats.tbody.find_all('tr'):
    cols = [j.get_text() for j in i.find_all('td')] 
    
    row_i = [x.encode('UTF8') for x in cols]
    
    teams.append(row_i)
Wins = np.zeros(30)
Conf = np.zeros(30)
for team in teams:
    
    index = TeamFull.index(team[0])
    Wins[index] = float(team[5])
    Conf[index] = int(team[1] == 'W')    
PERAvg = np.zeros(30)
GP = np.zeros(30)
Min = np.zeros(30)

for row in rows:
    if len(row)==0:
        continue
    if row[3]!='TOT':
        team = row[3]
        mins = row[5]
        gp = row[4]
        index = Teams.index(team)
        if float(mins)/float(gp) > 8.0:
            GP[index] += int(gp)
            Min[index] += int(mins)
            PERAvg[index] += float(row[6]) * int(mins)
PERAvg /= Min
PERAvg = np.array(PERAvg).reshape((30,1))
Wins = np.array(Wins).reshape((30,1))
####
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
#use cross validation to generate predictions
kf = KFold(len(Wins),n_folds=5,shuffle=True)
pred=[]
#test case Memphis Grizzlies
team_name = "MEM"
per = PERAvg[Teams.index(team_name)].reshape(1,-1)
regr = LinearRegression()
    # Iterate through folds
for train_index, test_index in kf:
        
    X_train, X_test = PERAvg_train[train_index], PERAvg_train[test_index]
    y_train, y_test = Wins_train[train_index], Wins_train[test_index]
    
    regr.fit(X_train,y_train)
    pred.append(regr.predict(per)[0][0])
 
    print "R^2 = "+str(regr.score(X_test,y_test))
   
predicted = sum(pred)/len(pred)
print "Predicted winning ratio for "+team_name+":"+str(predicted)

####
"""
colors = ['b','r']
plt.figure(figsize=(12,8))

for i in range(30):
    plt.plot(PERAvg[i], Wins[i],str(colors[int(Conf[i])])+'o')
    plt.annotate(Teams[i], xy=(PERAvg[i],Wins[i]), xytext=(PERAvg[i], Wins[i]))
plt.plot(PERAvg, predicted, 'k')
plt.ylabel("Win Ratio")
plt.xlabel("Weighted PER")
plt.title("2016-2017 Season prediction")
plt.show()
"""
###################
