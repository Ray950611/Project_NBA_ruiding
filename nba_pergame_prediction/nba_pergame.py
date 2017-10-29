from bs4 import BeautifulSoup
import requests

import numpy as np
import matplotlib.pyplot as plt
TeamFull = ['San Antonio Spurs', 'Golden State Warriors', 'Oklahoma City Thunder', 'Cleveland Cavaliers', 'Toronto Raptors', 'Los Angeles Clippers', 'Atlanta Hawks', 'Boston Celtics', 'Charlotte Hornets', 'Utah Jazz', 'Indiana Pacers', 'Miami Heat', 'Portland Trail Blazers', 'Detroit Pistons', 'Houston Rockets', 'Dallas Mavericks', 'Washington Wizards', 'Chicago Bulls', 'Orlando Magic', 'Memphis Grizzlies', 'Sacramento Kings', 'Denver Nuggets', 'New York Knicks', 'New Orleans Pelicans', 'Minnesota Timberwolves', 'Milwaukee Bucks', 'Phoenix Suns', 'Brooklyn Nets', 'Los Angeles Lakers', 'Philadelphia 76ers']
Teams = ['SAS','GSW','OKC','CLE','TOR','LAC','ATL','BOS','CHO','UTA','IND','MIA','POR','DET','HOU','DAL','WAS','CHI','ORL','MEM','SAC','DEN','NYK','NOP','MIN','MIL','PHO','BRK','LAL','PHI']
#regular season data scraping
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
############
Boxscore = []
X_team = []#guestteam,hometeam
X=[]
y=[]#home margin
z = []#hometeam win = 1
for month in ["october","november","december","january","february","march","april"]:
    boxscore = "http://www.basketball-reference.com/leagues/NBA_"+str(season_train)+"_games-"+str(month)+".html"
    req = requests.get(boxscore) 

    text = BeautifulSoup(req.text, 'html.parser')
    stats = text.find('div',{'id': 'all_schedule'}) 
    cols = [i.get_text() for i in stats.thead.find_all('th')] 

    # convert from unicode to string 
    cols = [x.encode('UTF8') for x in cols] 
    #print cols
    # get rows 

    for i in stats.tbody.find_all('tr'):
        cols = [j.get_text() for j in i.find_all('td')] 

        row_i = [x.encode('UTF8') for x in cols]
        if row_i:
            if row_i[2]:
                Boxscore.append([int(row_i[2]),int(row_i[4])])
                X_team.append([row_i[1],row_i[3]])
                index_0 = TeamFull.index(row_i[1])
                index_1 = TeamFull.index(row_i[3])
                X.append([PERAvg[index_0],PERAvg[index_1]])
                y.append(int(row_i[4])-int(row_i[2]))
                z.append(int(int(row_i[4])>int(row_i[2])))#hometeam win
X = np.array(X).reshape((len(X),2))
y = np.array(y).reshape((len(X),1))
z = np.array(z).reshape((len(X),1))
#######
#new inquiry
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


PERAvg_test = np.zeros(30)
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
            PERAvg_test[index] += float(row[6]) * int(mins)
PERAvg_test /= Min
############

#####
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
#game margin

regr = LinearRegression()
kf = KFold(len(y),n_folds=5,shuffle=True)
pred=[]
#test case
x_team = ['SAS','GSW']        
index_0 = Teams.index(x_team[0])
index_1 = Teams.index(x_team[1])
x = np.array([PERAvg_test[index_0],PERAvg_test[index_1]]).reshape(1,-1)            
# Iterate through folds
print "5-fold R^2_test (game margin):"
for train_index, test_index in kf:

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        regr.fit(X_train,y_train)
        pred.append(regr.predict(x)[0][0])
        print regr.score(X_test,y_test)
        ##############predict

predicted = sum(pred)/len(pred)



print "Home:"+x_team[1]+" vs "+"Guest:"+x_team[0]

print "Hometeam game margin:"+str(regr.predict(x)[0][0])
#game winner
regr = LogisticRegression()
kf = KFold(len(z),n_folds=5,shuffle=True)
pred=[]
       
index_0 = Teams.index(x_team[0])
index_1 = Teams.index(x_team[1])
x = np.array([PERAvg_test[index_0],PERAvg_test[index_1]])            
# Iterate through folds
def model(x):
    return 1 / (1 + np.exp(-x))
print "5-fold R^2_test (game winner):"
for train_index, test_index in kf:

        X_train, X_test = X[train_index], X[test_index]
        z_train, z_test = z[train_index], z[test_index]

        regr.fit(X_train,z_train)
        coef = regr.coef_[0]
        #print coef
        intercept = regr.intercept_[0]
        #print intercept
        pred.append(model(x[0] * coef[0] + x[1]*coef[1]+intercept))
        print regr.score(X_test,z_test)
        ##############predict

prob = sum(pred)/len(pred)


print "homwteam win prob:"+str(prob)
