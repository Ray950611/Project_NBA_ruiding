from bs4 import BeautifulSoup
import requests

import numpy as np
import matplotlib.pyplot as plt
def calc_momentum(k_period,d):
        n = len(k_period)
        if n==0:
            return 0
        init = k_period[0]
        flag = int(init[0])
        streak = int(init[1])
        momentum = 0
        for i in range(1,n):
            cur = k_period[i]
            flag_cur = int(cur[0])
            streak_cur = int(cur[1])
            if flag_cur==0 and flag==1:
                momentum*=d
                momentum+=streak*d
                flag = flag_cur
                streak = streak_cur
            elif flag_cur==1 and flag==0:
                momentum*=d
                momentum+= -streak*d
                flag = flag_cur
                streak = streak_cur
            else:
                flag = flag_cur
                streak = streak_cur
        if flag==1:
            momentum+=streak
        else:
            momentum-=streak
        return momentum
#example input
season = 2017
team = 'BOS'
#find team schedule and results
team_schedule = 'https://www.basketball-reference.com/teams/'+team+'/'+str(season)+'_games.html'
req = requests.get(team_schedule) 
text = BeautifulSoup(req.text, 'html.parser')
stats = text.find('div',{'id': 'all_games'}) 
cols = [i.get_text() for i in stats.thead.find_all('th')] 

    # convert from unicode to string 
cols = [x.encode('UTF8') for x in cols] 
#print cols
records = []
for i in stats.tbody.find_all('tr'):
        cols = [j.get_text() for j in i.find_all('td')] 
        row_i = [x.encode('UTF8') for x in cols]
        #store results in the records matrix
        if len(row_i)==14:
            date = row_i[0].split(',')[1][1:]
            home = len(row_i[4])
            opponent = row_i[5]
            #this records current game winning of losing
            result = int(row_i[12][0]=='W')
            #this records the current streak
            streak = int(row_i[12][2])
            records.append([date,home,opponent,result,streak])
records = np.array(records)
dates = list(records[:,0])
games_count = len(dates)
#example test momentum calculation for the 16-17 season in a moving 10 window
m_list_5 = [] 
m_list_10 = []
#hyperparameters k and d
k = 10
d = 0.5
for curgame_date in dates:
    #find its position in the team scheduls
    where = dates.index(curgame_date)
    #find previous k game  period, if current season has played less than k games, then return all that have been played   
    if where>=k:
        pre_period = records[where-k:where,3:5]
    else: 
        pre_period = records[0:where,3:5]
   
    moment = calc_momentum(pre_period,d)
    m_list_10.append(moment)
m_list_10 = np.array(m_list_10)
plt.plot(range(games_count),m_list_10,'bo-',label='k=10')
k = 5
for curgame_date in dates:
    #find its position in the team scheduls
    where = dates.index(curgame_date)
    #find previous k game  period, if current season has played less than k games, then return all that have been played 
    if where>=k:
        pre_period = records[where-k:where,3:5]
    else: 
        pre_period = records[0:where,3:5]  
    moment = calc_momentum(pre_period,d)
    m_list_5.append(moment)
m_list_5 = np.array(m_list_5)
plt.plot(range(games_count),m_list_5,'ro-',label='k=5')
plt.xlabel('Seasonal Progress')
plt.ylabel('Momentum')
plt.title(team+': Momentum comparison between k=5 and k=10, d='+str(d))
plt.legend(loc='upper right')
plt.show()
