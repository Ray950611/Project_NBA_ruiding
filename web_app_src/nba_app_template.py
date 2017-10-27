from flask import Flask,request
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

@app.route("/")
def nba():
    template = "<!doctype html><html><body><title>NBA Predictor</title><div align='center' style='border:3px solid red'><h1>############Welcome to My NBA Predictor!############</h1><img src='/static/nba-logo-on-wood.jpg' alt='NBA Logo'style='width:474px;height:268px;''>"
    template+= "<FORM METHOD='LINK' ACTION='/game/'><INPUT style='width: 400px; padding: 36px; cursor: pointer; box-shadow: 6px 6px 5px; #999; -webkit-box-shadow: 6px 6px 5px #999; -moz-box-shadow: 6px 6px 5px #999; font-weight: bold; background: #ffff00; color: #000; border-radius: 10px; border: 1px solid #999; font-size: 180%;' TYPE='submit' VALUE='Game Prediction'></FORM>"
    template+= "<FORM METHOD='LINK' ACTION='/season/'><INPUT style='width: 400px; padding: 36px; cursor: pointer; box-shadow: 6px 6px 5px; #999; -webkit-box-shadow: 6px 6px 5px #999; -moz-box-shadow: 6px 6px 5px #999; font-weight: bold; background: #ffff00; color: #000; border-radius: 10px; border: 1px solid #999; font-size: 180%;' TYPE='submit' VALUE='Season Prediction'></FORM>"
    template+= "<h3> Web App by Rui Ding</h3></div></body></html>"
    return template
@app.route('/season/')
def form_season():
   Teams = ['SAS','GSW','OKC','CLE','TOR','LAC','ATL','BOS','CHO','UTA','IND','MIA','POR','DET','HOU','DAL','WAS','CHI','ORL','MEM','SAC','DEN','NYK','NOP','MIN','MIL','PHO','BRK','LAL','PHI']
   template = "<!doctype html><html><body><title>NBA Season Predictor</title><div align='center' style='border:2px solid red'><h1>Welcome to My NBA Season Predictor!</h1>"
   template+= "<form action='/season_result/' method='post'> Team name(e.g. SAS):<br><select name='teamname'>"
   #select tag
   for team in Teams:
       template+="<option value="+team+">"+team+"</option>"
   #input text
   template+="</select><br> Season(yyyy after 2015, no more than current season):<br><input type='text' name='season'><br><br><input type='submit' value='Submit'></form>"
   template+= "</div></body></html>"
   return template

@app.route("/season_result/",methods=['POST'])
def season_predict():
    #validate current season against input season, the module can accept a season input after November of that season started
        currentYear = int(datetime.now().year)
        currentMonth = int(datetime.now().month)
        if currentMonth>=11:
            cur_season = currentYear+1
        else:
            cur_season = currentYear
        team_name = request.form['teamname']
        season_input = request.form['season']
        TeamFull = ['San Antonio Spurs', 'Golden State Warriors', 'Oklahoma City Thunder', 'Cleveland Cavaliers', 'Toronto Raptors', 'Los Angeles Clippers', 'Atlanta Hawks', 'Boston Celtics', 'Charlotte Hornets', 'Utah Jazz', 'Indiana Pacers', 'Miami Heat', 'Portland Trail Blazers', 'Detroit Pistons', 'Houston Rockets', 'Dallas Mavericks', 'Washington Wizards', 'Chicago Bulls', 'Orlando Magic', 'Memphis Grizzlies', 'Sacramento Kings', 'Denver Nuggets', 'New York Knicks', 'New Orleans Pelicans', 'Minnesota Timberwolves', 'Milwaukee Bucks', 'Phoenix Suns', 'Brooklyn Nets', 'Los Angeles Lakers', 'Philadelphia 76ers']
        Teams = ['SAS','GSW','OKC','CLE','TOR','LAC','ATL','BOS','CHO','UTA','IND','MIA','POR','DET','HOU','DAL','WAS','CHI','ORL','MEM','SAC','DEN','NYK','NOP','MIN','MIL','PHO','BRK','LAL','PHI']
        #check input
        if season_input=='':
            Result = "Null season input!"
            template = "<!doctype html><html><body><div align='center' style='border:2px solid red'><h1>"+Result+"</h1>"+"<form action='/season/'><input type='submit' value='Back'></form></div></body></html>"
            return template
        season_input = int(season_input)
        if season_input <2016 or season_input>cur_season:
            Result = "Error season input!Not valid for app use."
            template = "<!doctype html><html><body><div align='center' style='border:2px solid red'><h1>"+Result+"</h1>"+"<form action='/season/'><input type='submit' value='Back'></form></div></body></html>"
            return template
        if team_name not in Teams:

            Result = "Error input team name!"
            template = "<!doctype html><html><body><div align='center' style='border:2px solid red'><h1>"+Result+"</h1>"+"<form action='/season/'><input type='submit' value='Back'></form></div></body></html>"
            return template
        #database
        season_train = season_input - 1

        advanced_train = 'http://www.basketball-reference.com/leagues/NBA_'+str(season_train)+'_advanced.html'


        req = requests.get(advanced_train)

        text = BeautifulSoup(req.text, 'html.parser')
        stats = text.find('div',{'id': 'all_advanced_stats'})

        # get rows

        PERAvg_train = np.zeros(30)
        GP_train = np.zeros(30)
        Min_train = np.zeros(30)

        for i in stats.tbody.find_all('tr'):

                row = [j.get_text() for j in i.find_all('td')]

                if len(row)==0:
                    continue
                team = row[3]

                if team!='TOT':

                    mins = row[5]
                    gp = str(row[4])

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

        # get rows

        Wins_train = np.zeros(30)
        Conf = np.zeros(30)
        for i in stats.tbody.find_all('tr'):

                team = [j.get_text() for j in i.find_all('td')]



                index = TeamFull.index(team[0])
                Wins_train[index] = float(team[5])
                Conf[index] = int(team[1] == 'W')
        PERAvg_train = np.array(PERAvg_train).reshape((30,1))
        Wins_train = np.array(Wins_train).reshape((30,1))
        #new inquiry
        #regular season data wrapping
        advanced_test = 'http://www.basketball-reference.com/leagues/NBA_'+str(season_input)+'_advanced.html'

        req = requests.get(advanced_test)

        text = BeautifulSoup(req.text, 'html.parser')
        stats = text.find('div',{'id': 'all_advanced_stats'})

        #print cols
        # get rows
        PERAvg = np.zeros(30)
        GP = np.zeros(30)
        Min = np.zeros(30)
        for i in stats.tbody.find_all('tr'):
                row = [j.get_text() for j in i.find_all('td')]


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
        per = PERAvg[Teams.index(team_name)].reshape(1,-1)
        ####

        regr = LinearRegression()
        #CALCULATING PREDICTED RESULTS USEING K-FOLD CROSS-VALIDATION
        kf = KFold(len(Wins_train),n_folds=5,shuffle=True)
        pred=[]
        regr = LinearRegression()
            # Iterate through folds
        for train_index, test_index in kf:

            X_train, X_test = PERAvg_train[train_index], PERAvg_train[test_index]
            y_train, y_test = Wins_train[train_index], Wins_train[test_index]

            regr.fit(X_train,y_train)
            pred.append(regr.predict(per)[0][0])

        ##############predict

        predicted = sum(pred)/len(pred)
        Result= "Predicted Winning Ratio for "+team_name+":"+str(predicted)
        template = "<!doctype html><html><body><title>Season Results</title><div align='center' style='border:2px solid red'><img src='/static/logos/"+team_name+".jpg' alt='Logo'style='width:300px;height:300px;''>"
        template+= "<h1>"+Result+"</h1><form action='/'><input type='submit' value='Home'></form></div></body></html>"
        return template

    ###########
@app.route("/game/")
def form_game():
   Teams = ['SAS','GSW','OKC','CLE','TOR','LAC','ATL','BOS','CHO','UTA','IND','MIA','POR','DET','HOU','DAL','WAS','CHI','ORL','MEM','SAC','DEN','NYK','NOP','MIN','MIL','PHO','BRK','LAL','PHI']
   template = "<!doctype html><html><body><title>NBA Game Predictor</title><div align='center' style='border:2px solid red'><h1>Welcome to My NBA Game Predictor!</h1>"
   template += "<form action='/game_result/' method='post'> Home Team name(e.g. SAS):<br><select name = 'homename'>"
   #select home/guest tag
   for team in Teams:
       template+="<option value="+team+">"+team+"</option>"
   template+="</select><br>Guest Team name(e.g. GSW):<br><select name = 'guestname'>"
   for team in Teams:
       template+="<option value="+team+">"+team+"</option>"
    #input text
   template+="</select><br>Season(yyyy after 2015, no more than current season):<br><input type='text' name='season'><br><br><input type='submit' value='Submit'></form>"
   template+= "</div></body></html>"
   return template
@app.route('/game_result/',methods=['POST'])
def game_predict():
        currentYear = int(datetime.now().year)
        currentMonth = int(datetime.now().month)
        if currentMonth>=11:
            cur_season = currentYear+1
        else:
            cur_season = currentYear
        home_team = request.form['homename']
        guest_team = request.form['guestname']
        season_input = request.form['season']
        #check input
        if season_input=='':
            Result = "Null Season Input!"
            template = "<!doctype html><html><body><div align='center' style='border:2px solid red'><h1>"+Result+"</h1>"+"<form action='/game/'><input type='submit' value='Back'></form></div></body></html>"
            return template

        season_input = int(season_input)
        if season_input <2016 or season_input>cur_season:
            Result="Error season input!Not valid for app use."
            template = "<!doctype html><html><body><div align='center' style='border:2px solid red'><h1>"+Result+"</h1>"+"<form action='/game/'><input type='submit' value='Back'></form></div></body></html>"
            return template


        ###initialization
        TeamFull = ['San Antonio Spurs', 'Golden State Warriors', 'Oklahoma City Thunder', 'Cleveland Cavaliers', 'Toronto Raptors', 'Los Angeles Clippers', 'Atlanta Hawks', 'Boston Celtics', 'Charlotte Hornets', 'Utah Jazz', 'Indiana Pacers', 'Miami Heat', 'Portland Trail Blazers', 'Detroit Pistons', 'Houston Rockets', 'Dallas Mavericks', 'Washington Wizards', 'Chicago Bulls', 'Orlando Magic', 'Memphis Grizzlies', 'Sacramento Kings', 'Denver Nuggets', 'New York Knicks', 'New Orleans Pelicans', 'Minnesota Timberwolves', 'Milwaukee Bucks', 'Phoenix Suns', 'Brooklyn Nets', 'Los Angeles Lakers', 'Philadelphia 76ers']
        Teams = ['SAS','GSW','OKC','CLE','TOR','LAC','ATL','BOS','CHO','UTA','IND','MIA','POR','DET','HOU','DAL','WAS','CHI','ORL','MEM','SAC','DEN','NYK','NOP','MIN','MIL','PHO','BRK','LAL','PHI']
        #regular season data wrapping
        if guest_team not in Teams or home_team not in Teams:
            Result = "Error input team name!"
            template = "<!doctype html><html><body><div align='center' style='border:2px solid red'><h1>"+Result+"</h1>"+"<form action='/game/'><input type='submit' value='Back'></form></div></body></html>"
            return template


        season_train = season_input - 1

        #database
        advanced_train = 'http://www.basketball-reference.com/leagues/NBA_'+str(season_train)+'_advanced.html'

        req = requests.get(advanced_train)

        text = BeautifulSoup(req.text, 'html.parser')
        stats = text.find('div',{'id': 'all_advanced_stats'})
        PERAvg = np.zeros(30)
        GP = np.zeros(30)
        Min = np.zeros(30)

        # get rows

        for i in stats.tbody.find_all('tr'):
            row = [j.get_text() for j in i.find_all('td')]

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
        ############get regression data
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
            # get rows

            for i in stats.tbody.find_all('tr'):
                row_i = [j.get_text() for j in i.find_all('td')]


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

        # get rows

        PERAvg_test = np.zeros(30)
        GP = np.zeros(30)
        Min = np.zeros(30)
        for i in stats.tbody.find_all('tr'):
            row = [j.get_text() for j in i.find_all('td')]

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
    ############linear regression for game margin prediction
        x_team = [guest_team,home_team]
        index_0 = Teams.index(x_team[0])
        index_1 = Teams.index(x_team[1])
        x = np.array([PERAvg_test[index_0],PERAvg_test[index_1]]).reshape(1,-1)
        regr = LinearRegression()
        kf = KFold(len(y),n_folds=5,shuffle=True)
        pred=[]
        # Iterate through folds
        for train_index, test_index in kf:

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            regr.fit(X_train,y_train)
            pred.append(regr.predict(x)[0][0])

        ##############predict

        predicted = sum(pred)/len(pred)

        #Result = "Game: "+TeamFull[index_1]+" vs "+TeamFull[index_0]+" <br/> "
        Result ="Hometeam Game Margin:"+str(predicted)+" <br/> "
        #####logistic regression for game winner prediction
        regr = LogisticRegression()
        kf = KFold(len(z),n_folds=5,shuffle=True)
        pred=[]

        x = np.array([PERAvg_test[index_0],PERAvg_test[index_1]])
        # Iterate through folds
        def model(x):
            return 1 / (1 + np.exp(-x))
        for train_index, test_index in kf:

                X_train, X_test = X[train_index], X[test_index]
                z_train, z_test = z[train_index], z[test_index]
                regr.fit(X_train,z_train)
                coef = regr.coef_[0]
                intercept = regr.intercept_[0]
                pred.append(model(x[0] * coef[0] + x[1]*coef[1]+intercept))

            ##############predict

        prob = sum(pred)/len(pred)
        Result+= "Homwteam Win Probability:"+str(prob)
        template = "<!doctype html><html><body><title>Game Results</title><div align='center' style='border:2px solid red'><p><img src='/static/logos/"+home_team+".jpg' alt='Home Logo'style='width:200px;height:200px;''> <h1>vs.</h1> <img src='/static/logos/"+guest_team+".jpg' alt='Guest Logo'style='width:200px;height:200px;''></p><h1>"+Result+"</h1>"+"<form action='/'><input type='submit' value='Home'></form></div></body></html>"
        return template





if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')