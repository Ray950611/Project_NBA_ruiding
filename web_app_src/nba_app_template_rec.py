from flask import Flask,request,render_template
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
def similarity_cos(a,b,w):
    similarity = sum(a*b*w)/(np.sqrt(sum(a*a*w))*np.sqrt(sum(b*b*w)))
    return similarity
def nn(player,adv_data,cul_data,names_data):
    weight = np.array([4,3,2,5,5,5,5,4,4,4,2,1,1,3,5,4,5,2,5,5,4,4,3,3,3,2,4])
    player_stat = adv_data[player]
    pos = player_stat[0]
    #age = player_stat[1]
    #mp = player_stat[2]
    stat = np.array(player_stat[1:])
    base = np.array(cul_data[pos])
    if pos==0:
        basenames = names_data[pos]+names_data[pos+1]
    elif pos==4:
        basenames = names_data[pos]+names_data[pos-1]

    else:
        basenames = names_data[pos]+names_data[pos-1]+names_data[pos+1]
    means = []
    stds = []
    for i in range(len(stat)):
        datalist = base[:,i+1]
        means.append(np.mean(np.array(datalist)))
        stds.append(np.std(np.array(datalist)))
    means = np.array(means)
    stds = np.array(stds)
    sim_list = []
    for name in basenames:
        stat_other = np.array(adv_data[name][1:])
        z_1 = (stat-means)/stds
        z_2 = (stat_other-means)/stds
        diff_1 = 0.5*sum(weight*abs(z_1-z_2))/sum(weight)
        diff_2 = 1-similarity_cos(z_1,z_2,weight)
        diff = abs((diff_1+diff_2)*0.5)
        sim_list.append([round(diff,4),name])
    nn = sorted(sim_list)
    return nn

def calc_momentum(k_period,d=0.9):
        n = len(k_period)
        if n==0:
            return 0
        if n>10:
            k_period = k_period[-10:]
        n = len(k_period)
        init = k_period[0]
        flag = int(init)
        momentum = 0
        streak = 1
        for i in range(1,n):
            cur = k_period[i]
            flag_cur = int(cur)
            if flag_cur==0 and flag==1:
                momentum*=d
                momentum+=streak*d
                flag = flag_cur
                streak = 1
            elif flag_cur==1 and flag==0:
                momentum*=d
                momentum+= -streak*d
                flag = flag_cur
                streak = 1
            else:
                flag = flag_cur
                streak +=1
        if flag==1:
            momentum+=streak
        else:
            momentum-=streak
        return momentum

app = Flask(__name__)

@app.route("/")
def nba():

    return render_template('form.html')
@app.route('/season/')
def form_season():

   return render_template('form_season.html')

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
            image = ""
            result="<h1>"+Result+"</h1>"
            return render_template('season_result.html',image=image,result=result)
        season_input = int(season_input)
        if season_input <2016 or season_input>cur_season:
            Result = "Error season input!Not valid for app use."
            image = ""
            result="<h1>"+Result+"</h1>"
            return render_template('season_result.html',image=image,result=result)
        if team_name not in Teams:

            Result = "Error input team name!"
            image = ""
            result="<h1>"+Result+"</h1>"
            return render_template('season_result.html',image=image,result=result)
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
        Team_Name = TeamFull[Teams.index(team_name)]
        predicted = sum(pred)/len(pred)
        Result= "Predicted Winning Ratio for "+Team_Name+":"+str(predicted)
        image = "<img src='/static/logos/"+team_name+".jpg' alt='Logo'style='width:300px;height:300px;'>"
        result= "<h1>"+Result+"</h1>"
        return render_template('season_result.html',image=image,result=result)

    ###########
@app.route("/contract/")
def form_contract():

   return render_template('form_contract.html')
@app.route("/contract_result/",methods=['POST'])
def player_contract():
    player_name = request.form['player']
    currentYear = int(datetime.now().year)
    currentMonth = int(datetime.now().month)
    if currentMonth>=11:
        cur_season = currentYear+1
    else:
        cur_season = currentYear
    season_input = request.form['season']
    if season_input=='':
            Result = "Null Season Input!"
            image = ""
            result="<h1>"+Result+"</h1>"
            return render_template('contract_result.html',image=image,result=result)

    season_input = int(season_input)
    if season_input>cur_season:
        Result="Error season input!Not valid for app use."
        image = ""
        result="<h1>"+Result+"</h1>"
        return render_template('contract_result.html',image=image,result=result)
    advanced_train = 'http://www.basketball-reference.com/leagues/NBA_'+str(season_input)+'_advanced.html'
    req = requests.get(advanced_train)
    text = BeautifulSoup(req.text, 'html.parser')
    stats = text.find('div',{'id': 'all_advanced_stats'})
    data = {}
    pics = {}
    adv_data = {}
    Pos = ['PG','SG','SF','PF','C']
    for i in stats.tbody.find_all('tr'):
        src = i.find('a')
        source = 'https://www.basketball-reference.com/'+str(src)[str(src).find('"/')+2:str(src).find('">')]
        row = [j.get_text() for j in i.find_all('td')]
        ind = [0,0,0,0,0]
        if len(row)==0:
            continue
        pos_str = row[1]
        pos = pos_str
        if len(pos_str)>2:
            plist = pos_str.split('-')
            pos = plist[0]
        if pos not in Pos:
            continue

        pos_num = Pos.index(pos)
        name = str(row[0])
        ind[pos_num]=1
        mins = row[5]
        gp = row[4]
        mp = float(mins)/float(gp)
        age = int(row[2])
        #vet = int(age>=35)
        #roo = int(age<=25)
        pics.update({name:source})
        if mp > 8.0 and age>24:

            per = float(row[6])
            wS = float(row[21])
            if name not in data.keys() and per>=0 and wS>=0:

                data.update({name:[pos_num,mp,wS,per,mp*per]+ind})
        if float(mp) > 10.0 and float(gp)>10:
            name = str(row[0])
            per = float(row[6])
            ts = float(row[7])
            att3 = float(row[8])
            trb = float(row[12])/100
            asr = float(row[13])/100
            slr = float(row[14])/100
            blr = float(row[15])/100
            tov = float(row[16])/100
            usg = float(row[17])/100
            wS = float(row[21])
            vorp = float(row[27])

            if name not in adv_data.keys() and per>=0 and wS>=0:
                adv_data.update({name:[pos_num,age,mp,per,ts,att3,trb,asr,slr,blr,tov,usg,wS,vorp]})

    basic_train = 'https://www.basketball-reference.com/leagues/NBA_'+str(season_input)+'_per_minute.html'
    req = requests.get(basic_train)
    text = BeautifulSoup(req.text, 'html.parser')
    stats = text.find('div',{'id': 'all_per_minute_stats'})
    for i in stats.tbody.find_all('tr'):

        row = [j.get_text() for j in i.find_all('td')]

        if len(row)==0:
            continue

        name = str(row[0])
        if name in adv_data.keys() and len(adv_data[name])==14:
            fg=0
            if len(row[9])!=0:
                fg = float(row[9])
            fa = float(row[8])
            fg3=0
            if len(row[12])!=0:
                fg3 = float(row[12])
            fa3 = float(row[11])
            ft=0
            if len(row[18])!=0:
                ft = float(row[18])
            fta = float(row[17])
            orb = float(row[19])
            drb = float(row[20])
            ast = float(row[22])
            stl = float(row[23])
            blk = float(row[24])
            tov = float(row[25])
            pf = float(row[26])
            pts = float(row[27])


            adv_data[name] += [fg,fa,fg3,fa3,ft,fta,orb,drb,ast,stl,blk,tov,pf,pts]
    if player_name not in data.keys():
        Result="Error player input!Not valid for app use."
        image = ""
        result="<h1>"+Result+"</h1>"
        return render_template('contract_result.html',image=image,result=result)
    pos = data[player_name][0]
    cur_data = np.array(data[player_name])[2:]
    cap = requests.get('http://www.spotrac.com/nba/cap/'+str(season_input-1)+'/')
    soup= BeautifulSoup(cap.text,"html.parser")
    table = soup.find('div',{'id':'main','class':' xlarge'}).find()
    sal_cap = str(table)[int(str(table).find('$')+1):int(str(table).find('<br>'))]
    cap_value = float(sal_cap.replace(',',''))/10000
    ratio=sum(np.array([-0.00652719,-0.00527774, 0.00048289,-0.01375509,-0.01284177, 0.00440119, 0.01184942,0.02707469])*np.array(cur_data))+ 0.0167122889116
    result = int(cap_value*ratio)
    #if pos==0:
        #ratio = sum(np.array([-0.00070575,-0.00817469,0.00011821,-0.00726101,0.01755548,0.00047755])*np.array(cur_data))+0.0394395071817
        #result = int(cap_value*ratio)
    #elif pos==1:
        #ratio = sum(np.array([ 0.00127516,-0.00549081,0.00018027,-0.00601407, 0.00168783,0.00029858])*np.array(cur_data))+0.000581801836712
        #result = int(cap_value*ratio)
    #elif pos==2:
        #ratio = sum(np.array([-0.00807138,-0.01120941,-0.00030995,-0.01345237,-0.00360127,0.00116188])*np.array(cur_data))+0.152946682404
        #result = int(cap_value*ratio)
    #elif pos==3:
        #ratio = sum(np.array([-1.55533327e-03,-9.88465488e-05,-1.94250561e-04,-3.16948117e-03,3.70776890e-04, 5.38489144e-04])*np.array(cur_data))+0.000464113193763
        #result = int(cap_value*ratio)
    #elif pos==4:
        #ratio = sum(np.array([2.34851775e-03,-3.02986786e-03,-5.13931514e-05,-6.32776724e-03,6.21444475e-04,4.00999095e-04])*np.array(cur_data))+0.000328999854055
        #result = int(cap_value*ratio)
    Result = Pos[pos]+" "+player_name+" : $"+str(result*10000)
    p_im = requests.get(pics[player_name])
    img= BeautifulSoup(p_im.text,"html.parser")
    image = img.find('div',{'class':'players','id':'info'}).find('div',{'class':'media-item'}).find('img')
    plink = str(image)[str(image).find('http'):str(image).find('">')]

    Image="<img src='"+plink+"' alt='Headshot' style='width:300px;height:450px;'>"
    result = "<h1>"+Result+"</h1>"
    #total_data = []
    #total_names = []
    cul_data = [[],[],[],[],[]]
    names_data = [[],[],[],[],[]]
    for name in adv_data.keys():
        #total_names.append(name)
        #total_data.append(adv_data[name])
        if len(adv_data[name])==28:
            pos = adv_data[name][0]
            cul_data[pos].append(adv_data[name])
            names_data[pos].append(name)

    NN = nn(player_name,adv_data,cul_data,names_data)[1:6]
    #sim = NN[0][1]+','+NN[1][1]+','+NN[2][1]+','+NN[3][1]+','+NN[4][1]
    result+="<h1>Similar Players:</h1>"
    for i in range(5):
        p_im = requests.get(pics[NN[i][1]])
        img= BeautifulSoup(p_im.text,"html.parser")
        image = img.find('div',{'class':'players','id':'info'}).find('div',{'class':'media-item'}).find('img')
        plink = str(image)[str(image).find('http'):str(image).find('">')]

        result+="<h2>"+NN[i][1]+"</h2><img src='"+plink+"' alt='Headshot' style='width:100px;height:150px;'>"

    return render_template('contract_result.html',image = Image,result=result)
@app.route("/game/")
def form_game():

   return render_template('form_game.html')
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
            image = ""
            result="<h1>"+Result+"</h1>"
            return render_template('game_result.html',image=image,result=result)

        season_input = int(season_input)
        if season_input <2016 or season_input>cur_season:
            Result="Error season input!Not valid for app use."
            image = ""
            result="<h1>"+Result+"</h1>"
            return render_template('game_result.html',image=image,result=result)


        ###initialization
        #TeamFull = ['San Antonio Spurs', 'Golden State Warriors', 'Oklahoma City Thunder', 'Cleveland Cavaliers', 'Toronto Raptors', 'Los Angeles Clippers', 'Atlanta Hawks', 'Boston Celtics', 'Charlotte Hornets', 'Utah Jazz', 'Indiana Pacers', 'Miami Heat', 'Portland Trail Blazers', 'Detroit Pistons', 'Houston Rockets', 'Dallas Mavericks', 'Washington Wizards', 'Chicago Bulls', 'Orlando Magic', 'Memphis Grizzlies', 'Sacramento Kings', 'Denver Nuggets', 'New York Knicks', 'New Orleans Pelicans', 'Minnesota Timberwolves', 'Milwaukee Bucks', 'Phoenix Suns', 'Brooklyn Nets', 'Los Angeles Lakers', 'Philadelphia 76ers']
        Teams = ['SAS','GSW','OKC','CLE','TOR','LAC','ATL','BOS','CHO','UTA','IND','MIA','POR','DET','HOU','DAL','WAS','CHI','ORL','MEM','SAC','DEN','NYK','NOP','MIN','MIL','PHO','BRK','LAL','PHI']
        #regular season data wrapping
        if guest_team not in Teams or home_team not in Teams or home_team==guest_team:
            Result = "Error input team names! Cannot be the same!"
            image = ""
            result="<h1>"+Result+"</h1>"
            return render_template('game_result.html',image=image,result=result)



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
        team_schedule = 'https://www.basketball-reference.com/teams/'+guest_team+'/'+str(season_input)+'_games.html'
        req = requests.get(team_schedule)
        text = BeautifulSoup(req.text, 'html.parser')
        stats = text.find('div',{'id': 'all_games'})
        cols = [i.get_text() for i in stats.thead.find_all('th')]

            # convert from unicode to string
        cols = [x.encode('UTF8') for x in cols]
        #print cols
        records_guest = []
        for i in stats.tbody.find_all('tr'):
                cols = [j.get_text() for j in i.find_all('td')]
                row_i = [x for x in cols]
                #store results in the records matrix
                if len(row_i)==14:
                    if len(row_i[6])==1:
                        result = int(row_i[6]=='W')

                        records_guest.append(result)

        guest_m = calc_momentum(records_guest)
        team_schedule = 'https://www.basketball-reference.com/teams/'+home_team+'/'+str(season_input)+'_games.html'
        req = requests.get(team_schedule)
        text = BeautifulSoup(req.text, 'html.parser')
        stats = text.find('div',{'id': 'all_games'})
        cols = [i.get_text() for i in stats.thead.find_all('th')]

            # convert from unicode to string
        cols = [x.encode('UTF8') for x in cols]
        #print cols
        records_home = []
        for i in stats.tbody.find_all('tr'):
                cols = [j.get_text() for j in i.find_all('td')]
                row_i = [x for x in cols]
                #store results in the records matrix
                if len(row_i)==14:
                    if len(row_i[6])==1:
                        result = int(row_i[6]=='W')

                        records_home.append(result)

        home_m = calc_momentum(records_home)
        x = np.array([PERAvg_test[index_0],PERAvg_test[index_1],guest_m,home_m]).reshape(1,-1)
        predicted = -3.44642028*PERAvg_test[index_0]+3.81685163*PERAvg_test[index_1]-0.16053294*guest_m+0.23453748*home_m-2.86179695417



        Result ="Hometeam Game Margin:"+str(predicted)+" <br/> "
        #####logistic regression for game winner prediction

        def model(x):
            return 1 / (1 + np.exp(-x))

        coef = [-0.51805045 ,0.57991114 ,-0.03056673 , 0.04023239]
        intercept =-0.532167425084

        prob = model(x[0][0] * coef[0] + x[0][1]*coef[1]+x[0][2]*coef[2]+x[0][3]*coef[3]+intercept)
        Result+= "Hometeam Winning Probability:"+str(prob)
        image = "<img src='/static/logos/"+home_team+".jpg' alt='Home Logo'style='width:200px;height:200px;''> <h1>vs.</h1> <img src='/static/logos/"+guest_team+".jpg' alt='Guest Logo'style='width:200px;height:200px;'>"
        result = "<h1>"+Result+"</h1>"
        ###plot momentum
        import matplotlib.pyplot as plt, mpld3
        h_m = []
        g_m = []
        for i in range(1,len(records_home)+1):
            cur = records_home[0:i]
            cur_m = calc_momentum(cur)
            h_m.append(cur_m)
        for i in range(1,len(records_guest)+1):
            cur = records_guest[0:i]
            cur_m = calc_momentum(cur)
            g_m.append(cur_m)
        fig = plt.figure()
        plt.plot(range(1,len(records_home)+1),h_m,'ro-',label=home_team)
        plt.plot(range(1,len(records_guest)+1),g_m,'bo-',label=guest_team)
        plt.legend()
        plt.xlabel('Game')
        plt.ylabel('Momentum')
        plt.ylim((-10.5,10.5))
        plt.title('Comparison of Team Momentums')
        p = mpld3.fig_to_html(fig)
        return render_template('game_result.html',image=image,result=result,plot=p)





if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')