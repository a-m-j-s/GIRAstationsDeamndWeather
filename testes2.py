from datetime import datetime
from datetime import timedelta
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import scipy.optimize as opt
import scipy.stats as st


####get data from gira corrected
    girafev=pandas.read_csv('gira.csv')
    #gira =gira.rename(columns={'doca_id': 'station'})
    girafev['date'] = pandas.to_datetime(girafev['date'])
    girajan = pandas.read_csv('gira_jan.csv')
    # gira =gira.rename(columns={'doca_id': 'station'})
    girajan['date'] = pandas.to_datetime(girajan['date'])

    meteo_dfs=[0]
    meteo_dfs[0]=pandas.read_csv('meteo.csv')
    meteo_dfs[0]['date']=pandas.to_datetime(meteo_dfs[0]['date'])



    frames=[girajan,girafev]
    gira=pandas.concat(frames)

    print(len(girajan))
    print(len(girafev))
    print(len(gira))

#####start and end times
    starttime = datetime(year=2019, month=2, day=1, hour=1)
    endtime = datetime(year=2019, month=2, day=28,hour=23)
    deltatime = timedelta(minutes=60)
    datelist = pandas.date_range(start=starttime, end=endtime, freq=deltatime)

###what code does
    Print = 0  #prints head(100) of time series
    plot = 0  #plots the time series
    analysismaxmin=0 #prints max min mean and std of time series
    analysispearson=0 #calculates the pearson correlation using a certain 2 hours for every 'useful' day
    analisisDCCA=0 #plots the coeficient form DCCA in function of the size of the box from which linear tendencies are removed
    analysisDCCAl=1 #plots the coeficient form DCCA-l in function of the size of the box from which non linear tendencies are removed


    gira_df_with_extra_collums = [] #creates list of dataframes

    for station in gira.station.unique(): #loop for every station in original dataframe
        gira = gira.sort_values(by='date')
        gira = gira.reset_index(drop=True)
        df = pandas.DataFrame(datelist, columns=['date'])
        #print(gira.loc[gira['station']==station].head(10))
        for date in datelist:
            df.at[df[df['date'] == date].index, 'num_bicicletas'] = \
            gira.loc[(gira['date'] <= date) & (gira['station'] == station)][-1:][
                'num_bicicletas'].values[0] #last value for the delta t

            #df.at[df[df['date'] == date].index, 'num_docas_vazias'] = \
            #gira.loc[(gira['date'] <= date) & (gira['station'] == station)][-1:][
            #    'num_docas_vazias'].values[0]

            aux = gira.loc[(gira['date'] <= date) & (gira['date'] > (date - deltatime))& (gira['station'] == station)] #gets datapoints in deltat
            aux.sort_values(by='date')
            aux = aux.reset_index(drop=True)
            #print(aux.head())
            nplus = 0
            nmin = 0

            #counts number of check in and out
            for ii in range(0, len(aux.index) - 1):
                auxn = aux.at[ii + 1, 'num_bicicletas'] - aux.at[ii, 'num_bicicletas']
                if auxn > 0:
                    nplus = nplus + auxn
                if auxn < 0:
                    nmin = nmin - auxn
            df.at[df[df['date'] == date].index, 'num_bicicletas_colocadas'] = nplus
            df.at[df[df['date'] == date].index, 'num_bicicletas_retiradas'] = nmin

        df = df.set_index('date')
        df.name = str(station)
        df.fillna(method='ffill', inplace=True)
        gira_df_with_extra_collums.append(df)

    #cretates new dataframe that is the sum
    dfaux = gira_df_with_extra_collums[0]
    for i in range(1, len(gira_df_with_extra_collums)):
        dfaux = pandas.DataFrame.add(self=dfaux, other=gira_df_with_extra_collums[i], axis='columns')
    dfaux.name = 'sum'
    gira_df_with_extra_collums.append(dfaux)

    ###METEREOLOGIA######
    # this portion of the code creates dataframes for the metreology data but with a constant delat t
    meteo_dfs_min = []
    for meteo_df in meteo_dfs:
        meteo_df.sort_values(by='date')
        meteo_df.reset_index(drop=True)
        df = pandas.DataFrame(datelist, columns=['date'])

        for date in datelist:
            df.at[df[df['date'] == date].index, 'humidity'] = \
                meteo_df.loc[meteo_df['date'] <= date][-1:]['humidade'].values[0]

            df.at[df[df['date'] == date].index, 'wind_intensity'] = \
                meteo_df.loc[meteo_df['date'] <= date][-1:]['intensidade_vento'].values[0]

            df.at[df[df['date'] == date].index, 'accumulated_precipitation'] = \
                meteo_df.loc[meteo_df['date'] <= date][-1:]['prec_acumulada'].values[0]

            df.at[df[df['date'] == date].index, 'temperature'] = \
                meteo_df.loc[meteo_df['date'] <= date][-1:]['temperatura'].values[0]

        df = df.set_index('date')
        df.fillna(method='ffill',inplace=True)
        df.name = str('579')
        meteo_dfs_min.append(df)

    if Print==1:
        # prints new dataframes
        for df in gira_df_with_extra_collums:
            print(df.name)
            print(df.head(100))

        for df in meteo_dfs:
            print(df.name)
            print(df.head(100))

    if plot==1:
        # plots dataframes
        N = {'407': 17, '417': 23, '416': 16, '406': 17, '408': 10, 'sum': 83}
        color = {'407': 'lightblue', '417': 'darkblue', '416': 'aquamarine', '406': 'coral', '408': 'teal',
                 'sum': 'lightgreen'}
        station = [407, 417, 416, 406, 408, 'sum']
        cmap = ListedColormap(['lightblue', 'darkblue', 'aquamarine', 'coral', 'teal', 'lightgreen'], name='cmap',
                              N=None)

        fig, axs = plt.subplots(nrows=6, figsize=(20, 10))
        i = 0
        for st in station:
            x = gira_df_with_extra_collums[i].index
            y = gira_df_with_extra_collums[i]['num_bicicletas_colocadas']
            z = gira_df_with_extra_collums[i]['num_bicicletas_retiradas']
            axs[i].plot(x, y, label=gira_df_with_extra_collums[i].name + ' check-in', color='lightblue', linewidth=2)
            axs[i].plot(x, z, label=gira_df_with_extra_collums[i].name + ' check-out', color='coral',
                        linewidth=2)
            axs[i].set_xlabel('Date')
            axs[i].set_ylabel('# Bikes')
            axs[i].set_yticks(np.arange(0, max(max(z), max(y)), int(max(max(z), max(y)) / 6)))
            axs[i].grid()
            axs[i].legend(loc=2)
            i = i + 1
        plt.show()
        fig.savefig('fev/delta60week.png')

        fig, axs = plt.subplots(nrows=6, figsize=(20, 10))
        i = 0
        for st in station:
            x = gira_df_with_extra_collums[i].index
            y = gira_df_with_extra_collums[i]['num_bicicletas']
            axs[i].plot(x, y, label=gira_df_with_extra_collums[i].name, color=color[gira_df_with_extra_collums[i].name],
                        linewidth=2)
            axs[i].axhline(y=N[gira_df_with_extra_collums[i].name], color='black')
            axs[i].set_xlabel('Date')
            axs[i].set_ylabel('# Bikes')
            axs[i].set_yticks(
                np.arange(0, N[gira_df_with_extra_collums[i].name], int(N[gira_df_with_extra_collums[i].name] / 6)))
            axs[i].grid()
            axs[i].legend(loc=2)
            i = i + 1
        plt.show()
        fig.savefig('fev/number60week.png')

        for df in meteo_dfs_min:
            fig = df.plot(kind='line', subplots=True, sharex=True, sharey=False, use_index=True, \
                          title='Station ' + df.name, colormap=cmap, legend=True, grid=True)[0].get_figure()
            fig.show()
            fig.savefig('fev/Station_' + df.name + '60week.png')

    if analysismaxmin==1:
        def integrated(dataframe):
            dict_st = dict()
            for df in dataframe:
                dictaux=dict()
                for col in df.columns:
                    aux = df[col].sum()
                    dictaux.update({col: aux})
                dict_st.update({df.name:dictaux})
            return dict_st
        def maximum(dataframe):
            dict_st = dict()
            for df in dataframe:
                dictaux=dict()
                for col in df.columns:
                    aux = df[col].max()
                    dictaux.update({col: aux})
                dict_st.update({df.name:dictaux})
            return dict_st
        def minimum(dataframe):
            dict_st = dict()
            for df in dataframe:
                dictaux=dict()
                for col in df.columns:
                    aux = df[col].min()
                    dictaux.update({col: aux})
                dict_st.update({df.name:dictaux})
            return dict_st
        def mean(dataframe):
            dict_st = dict()
            for df in dataframe:
                dictaux=dict()
                for col in df.columns:
                    aux = df[col].mean()
                    dictaux.update({col: aux})
                dict_st.update({df.name:dictaux})
            return dict_st
        def std(dataframe):
            dict_st = dict()
            for df in dataframe:
                dictaux=dict()
                for col in df.columns:
                    aux = df[col].std()
                    dictaux.update({col: aux})
                dict_st.update({df.name:dictaux})
            return dict_st

        #print(sum_number_bicicletas)
        #print(sum_number_colocadas)
        #print(sum_number_retiradas)
        #print(temperatura)
        #print(humiddade)
        #print(vento)
        #print(type(vento))


        dictgiraint=integrated(gira_df_with_extra_collums)
        dictgiramax = maximum(gira_df_with_extra_collums)
        dictgiramin = minimum(gira_df_with_extra_collums)
        dictgiramean = mean(gira_df_with_extra_collums)
        dictgirastd = std(gira_df_with_extra_collums)
        print(dictgiramax)
        print(dictgiramin)
        print(dictgiramean)
        print(dictgirastd)

        dictmeteoint = integrated(meteo_dfs_min)
        dictmeteomax = maximum(meteo_dfs_min)
        dictmeteomin = minimum(meteo_dfs_min)
        dictmeteomean = mean(meteo_dfs_min)
        dictmeteostd = std(meteo_dfs_min)
        print(dictmeteomax)
        print(dictmeteomin)
        print(dictmeteomean)
        print(dictmeteostd)

    if analisisDCCA==1:
        def recta(x,A,B):
            return A*x+B

        def Regression(list):
            listt = [ii for ii in range(0, len(list))]
            par, cov = opt.curve_fit(xdata=listt, ydata=list, f=recta)
            listy = [recta(t, par[0], par[1]) for t in listt]
            return listy

        def X(list):
            aux = 0
            mean=np.mean(list)
            listr=[]
            for iter in range(0, len(list)):
                aux = (list[iter]-mean) + aux
                listr.append(aux)
            return listr


        def f_DCCA_squared(list1, list2, leng, box):
            aux = 0
            print(list1)
            listaux1 = list1[box:box + leng]
            print(listaux1)
            listaux2 = list2[box:box + leng]
            listreg1 = Regression(listaux1)
            listreg2 = Regression(listaux2)
            for iter in range(0, leng):
                aux1 = listaux1[iter]-listreg1[iter]
                aux2 = listaux2[iter]-listreg2[iter]
                aux = aux + aux1 * aux2
            aux = aux / (leng)
            return aux


        def F_DCCA_squared(list1, list2, leng):
            N = len(list1)
            aux = 0
            for iter in range(0, N - leng):
                aux1 = f_DCCA_squared(list1, list2, leng, iter)
                aux = aux + aux1
            aux = aux / (N - leng)
            return aux


        def DCCA_coeff(list1, list2, leng):
            F11 = F_DCCA_squared(list1, list1, leng)
            F22 = F_DCCA_squared(list2, list2, leng)
            F12 = F_DCCA_squared(list1, list2, leng)
            return F12 / np.sqrt(F11 * F22)


        temperatura = meteo_dfs_min[0]['temperature'].values
        humiddade = meteo_dfs_min[0]['humidity'].values
        vento = meteo_dfs_min[0]['wind_intensity'].values
        prec=meteo_dfs_min[0]['accumulated_precipitation'].values
        n = np.geomspace(start=5, stop=int(len(temperatura)*2/3), num=30,dtype=int)
        fig=plt.figure()
        sigmatemp = []
        sigmawind=[]
        sigmahum=[]
        sigmapre=[]
        number_bike = gira_df_with_extra_collums[-1]['num_bicicletas'].values
        number_ret = gira_df_with_extra_collums[-1]['num_bicicletas_retiradas'].values
        number_col = gira_df_with_extra_collums[-1]['num_bicicletas_colocadas'].values
        Xnumber_ret=X(number_ret)
        Xnumber_col = X(number_col)
        Xvento = X(vento)
        Xhum=X(humiddade)
        Xprec=X(prec)
        Xtemp=X(temperatura)
        for i in n:
            sigmaw = DCCA_coeff(Xnumber_col,Xvento , i)
            sigmawind.append(sigmaw)
            sigmat = DCCA_coeff(Xnumber_col, Xtemp, i)
            sigmatemp.append(sigmat)
            sigmah = DCCA_coeff(Xnumber_col, Xhum, i)
            sigmahum.append(sigmah)
            sigmap = DCCA_coeff(Xnumber_col, Xprec, i)
            sigmapre.append(sigmap)
        plt.plot(n, sigmawind, label='wind_intensity',color='lightblue')
        plt.plot(n, sigmatemp, label='temperature',color='coral')
        plt.plot(n, sigmahum, label='humidity',color='darkblue')
        plt.plot(n, sigmapre, label='accumulated_precipitation',color='teal')
        plt.xlabel('n')
        plt.ylim(-1,1)
        plt.ylabel('$\sigma$')
        plt.grid()
        plt.title(gira_df_with_extra_collums[-1].name+'#check-in')
        plt.xscale('log')
        plt.legend()
        plt.show()
        plt.savefig('fev/sigmaret.png',bbox_inches='tight')

    if analysisDCCAl == 1:
        def Preg(list, l):
            listt = [ii for ii in range(0, len(list))]
            p = np.polyfit(x=listt, y=list, deg=l)
            pol = np.poly1d(p)
            listy = [pol(t) for t in listt]
            return listy

        def X(list):
            aux = 0
            mean = np.mean(list)
            listr = []
            for iter in range(0, len(list)):
                aux = (list[iter] - mean) + aux
                listr.append(aux)
            return listr

        def f_DCCA_squared_pol(list1, list2, leng, box, l):
            aux = 0
            listaux1 = list1[box:box + leng]
            listaux2 = list2[box:box + leng]
            listreg1 = Preg(listaux1,l)
            listreg2 = Preg(listaux2,l)
            for iter in range(0, leng):
                aux1 = listaux1[iter] - listreg1[iter]
                aux2 = listaux2[iter] - listreg2[iter]
                aux = aux + aux1 * aux2
            aux = aux / (leng)
            return aux

        def F_DCCA_squared_pol(list1, list2, leng, l):
            N = len(list1)
            aux = 0
            for iter in range(0, N - leng):
                aux1 = f_DCCA_squared_pol(list1, list2, leng, iter, l)
                aux = aux + aux1
            aux = aux / (N - leng)
            return aux

        def DCCA_coeff_pol(list1, list2, leng):
            if leng<=4:
                l=1
            if leng<=6 and leng>4:
                l=2
            if leng<=10 and leng>6:
                l=4
            if leng<=18 and leng>10:
                l=8
            if leng>18:
                l=16
            #l=int(16*leng/len(list1))
            F11 = F_DCCA_squared_pol(list1, list1, leng,l)
            F22 = F_DCCA_squared_pol(list2, list2, leng,l)
            F12 = F_DCCA_squared_pol(list1, list2, leng,l)
            return F12 / np.sqrt(F11 * F22)


        temperatura = meteo_dfs_min[0]['temperature'].values
        humiddade = meteo_dfs_min[0]['humidity'].values
        vento = meteo_dfs_min[0]['wind_intensity'].values
        prec = meteo_dfs_min[0]['accumulated_precipitation'].values
        n = np.geomspace(start=5, stop=int(len(temperatura)*2/3), num=20, dtype=int)
        fig = plt.figure()
        sigmatemp = []
        sigmawind = []
        sigmahum = []
        sigmapre = []
        number_bike = gira_df_with_extra_collums[-1]['num_bicicletas'].values
        number_ret = gira_df_with_extra_collums[-1]['num_bicicletas_retiradas'].values
        number_col = gira_df_with_extra_collums[-1]['num_bicicletas_colocadas'].values
        Xn_col=X(number_col)
        Xn_ret=X(number_ret)
        Xtemp=X(temperatura)
        Xhum=X(humiddade)
        Xvento=X(vento)
        Xprec=X(prec)
        #print(number_col)
        #print(Xn_col)
        #print(len(number_col))
        #print(len(Xn_col))
        for i in n:
            sigmaw = DCCA_coeff_pol(Xn_ret, Xvento, i)
            sigmawind.append(sigmaw)
            sigmat = DCCA_coeff_pol(Xn_ret, Xtemp, i)
            sigmatemp.append(sigmat)
            sigmah = DCCA_coeff_pol(Xn_ret, Xhum, i)
            sigmahum.append(sigmah)
            sigmap = DCCA_coeff_pol(Xn_ret, Xprec, i)
            sigmapre.append(sigmap)
        plt.plot(n, sigmawind, label='wind_intensity', color='lightblue')
        plt.plot(n, sigmatemp, label='temperature', color='coral')
        plt.plot(n, sigmahum, label='humidity', color='darkblue')
        plt.plot(n, sigmapre, label='accumulated_precipitation', color='teal')
        plt.xlabel('n')
        plt.ylim(-1, 1)
        plt.ylabel('$\sigma$')
        plt.grid()
        plt.title(gira_df_with_extra_collums[-1].name + '# check-out')
        plt.xscale('log')
        plt.legend()
        plt.show()
        plt.savefig('1-2-28-2/sigmabike.png', bbox_inches='tight')

    if analysispearson ==1:
        def pearsoncorr(x,y):
            xmean=np.mean(x)
            ymean=np.mean(y)
            coef1=0
            coef2=0
            coef3=0
            for i in range(0,len(x)):
                coef1=coef1+(x[i]-xmean)*(y[i]-ymean)
                coef2=coef2+(x[i]-xmean)*(x[i]-xmean)
                coef3=coef3+(y[i]-ymean)*(y[i]-ymean)
            if coef2==0 or coef3==0:
                print('division by zero')
                return float('nan')
            return coef1/np.sqrt(coef2*coef3)

        corrtemp = dict()
        corrhum = dict()
        corrvento = dict()
        corrpre=dict()
        for df in gira_df_with_extra_collums:
            bike=[]
            ret=[]
            col=[]
            temp=[]
            prec=[]
            wind=[]
            hum=[]
            dayfev=[1,4,5,6,7,8,11,12,13,14,15,18,19,20,21,22,25,26,27,28]
            #dayjan=[1,2,3,4,7,8,10,11,15,16,17,18,23,24,25,28,29,30]
            #day=[[1,i] for i in dayjan]+[[2,i] for i in dayfev]
            day=[[2,i] for i in dayfev]
            for i in day:
                dateaux = datetime(year=2019, month=i[0], day=i[1])
                temp.append(np.mean(meteo_dfs_min[0].loc[(df.index.hour <= 16) & (df.index.hour >= 14) & (df.index.day == dateaux.day)]['temperature'].values))
                #print(meteo_dfs_min[0].loc[(df.index.hour <= 16) & (df.index.hour >= 14) & (df.index.day == dateaux.day)]['temperature'].values)
                #temp.append(np.mean(meteo_dfs_min[0].loc[(df.index.hour <= 13) & (df.index.hour >= 11) & (df.index.day == dateaux.day)]['temperature'].values))
                prec.append(np.mean( meteo_dfs_min[0].loc[(df.index.hour <= 16) & (df.index.hour >= 14) & (df.index.day == dateaux.day)]['accumulated_precipitation'].values))
                #prec.append(np.mean(meteo_dfs_min[0].loc[(df.index.hour <= 13) & (df.index.hour >= 11) & (df.index.day == dateaux.day)]['accumulated_precipitation'].values))
                wind.append(np.mean(meteo_dfs_min[0].loc[(df.index.hour <= 16) & (df.index.hour >= 14) & (df.index.day == dateaux.day)]['wind_intensity'].values))
                #wind.append(np.mean(meteo_dfs_min[0].loc[(df.index.hour <= 13) & (df.index.hour >= 11) & (df.index.day == dateaux.day)]['wind_intensity'].values))
                hum.append(np.mean(meteo_dfs_min[0].loc[(df.index.hour <= 16) & (df.index.hour >= 14) & (df.index.day == dateaux.day)]['humidity'].values))
                #hum.append(np.mean(meteo_dfs_min[0].loc[(df.index.hour <= 13) & (df.index.hour >= 11) & (df.index.day == dateaux.day)]['humidity'].values))
                bike.append(np.mean(df.loc[(df.index.hour <= 16) & (df.index.hour >= 14) & (df.index.day == dateaux.day)]['num_bicicletas'].values))
                #bike.append(np.mean(df.loc[(df.index.hour <= 13) & (df.index.hour >= 11) & (df.index.day == dateaux.day)]['num_bicicletas'].values))
                ret.append(np.mean(df.loc[(df.index.hour <= 16) & (df.index.hour >= 14) & (df.index.day == dateaux.day)]['num_bicicletas_retiradas'].values))
                #ret.append(np.mean(df.loc[(df.index.hour <= 13) & (df.index.hour >= 11) & (df.index.day == dateaux.day)]['num_bicicletas_retiradas'].values))
                col.append(np.mean(df.loc[(df.index.hour <= 16) & (df.index.hour >= 14) & (df.index.day == dateaux.day)]['num_bicicletas_colocadas'].values))
                #col.append(np.mean(df.loc[(df.index.hour <= 13) & (df.index.hour >= 11) & (df.index.day == dateaux.day)]['num_bicicletas_colocadas'].values))

            corrtemp[df.name]=[pearsoncorr(bike,temp),pearsoncorr(ret,temp),pearsoncorr(col,temp)]
            corrvento[df.name] = [pearsoncorr(bike, wind), pearsoncorr(ret, wind), pearsoncorr(col, wind)]
            corrhum[df.name] = [pearsoncorr(bike, hum), pearsoncorr(ret, hum), pearsoncorr(col, hum)]
            corrpre[df.name] = [pearsoncorr(bike, prec), pearsoncorr(ret, prec), pearsoncorr(col, prec)]

            fig, axs = plt.subplots(nrows=4, figsize=(20, 30))
            #axs[0].plot(bike,temp,'.',markersize=5 ,color='teal',label='Bike')
            axs[0].plot(ret, temp,'.',markersize=10 , color='darkblue', label='check-out')
            axs[0].plot(col, temp,'.',markersize=10, color='coral', label='check-in')
            axs[0].legend()
            axs[0].set_xlabel('#')
            axs[0].set_ylabel('Temperature')
            axs[0].set_title(df.name)
            #axs[1].plot(bike, hum,'.',markersize=5 , color='teal', label='Bike')
            axs[1].plot(ret, hum,'.',markersize=10 , color='darkblue', label='check-out')
            axs[1].plot(col, hum,'.',markersize=10 , color='coral', label='check-in')
            axs[1].legend()
            axs[1].set_xlabel('#')
            axs[1].set_ylabel('Humidity')
            axs[1].set_title(df.name)
            #axs[2].plot(bike, prec,'.',markersize=5 , color='teal', label='Bike')
            axs[2].plot(ret, prec,'.',markersize=10 , color='darkblue', label='check-out')
            axs[2].plot(col, prec,'.',markersize=10 , color='coral', label='check-in')
            axs[2].legend()
            axs[2].set_xlabel('#')
            axs[2].set_ylabel('Accumulated precipitation')
            axs[2].set_title(df.name)
            #axs[3].plot(bike, wind,'.',markersize=5 , color='teal', label='Bike')
            axs[3].plot(ret, wind,'.',markersize=10, color='darkblue', label='check-out')
            axs[3].plot(col, wind,'.',markersize=10 , color='coral', label='check-in')
            axs[3].legend()
            axs[3].set_xlabel('#')
            axs[3].set_ylabel('Wind Intensity')
            axs[3].set_title(df.name)
            fig.savefig('fev/'+df.name+'plots601416.png')

            print(df.name)
            print('bikes, temp, prec, wind, hum ')
            print('n_bike, {},{},{},{}'.format(corrtemp[df.name][0],corrpre[df.name][0],corrvento[df.name][0],corrhum[df.name][0]))
            print('n_ret, {},{},{},{}'.format(corrtemp[df.name][1], corrpre[df.name][1], corrvento[df.name][1],
                                               corrhum[df.name][1]))
            print('n_col, {},{},{},{}'.format(corrtemp[df.name][2], corrpre[df.name][2], corrvento[df.name][2],
                                               corrhum[df.name][2]))



