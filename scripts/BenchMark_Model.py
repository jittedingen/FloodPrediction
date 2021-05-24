import math
import sys
import pandas as pd
import numpy as np
import os
import datetime as dt   # Python standard library datetime  module
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def calc_performance_scores(obs, pred):
    f1 = f1_score(obs, pred)
    misses = sum((obs == 1) & (pred == 0))
    hits = sum((obs ==1) & (pred==1))
    false_alarms = sum((obs==0) & (pred==1))
    corr_neg = sum((obs==0) & (pred == 0))

    output = {}
    output['f1'] = f1
    output['hits'] = hits
    output['false_alarms'] = false_alarms
    output['misses'] = misses
    output['corr_neg'] = corr_neg

    if (hits + misses) == 0:
        HM = 0.0001
    else:
        HM = hits + misses

    if (hits + false_alarms) == 0:
        HF = 0.0001
    else:
        HF = hits + false_alarms

    if (false_alarms + corr_neg) == 0:
        FC = 0.0001
    else:
        FC = false_alarms + corr_neg

    if (hits + false_alarms + misses) == 0:
        HFM = 0.0001
    else:
        HFM = hits + false_alarms + misses

    output['pod'] = hits / (HM)
    output['far'] = false_alarms / (HF)
    output['pofd'] = false_alarms / (FC)
    output['csi'] = hits / (HFM)

    output = pd.Series(output)
    return output
#######################################################################

predictor = 'max_dt_3days' #max_dt_3days / max_future_3days / max_future_5days / max_future_7days
n_ahead = 1
loss = 'f1'
n_folds = 5

## Load the (downloaded) data
my_local_path = str(Path(os.getcwd()))
df_model = pd.read_csv(my_local_path + '/df_total_mediumqual.csv')

performance_scores = pd.DataFrame()
performance_SE = pd.DataFrame()

# loop over districts
for district in df_model.district.unique():
    df_district = df_model[df_model['district'] == district]
    performance_model = pd.DataFrame(columns = ['parameters', 'district'])

    # loop over stations and test all possible quantiles
    station_SE = pd.DataFrame()
    station_scores = pd.DataFrame()
    for station in df_district['station'].unique():
        df_station = df_district[df_district['station'] == station]

        # Continue if the number of floods is lower than 5
        n_floods = df_station['flood'].sum()
        if n_floods < n_folds:
            print("Too little flood events in district "+str(district))
            too_little = True
            continue
        else:
            too_little = False

        ### modify the actual and shift the data to predict ahead
        if 'future' in predictor:
            window_len = int(predictor[11])
            df_station['flood_old'] = df_station['flood']
            df_station['flood'] = df_station['flood'].rolling(window_len).max().shift(periods=-(window_len+n_ahead-1))
            if n_ahead > 1:
                df_station[predictor] = df_station[predictor].shift(periods=-(n_ahead-1))
            df_station = df_station.dropna()

        ##### Make 5 folds and iterate over all of those to get an average performance
        floods_perblock = int(n_floods // n_folds)
        floods_mod = int(n_floods % n_folds) #get the remainder
        flood_ix = df_station.index[df_station['flood'] == 1].tolist()

        if floods_mod == 0:
            fold1 = df_station.loc[0:(flood_ix[floods_perblock-1])]
            fold2 = df_station.loc[((flood_ix[floods_perblock-1]) + 1):(flood_ix[floods_perblock*2-1])]
            fold3 = df_station.loc[((flood_ix[floods_perblock*2-1]) + 1):(flood_ix[floods_perblock*3-1])]
            fold4 = df_station.loc[((flood_ix[floods_perblock*3-1]) + 1):(flood_ix[floods_perblock*4-1])]
            fold5 = df_station.loc[((flood_ix[floods_perblock*4-1]) + 1):]

        elif floods_mod == 1:
            fold1 = df_station.loc[0:(flood_ix[floods_perblock-1])]
            fold2 = df_station.loc[((flood_ix[floods_perblock-1]) + 1):(flood_ix[floods_perblock*2-1])]
            fold3 = df_station.loc[((flood_ix[floods_perblock*2-1]) + 1):(flood_ix[floods_perblock*3-1])]
            fold4 = df_station.loc[((flood_ix[floods_perblock*3-1]) + 1):(flood_ix[floods_perblock*4-1])]
            fold5 = df_station.loc[((flood_ix[floods_perblock*4-1]) + 1):] # this fold has floods_perblock + 1 floods

        elif floods_mod == 2:
            fold1 = df_station.loc[0:(flood_ix[floods_perblock])] #this one has two floods
            fold2 = df_station.loc[((flood_ix[floods_perblock]) + 1):(flood_ix[floods_perblock*2+1])] #this one has floods_perblock + 1 floods
            fold3 = df_station.loc[((flood_ix[floods_perblock*2+1]) + 1):(flood_ix[floods_perblock*3+1])]
            fold4 = df_station.loc[((flood_ix[floods_perblock*3+1]) + 1):(flood_ix[floods_perblock*4+1])]
            fold5 = df_station.loc[((flood_ix[floods_perblock*4+1]) + 1):]

        elif floods_mod == 3:
            fold1 = df_station.loc[0:(flood_ix[floods_perblock])]  # this one has floods_perblock + 1 floods
            fold2 = df_station.loc[((flood_ix[floods_perblock]) + 1):(flood_ix[floods_perblock * 2 + 1])]  # this one has floods_perblock + 1 floods
            fold3 = df_station.loc[((flood_ix[floods_perblock * 2 + 1]) + 1):(flood_ix[floods_perblock * 3 + 2])] # this one has floods_perblock + 1 floods
            fold4 = df_station.loc[((flood_ix[floods_perblock * 3 + 2]) + 1):(flood_ix[floods_perblock * 4 + 2])]
            fold5 = df_station.loc[((flood_ix[floods_perblock * 4 + 2]) + 1):]

        elif floods_mod == 4:
            fold1 = df_station.loc[0:(flood_ix[floods_perblock])]  # this one has floods_perblock + 1 floods
            fold2 = df_station.loc[((flood_ix[floods_perblock]) + 1):(flood_ix[floods_perblock * 2 + 1])]  # this one has floods_perblock + 1 floods
            fold3 = df_station.loc[((flood_ix[floods_perblock * 2 + 1]) + 1):(flood_ix[floods_perblock * 3 + 2])] # this one has floods_perblock + 1 floods
            fold4 = df_station.loc[((flood_ix[floods_perblock * 3 + 2]) + 1):(flood_ix[floods_perblock * 4 + 3])] #  floods_perblock + 1 floods
            fold5 = df_station.loc[((flood_ix[floods_perblock * 4 + 3]) + 1):]

        # Repeat for several folds to get an average performance
        performance_folds = pd.DataFrame()
        all_folds = [fold1, fold2, fold3, fold4, fold5]
        for fold in range(0, 5):
            test = all_folds[fold]
            train = df_station[~df_station['time'].isin(test['time'])]

            # Train the model with the training set (on the position of train it was df_station before)
            train['time'] = pd.to_datetime(train['time'])
            extreme_dis = train.set_index('time')['max_dt_3days'].groupby(pd.Grouper(freq='6M')).max()
            for q in range(50, 100):
                threshold = extreme_dis.quantile(q/100)
                train['predictions'] = np.where((train[predictor] >= threshold), 1, 0)

                perf = train.groupby(['district', 'station']).\
                   apply(lambda row: calc_performance_scores(row['flood'], row['predictions']))
                perf['parameters'] = str((station, str(q)))
                perf['district'] = district

                performance_model = performance_model.append(perf, ignore_index=True)

            # Now use the best station and quantile to make prediction on the test data
            if performance_model[loss].count() != 0: #counts the number of NaN
                best_quantile = performance_model.iloc[performance_model[loss].idxmax][0].split(',')[1]

            else:
                best_quantile = performance_model.iloc[performance_model['hits'].idxmax][0].split(',')[1]

            #remove backslashes and brackets from strings
            best_quantile = int(best_quantile.replace('\'', "").replace(')', ""))

            # Test on the test set of this iteration
            threshold = extreme_dis.quantile(best_quantile / 100)
            test['predictions'] = np.where((test[predictor] >= threshold), 1, 0)
            best_score = calc_performance_scores(test['flood'], test['predictions'])

            # Save this performance for this fold
            performance_folds = performance_folds.append([[station,
                                                           best_score['f1'], best_score['hits']/len(test),
                                                           best_score['misses']/len(test), best_score['false_alarms']/len(test),
                                                           best_score['corr_neg']/len(test), best_score['csi'],
                                                           best_score['pod'], best_score['far'],
                                                           best_score['pofd'], test['flood'].sum()]],
                                                         ignore_index=True)


        # Get the average performance score for this station
        station_scores = station_scores.append(performance_folds.mean().append(pd.Series(station)), ignore_index=True)
        station_SE = station_SE.append(performance_folds.std().append(pd.Series(station)), ignore_index=True)

    if too_little == True:
        continue

    # Add column names
    station_scores.columns = ['station', 'f1', 'hits', 'misses', 'false_alarms', 'corr_neg', 'csi', 'pod', 'far', 'pofd', 'n_floods']
    station_SE.columns = ['station', 'f1', 'hits', 'misses', 'false_alarms', 'corr_neg', 'csi', 'pod', 'far', 'pofd', 'n_floods']

    # Get the best performing station and its cross-validated performance
    best_station = station_scores.iloc[station_scores[loss].idxmax]['station']
    best_score = station_scores[station_scores['station'] == best_station]
    best_SE = station_SE[station_SE['station'] == best_station]

    # save performance of all folds and best station
    performance_scores = performance_scores.append([[district, best_station,
                                                             best_score['f1'].iloc[0], best_score['hits'].iloc[0],
                                                             best_score['misses'].iloc[0], best_score['false_alarms'].iloc[0],
                                                             best_score['corr_neg'].iloc[0], best_score['csi'].iloc[0],
                                                             best_score['pod'].iloc[0], best_score['far'].iloc[0],
                                                             best_score['pofd'].iloc[0], best_score['n_floods'].iloc[0]]], ignore_index=True)

    #change this to SE
    performance_SE = performance_SE.append([[district, best_station,
                                                             best_SE['f1'].iloc[0], best_SE['hits'].iloc[0],
                                                             best_SE['misses'].iloc[0], best_SE['false_alarms'].iloc[0],
                                                             best_SE['corr_neg'].iloc[0], best_SE['csi'].iloc[0],
                                                             best_SE['pod'].iloc[0], best_SE['far'].iloc[0],
                                                             best_SE['pofd'].iloc[0], best_SE['n_floods'].iloc[0]]], ignore_index=True)



# Add column names to the performances
performance = pd.DataFrame(performance_scores)
performance.columns = ['district', 'best_station', 'f1', 'hits', 'misses', 'false_alarms', 'corr_neg', 'csi', 'pod', 'far', 'pofd', 'floods_intest']

performance_SE = pd.DataFrame(performance_SE)
performance_SE.columns = ['district', 'best_station', 'f1', 'hits', 'misses', 'false_alarms', 'corr_neg', 'csi', 'pod', 'far', 'pofd', 'floods_intest']

# Save performance
performance.to_csv(str(Path(os.getcwd()))+"/Results/Benchmark/performance_OriginalBenchM_NEW.csv")
performance_SE.to_csv(str(Path(os.getcwd()))+"/Results/Benchmark/performance_OriginalBenchM_NEW_SE.csv")

print('done')