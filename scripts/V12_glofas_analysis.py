# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:24:03 2019

@authors: ABucherie, JMargutti
"""
# this script objective is to extract and analyse Glofas historical data for specific stations, against flood impact events at district level
# and to compute the prediction performance of a model using only Glofas discharge thresholds.

# import necessary modules
import math
import sys
sys.path.append("scripts")
import xarray as xr
import pandas as pd
import numpy as np
import os
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import datetime as dt   # Python standard library datetime  module
from database_utils import get_glofas_data # utility functions to access database
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
import dateutil.relativedelta
import matplotlib as plt
from sklearn.metrics import f1_score
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

def normalize(df):
    """
    normalize dataframe
    """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def prepare_glofas_stations_data(country='Uganda', path='.'):
    """
    Read the path to the Africa .csv file of all glofas virtual stations list of Africa
     and select only the one for the country/ save the table as .csv
    """
    Gl_stations = pd.read_csv(path + '/africa/glofas/Glofaspoints_Africa_510.csv')  # do not change
    Gl_stations = Gl_stations[Gl_stations['CountryNam'] == country]
    Gl_stations['station'] = Gl_stations['ID']
    Gl_stations = Gl_stations[['ID', 'station', 'Stationnam', 'CountryNam', 'XCorrected', 'YCorrected']].set_index(
        'ID').rename(
        columns={'Stationnam': 'location', 'CountryNam': 'Country', 'XCorrected': 'lon', 'YCorrected': 'lat'})
    Gl_stations = Gl_stations[['Country', 'station', 'location', 'lon', 'lat']]
    return Gl_stations


def prepare_dataframe_stations_discharge(GloFAS_stations, GloFAS_data, path):
    """
        Create dataframe of glofas stations, daily discharge and max-over-3-days discharge
    """
    df_discharge = pd.DataFrame(columns=['station', 'time', 'dis', 'max_dt_3days', 'max_future_3days',
                                         'max_future_4days', 'max_future_5days', 'max_future_6days', 'max_future_7days'])
    # loop over stations
    for station in np.unique(GloFAS_stations['station']):
        # For the selected glofas station, extract the glofas discharge from the glofas Grid  coordinate and save in a dictionary
        Longitude = GloFAS_stations[GloFAS_stations['station'] == station].lon
        Latitude = GloFAS_stations[GloFAS_stations['station'] == station].lat
        nc_loc = GloFAS_data.sel(lon=Longitude, lat=Latitude, method='nearest').rename({'dis24': 'dis'})
        station_discharge = nc_loc.dis

        # Extract the daily discharge time-series station data (and save in a .csv per station if needed)
        df_dis = pd.DataFrame(columns=['station', 'time', 'dis', 'max_dt_3days', 'max_future_3days',
                                       'max_future_4days', 'max_future_5days', 'max_future_6days', 'max_future_7days'])
        df_dis['time'] = station_discharge['time'].values
        df_dis['dis'] = pd.Series(station_discharge.values.flatten())
        df_dis['station'] = station
        # if needed / step for saving all station discharge time serie in .csv
        df_dis[['station', 'time', 'dis']].to_csv(path + 'input/Glofas/station_csv/GLOFAS_data_for_%s.csv' % station)

        # 4- Create a dataframe with all daily discharge time-series station data, together with the daily lag of + and - 3 days
        df_dis['max_dt_3days'] = df_dis.dis.rolling(7, min_periods=3, center=True).max()

        # 5 - Create a new column which calculates the maximum in the upcoming 3 days
        df_dis['max_future_3days'] = df_dis.dis.rolling(3).max().shift(periods = -3)
        df_dis['max_future_4days'] = df_dis.dis.rolling(4).max().shift(periods = -4)
        df_dis['max_future_5days'] = df_dis.dis.rolling(5).max().shift(periods = -5)
        df_dis['max_future_6days'] = df_dis.dis.rolling(6).max().shift(periods = -6)
        df_dis['max_future_7days'] = df_dis.dis.rolling(7).max().shift(periods = -7)

        df_discharge = df_discharge.append(df_dis, ignore_index=True)
    return df_discharge


def prepare_rainfall_data(df, n_timesteps=10):

    df[['Year', 'Month', 'Day']] = df[['Year', 'Month', 'Day']].astype(int)
    df['time'] = pd.to_datetime((df.Year*10000+df.Month*100+df.Day).apply(str), format='%Y-%m-%d')
    df = df.drop(columns=['Year', 'Month', 'Day', 'PCODE'])
    df = df.rename(columns={'District': 'district'})
    df.district = df.district.str.lower()
    df = df.groupby(['district', 'time']).first()

    for t in range(0, n_timesteps):
        df['rainfall_max_' + str(t)] = np.nan
        df['rainfall_cum_' + str(t)] = np.nan

    for ix, row in df.reset_index().iterrows():
        date_start = pd.to_datetime(row['time'])
        for t in range(0, n_timesteps):
            date_old = pd.to_datetime(date_start - dateutil.relativedelta.relativedelta(days=t))
            try:
                data_old = df.loc[(row['district'], date_old):(row['district'], date_start)]
                max_old = data_old['max'].max()
                cum_old = data_old['mean'].sum()
                df.at[(row['district'], date_start), 'rainfall_max_' + str(t)] = max_old
                df.at[(row['district'], date_start), 'rainfall_cum_' + str(t)] = cum_old
            except:
                continue
    df = df.reset_index()
    df = df.drop(columns=['max', 'mean'])
    return df


def get_impact_data(date_format, admin_column, ct_code, path):
    flood_events = pd.read_csv(path + 'input/%s_impact_data.csv' % ct_code, encoding='latin-1')
    flood_events['Date'] = pd.to_datetime(flood_events['Date'], format=date_format)
    flood_events = flood_events.query("Date >= '2000-01-01' ")
    flood_events = flood_events[['Date', admin_column, 'flood']].drop_duplicates()\
        .rename(columns={admin_column: 'district'}).dropna().set_index('Date')
    # possibility to filter on flood event certainty/impact severity column for Uganda instead of previous line
    #flood_events = flood_events[['Date', Admin_column,'Certainty', 'Impact', 'flood']].\
    # drop_duplicates().rename(columns={Admin_column: 'district'}).dropna().set_index('Date')
    #flood_events= flood_events[flood_events['Certainty'] > 6]
    flood_events['district'] = flood_events['district'].str.lower()
    flood_events = flood_events.reset_index().rename(columns={'Date': 'time'})
    return flood_events


def calc_performance_scores(obs, pred):
    """
    compute confusion matrix (hits, false_al, misses, correct negatives) and performance indexes (FAR, POD, POFD, CSI)):
    Methodology adapted taking into account the consecutive day above thresholds as a unique flood period
    hits:              nb of peak period above thresholds that have at least one observation day within the period
    false alarm :      number of peak above threshold(consecutive day above discharge threshold as an event), minus the number of hits
    misses :           number of observed flood events no in a discharge peak period o above threshold
    correct negative : forcing the correct negative number to be the same than the number of observed flood events (misses + hits)
    """
    # print(obs, pred)
    #df = pd.DataFrame({'cons_class': pred.diff().ne(0).cumsum(), 'hits': (obs == 1) & (pred == 1)})
    #hits = df.hits[df.hits].count()
    #false_al = (pred.loc[pred.shift() != pred].sum()) - hits
    #if false_al < 0:
     #   false_al = 0
    #misses = sum((obs == 1) & (pred == 0))
    #corr_neg = misses + hits
    #output['hits'] = hits
    #output['misses'] = misses
    #output['false_alarms'] = false_al
    #output['corr_neg'] = corr_neg

    ##### Newly added by Jitte
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


def train_test_model(df_model,
                     predictor='max_future_3days',
                     model_type='quantile_discharge',
                     loss='f1', n_ahead = 1):

    performance_scores = pd.DataFrame()

    # loop over districts
    for district in df_model.district.unique():
        df_district = df_model[df_model['district'] == district]
        n_stations = len(df_district['station'].unique())
        if df_district['flood'].sum()/n_stations < 2:
            continue

        if model_type == 'quantile_discharge':
            performance_model = pd.DataFrame(columns = ['parameters', 'district'])

            # loop over stations and test all possible quantiles
            for station in df_district['station'].unique():
                df_station = df_district[df_district['station'] == station]

                ### Added by Jitte - modify the actual and shift the data to predict ahead
                if 'future' in predictor:
                    window_len = int(predictor[11])
                    df_station['flood_old'] = df_station['flood']
                    df_station['flood'] = df_station['flood'].rolling(window_len).max().shift(periods=-(window_len+n_ahead-1))
                    if n_ahead > 1:
                        df_station[predictor] = df_station[predictor].shift(periods=-(n_ahead-1))
                    df_station = df_station.dropna()

                ### ADDED BY JITTE - SPLIT INTO TRAIN AND TEST FOR BETTER COMPARISON WITH OTHER MODELS
                n_floods = df_station['flood'].sum()
                flood_points = df_station.index[df_station['flood'] == 1].tolist()
                train_points = flood_points[:int(n_floods - math.ceil(0.2*n_floods))]
                train = df_station.loc[:train_points[-1]]

                # Train the model with the training set (on the position of train it was df_station before)
                extreme_dis = train.set_index('time')['max_dt_3days'].groupby(pd.Grouper(freq='6M')).max()
                for q in range(50, 100):
                    threshold = extreme_dis.quantile(q/100)
                    train['predictions'] = np.where((train[predictor] >= threshold), 1, 0)

                    perf = train.groupby(['district', 'station']).\
                       apply(lambda row: calc_performance_scores(row['flood'], row['predictions']))
                    perf['parameters'] = str((station, str(q)))
                    perf['district'] = district

                    performance_model = performance_model.append(perf, ignore_index=True)

            # ADDED BY JITTE - Now use the best station and quantile to make prediction on the test data
            if performance_model[loss].count() != 0: #counts the number of NaN
                best_station = performance_model.iloc[performance_model[loss].idxmax][0].split(',')[0]
                best_quantile = performance_model.iloc[performance_model[loss].idxmax][0].split(',')[1]

            else:
                best_station = performance_model.iloc[performance_model['hits'].idxmax][0].split(',')[0]
                best_quantile = performance_model.iloc[performance_model['hits'].idxmax][0].split(',')[1]

            #remove backslashes and brackets from strings
            best_station = best_station.replace('\'', "").replace('(', "")
            best_quantile = int(best_quantile.replace('\'', "").replace(')', ""))

            # Calculate performance on the test set
            df_station = df_district[df_district['station'] == best_station]
            if 'future' in predictor:
                window_len = int(predictor[11])
                df_station['flood_old'] = df_station['flood']
                df_station['flood'] = df_station['flood'].rolling(window_len).max().shift(
                    periods=-(window_len + n_ahead - 1))
                if n_ahead > 1:
                    df_station[predictor] = df_station[predictor].shift(periods=-(n_ahead - 1))
                df_station = df_station.dropna()

            n_floods = df_station['flood'].sum()
            flood_points = df_station.index[df_station['flood'] == 1].tolist()
            bound = int(n_floods - math.ceil(0.2 * n_floods))
            train_points = flood_points[:bound]

            train = df_station.loc[:train_points[-1]]
            test = df_station.loc[(train_points[-1]+1):]
            extreme_dis = train.set_index('time')['max_dt_3days'].groupby(pd.Grouper(freq='6M')).max()

            threshold = extreme_dis.quantile(best_quantile / 100)
            test['predictions'] = np.where((test[predictor] >= threshold), 1, 0)

            #check = test[['time', 'dis', 'flood_old', 'flood', 'predictions', predictor]]

            best_score = calc_performance_scores(test['flood'], test['predictions'])


            #### ADDED BY JITTE: Try to calculate ROC Curve and plot it and AUCROC
            #TPR = []
            #FPR = []
            #for q in range(0,100, 2):
                #threshold = extreme_dis.quantile(q / 100)
                #test['predictions'] = np.where((test[predictor] >= threshold), 1, 0)
                #scores = f1_score(test['flood'], test['predictions'])
                #TP_rate = scores['hits']/(scores['hits'] + scores['misses'])
                #FP_rate = scores['false_alarms'] / (scores['false_alarms'] + scores['corr_neg'])

                #TPR.append(TP_rate)
                #FPR.append(FP_rate)

            #auc_score = auc(FPR, TPR)
            #auc_score = -1 * np.trapz(TPR, FPR)
            #plt.plot(FPR, TPR, linestyle='-', color='darkorange', lw=2, label='ROC curve', clip_on=False)
            #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.0])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC curve, AUC = %.2f' % auc_score)
            #plt.legend(loc="lower right")
            #plt.savefig(str(Path(os.getcwd()))+"/Results/Benchmark/Figures/ROCAUC_" + district + '.png')
            #plt.clf() #clear this figure for upcoming districts

            # save performance
            performance_scores = performance_scores.append([[district,
                                                             best_score['f1'], best_score['hits'],
                                                             best_score['misses'], best_score['false_alarms'],
                                                             best_score['corr_neg'], best_score['csi'],
                                                             best_score['pod'], best_score['far'],
                                                             best_score['pofd'], len(test), test['flood'].sum()]], ignore_index=True)


        elif model_type == 'bdt_discharge':
            # prepare training data
            X, y = [], []
            df_ordered = df_district.groupby(['station', 'time'])[predictor].max()
            for time in df_district.time.unique():
                X.append([df_ordered.loc[(station, time)] for station in df_district.station.unique()])
                y.append(df_district[df_district['time'] == time]['flood'].values[0])
            # train and predict
            model = GradientBoostingClassifier(max_features='auto', loss='exponential')
            sample_weight = [len(y) / y.count(i) for i in y]
            model.fit(X, y, sample_weight)
            predictions = model.predict(X)
            # save performance
            best_performance = calc_performance_scores(pd.Series(y), pd.Series(predictions))
            best_performance['parameters'] = str(model.get_params())
            best_performance['district'] = district
            performance_scores = performance_scores.append(best_performance, ignore_index=True)

        elif model_type == 'bdt_discharge_rainfall':
            # get number of timesteps in rainfall data
            n_timesteps = len([x for x in df_district.columns if 'rainfall_cum' in x])
            # prepare training data
            X, y = [], []
            df_ordered = df_district.groupby(['station', 'time'])[predictor].max()
            for time in df_district.time.unique():
                X_t = [df_ordered.loc[(station, time)] for station in df_district.station.unique()] # discharge
                X_t.extend([df_district[df_district['time'] == time]['rainfall_max_'+str(t)].values[0] for t in
                           range(n_timesteps)])  # rainfall max
                X_t.extend([df_district[df_district['time'] == time]['rainfall_cum_' + str(t)].values[0] for t in
                           range(n_timesteps)])  # rainfall cum
                # fix nan
                X_t = np.nan_to_num(X_t, nan=0)
                X.append(X_t)
                y.append(df_district[df_district['time'] == time]['flood'].values[0])
            # train and predict
            model = GradientBoostingClassifier(max_features='auto', loss='exponential')
            sample_weight = [len(y) / y.count(i) for i in y]
            model.fit(X, y, sample_weight)
            predictions = model.predict(X)
            # save performance
            best_performance = calc_performance_scores(pd.Series(y), pd.Series(predictions))
            best_performance['parameters'] = str(model.get_params())
            best_performance['district'] = district
            performance_scores = performance_scores.append(best_performance, ignore_index=True)

    return performance_scores


def main(country='Uganda',
         ct_code='uga',
         model='quantile_discharge',
         loss='f1'):

    # Path name to t:wqhe folder and local path
    my_local_path = str(Path(os.getcwd()))
    path = my_local_path + '/' + country + '/'

    # Set path to admin level shape to use for the study
    if country == 'Uganda':
        Admin = path + 'input/Admin/uga_admbnda_adm1_UBOS_v2.shp'     #activate for Uganda
    elif country == 'Kenya':
        Admin = path + 'input/Admin/KEN_adm1_mapshaper_corrected.shp' # activate for Kenya

    #%% GLOFAS DATA EXTRACTION AND ANALYSIS

    # Find the Glofas Stations in the Country
    #  extract discharge time series for each station from the Glofas Grid data
    # Compute the extreme annual discharge per station and the threshold quantiles

    # get glofas stations (dataframe)
    Gl_stations = prepare_glofas_stations_data(country, my_local_path)

    # get glofas data (grid Netcdf file)
    # if not found, download
    glofas_grid = path +'input/Glofas/%s_glofas_all.nc' %ct_code
    if not os.path.exists(glofas_grid):
        print('GloFAS data not found, downloading it (this might take some time)')
        nc = get_glofas_data(country=country.lower(),
                             return_type='xarray',
                             credentials_file='settings.cfg')
        nc.to_netcdf(glofas_grid)
        print('download complete, continuing')
    else:
        nc = xr.open_dataset(glofas_grid)

    # Create dataframe of glofas stations, daily discharge and max-over-3-days discharge
    df_discharge = prepare_dataframe_stations_discharge(Gl_stations, nc, path)

    # get impact data
    # NB Change date format and the name of the admin column depending on the input of the country impact data .csv!!
    if country.lower() == 'uganda':
        admin_column = 'Area'
        date_format = '%d/%m/%Y'
    elif country.lower() == 'kenya':
        admin_column = 'County'
        date_format = '%m/%d/%Y'
    impact_floods = get_impact_data(date_format, admin_column, ct_code, path)

    # open the impacted_area and Glofas related stations per district files
    df_dg = pd.read_csv(path + 'input/%s_affected_area_stations.csv' % ct_code, encoding='latin-1')
    df_dg['name'] = df_dg['name'].str.lower()
    df_dg_long = df_dg[['name', 'Glofas_st', 'Glofas_st2', 'Glofas_st3', 'Glofas_st4']]\
        .melt(id_vars='name', var_name='glofas_n', value_name='station').drop('glofas_n', 1).dropna()
    df_dg_long = df_dg_long.rename(columns={'name': 'district'})

    # if model includes rainfall, prepare dataset
    if 'rainfall' in model:
        # if not found, download
        rainfall_processed_data_path = path + 'input/rainfall/CHIRPS_data_processed.csv'
        if os.path.exists(rainfall_processed_data_path):
            df_rainfall = pd.read_csv(rainfall_processed_data_path)
            df_rainfall.time = pd.to_datetime(df_rainfall.time)
        else:
            rainfall_raw_data_path = path + 'input/rainfall/CHIRPS_data_raw.csv'
            if not os.path.exists(rainfall_raw_data_path):
                from CHIRPS_utils import get_CHIRPS_data
                ## Code CHIRPS_utils zelf aangepast
                print('CHIRPS rainfall data not found, downloading it (this might take some time)')
                get_CHIRPS_data(country=country,
                                catchment_shapefile=path+'input/catchment/'+ct_code+'_catchment_districts.shp',
                                year_start=2020,
                                year_end=2020,
                                output_dir=path+'input/rainfall')
                print('download complete, continuing')
            df_rainfall = pd.read_csv(rainfall_raw_data_path)
            df_rainfall.to_csv('rainfall2016-2019_raw.csv')
            print('processing CHIRPS rainfall data (this might take some time)')
            df_rainfall = prepare_rainfall_data(df_rainfall, n_timesteps=8)
            df_rainfall.to_csv(rainfall_processed_data_path)
            print('processing complete, continuing')
            df_rainfall.to_csv('rainfall2016-2019_processed.csv')

    # join together tables and extract discharge data to create a prediction model table (df_model)
    df_model = pd.merge(df_discharge, df_dg_long, how='left', on='station').dropna()
    df_model = pd.merge(df_model, impact_floods, how='left', on=['time', 'district'])
    df_model = pd.merge(df_model, Gl_stations[['station']], how='left', on='station')

    if 'rainfall' in model:
        df_model = pd.merge(df_model, df_rainfall, how='left', on=['time', 'district'])
    df_model = df_model[df_model['time'] > (impact_floods.time.min() - dt.timedelta(days=7))]    #filtering the df to date after the first observed event
    df_model = df_model[df_model['time'] < (impact_floods.time.max() + dt.timedelta(days=7))]    #filtering the df to date before the last observed event
    df_model['flood'] = df_model['flood'].fillna(0)
    #df_model.to_csv('df_model_discharge_ken.csv')

    #df_model.to_csv('df_Uganda.csv') #zelf toegevoegd
    #df_model.to_pickle("./df_Uganda.pkl")
    #df_model.to_csv('df_Kenya.csv') #zelf toegevoegd
    #df_model.to_pickle("./df_Kenya.pkl")

    # train & test model, compute performance
    print('starting model training & testing')
    performance = train_test_model(df_model,
                                   predictor='max_future_3days', #max_dt_3days / max_future_3days
                                   model_type='quantile_discharge',
                                   loss=loss, n_ahead = 5)

    # Add column names to the performances
    performance.columns = ['district', 'f1', 'hits', 'misses', 'false_alarms', 'corr_neg', 'csi', 'pod', 'far', 'pofd', 'n_test_obs', 'floods_intest']

    # add to performance the number of floods per district
    #floods_per_district = impact_floods.groupby('district')['flood'].count()
    #performance = pd.merge(floods_per_district, performance, how='left', on=['district'])
    #performance = performance.rename(columns={'flood': 'nb_event'})
    performance.to_csv(str(Path(os.getcwd()))+"/Results/Benchmark/performance_BenchM_"+country+'_future3days_n5.csv')

    #performance.to_csv(path + 'output/Performance_scores/{}_glofas_{}_performance_score.csv'.format(ct_code, model), index=False)

    #print('median performance:')
    #print(performance[['pod', 'far', 'pofd', 'csi']].median())



if __name__ == "__main__":
    import plac
    plac.call(main)