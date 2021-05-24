###### Script loads the needed data for the model ######
########################################################
import numpy as np
import pandas as pd
import os
from pathlib import Path
from V12_glofas_analysis import prepare_glofas_stations_data
from database_utils import get_glofas_data
from V12_glofas_analysis import prepare_dataframe_stations_discharge
import xarray as xr
import datetime

##### Function to retrieve data per district
def get_district_level(district, df_total):
    df_district = df_total[df_total['district'] == district]
    num = 1
    min_date = pd.to_datetime(df_district['time'].min())
    max_date = pd.to_datetime(df_district['time'].max())
    delta = max_date - min_date
    n_days = delta.days + 1

    for station in df_district['station'].unique():
        df_station = df_district[df_district['station'] == station]
        df_district['dis_station' + str(num)] = df_station['dis']
        df_district['maxdis_station' + str(num)] = df_station['max_dt_3days']

        if num > 1:
            df_district['dis_station' + str(num)] = df_district['dis_station' + str(num)].shift(-(num-1)*n_days)
            df_district['maxdis_station' + str(num)] = df_district['maxdis_station' + str(num)].shift(-(num-1)*n_days)

        num = num + 1

    df_district = df_district.iloc[0:n_days]
    return df_district

def get_impact_data(date_format, admin_column, ct_code, path):
    flood_events = pd.read_csv(path + 'input/%s_impact_data.csv' % ct_code, encoding='latin-1')
    flood_events['Date'] = pd.to_datetime(flood_events['Date'], format=date_format)
    flood_events = flood_events.query("Date >= '2000-01-01' ")

    if ct_code == 'uga':
        flood_events= flood_events[flood_events['data_quality_score'] > 3]
    elif ct_code == 'ken':
        flood_events= flood_events[flood_events['data_quality'] > 0]

    flood_events = flood_events[['Date', admin_column, 'flood']].drop_duplicates()\
        .rename(columns={admin_column: 'district'}).dropna().set_index('Date')

    flood_events['district'] = flood_events['district'].str.lower()
    flood_events = flood_events.reset_index().rename(columns={'Date': 'time'})
    return flood_events

def get_season(month):
    if month in (6, 7, 8):
        seasons = 'summer'
    elif month in (9, 10, 11):
        seasons = 'fall'
    elif month in (12, 1, 2):
        seasons = 'winter'
    elif month in (3, 4, 5):
        seasons = 'spring'

    return seasons

def column_adjustment(df):
    df[['Year', 'Month', 'Day']] = df[['Year', 'Month', 'Day']].astype(int)
    df['time'] = pd.to_datetime((df.Year * 10000 + df.Month * 100 + df.Day).apply(str), format='%Y-%m-%d')
    if 'PCODE' in df.columns:
        df = df.drop(columns = ['Year', 'Month', 'Day', 'PCODE', 'Unnamed: 0', 'Unnamed: 0.1'])
    else:
        df = df.drop(columns = ['Year', 'Month', 'Day'])
    df.district = df.district.str.lower()
    df = df.groupby(['district', 'time']).first()
    return df

def load_data():

    #Specify the columns you want to extract from the soil moisture data
    cols_uga = ['Day', 'Month', 'Year', 'District', 'SoilMoi0_10cm_inst_mean', 'SoilMoi10_40cm_inst_mean',
                'SoilMoi40_100cm_inst_mean', 'SoilMoi100_200cm_inst_mean']
    cols_ken = ['Day', 'Month', 'Year', 'ADMIN', 'SoilMoi0_10cm_inst_mean', 'SoilMoi10_40cm_inst_mean',
                'SoilMoi40_100cm_inst_mean', 'SoilMoi100_200cm_inst_mean']

    soil_ken = pd.read_csv(r'~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\soil_moisture\soil_moisture_Kenya.csv', usecols = cols_ken)
    rainfall_ken = pd.read_csv(r'~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\rainfall\CHIRPS_data_raw.csv')
    soil_uga = pd.read_csv(r'~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\uganda\input\soil_moisture\soil_moisture_Uganda.csv', usecols = cols_uga)
    rainfall_uga = pd.read_csv(r'~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\uganda\input\rainfall\CHIRPS_data_raw.csv')

    # Rename columns in soil moisture and rainfall data
    soil_ken = soil_ken.rename(columns={'ADMIN': 'district'})
    rainfall_ken = rainfall_ken.rename(columns={'District': 'district'})
    soil_uga = soil_uga.rename(columns={'District': 'district'})
    rainfall_uga = rainfall_uga.rename(columns={'District': 'district'})

    # Make date column out of the Day, Month and Year columns
    soil_ken = column_adjustment(soil_ken)
    rainfall_ken = column_adjustment(rainfall_ken)
    soil_uga = column_adjustment(soil_uga)
    rainfall_uga = column_adjustment(rainfall_uga)

    # Discharge data with original impact data file
    #dis_uga = pd.read_csv(r'~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\uganda\input\Glofas\discharge_uga.csv')
    #dis_ken = pd.read_csv(r'~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\Glofas\discharge_ken.csv')
    ##### Get discharge data & impact data #####

    for country in ['Kenya', 'Uganda']:

        my_local_path = str(Path(os.getcwd()))
        path = my_local_path + '/' + country + '/'

        # Set path to admin level shape to use for the study
        if country =='Kenya':
            Admin = path + 'input/Admin/KEN_adm1_mapshaper_corrected.shp'  # activate for Kenya
            ct_code = 'ken'
        elif country == 'Uganda':
            Admin = path + 'input/Admin/uga_admbnda_adm1_UBOS_v2.shp'     #activate for Uganda
            ct_code = 'uga'


        # get glofas stations (dataframe)
        Gl_stations = prepare_glofas_stations_data(country, my_local_path)

        # get glofas data (grid Netcdf file)
        # if not found, download
        glofas_grid = path + 'input/Glofas/%s_glofas_all.nc' % ct_code
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
            df_rainfall = rainfall_uga
            df_soil = soil_uga
            date_format = '%d/%m/%Y'
        elif country.lower() == 'kenya':
            admin_column = 'County'
            df_rainfall = rainfall_ken
            df_soil = soil_ken
            date_format = '%m/%d/%Y'
        impact_floods = get_impact_data(date_format, admin_column, ct_code, path)

        # open the impacted_area and Glofas related stations per district files
        df_dg = pd.read_csv(path + 'input/%s_affected_area_stations.csv' % ct_code, encoding='latin-1')
        df_dg['name'] = df_dg['name'].str.lower()
        df_dg_long = df_dg[['name', 'Glofas_st', 'Glofas_st2', 'Glofas_st3', 'Glofas_st4']]\
            .melt(id_vars='name', var_name='glofas_n', value_name='station').drop('glofas_n', 1).dropna()
        df_dg_long = df_dg_long.rename(columns={'name': 'district'}) #contains districts linked to stations

        # join together tables and extract discharge data to create a prediction model table (df_model)
        df_model = pd.merge(df_discharge, df_dg_long, how='left', on='station').dropna()
        df_model = pd.merge(df_model, impact_floods, how='left', on=['time', 'district'])
        #df_model = pd.merge(df_model, Gl_stations[['station']], how='left', on='station') does not do anything?
        df_model['flood'] = df_model['flood'].fillna(0)

        #Remove districts which do not have information on impact data
        all_districts = df_model['district'].unique()
        for dis in all_districts:
            df_dis = df_model[df_model['district'] == dis]
            floods = df_dis['flood'].sum()
            if floods == 0:
                df_model = df_model[df_model['district'] != dis]

        df_model['time'] = pd.to_datetime(df_model['time'], format='%Y-%m-%d')
        min_date = df_model['time'].min()
        max_date = df_model['time'].max()
        delta = max_date - min_date
        n_days = delta.days + 1

        # Check if for each combination of station/district there are no missing data points
        print(country)
        test = df_model
        test['station_dis'] = df_model['station'] + df_model['district']
        for d in df_model['station_dis'].unique():
            days = len(df_model[df_model['station_dis'] == d])
            if days != n_days:
                print(days)
                print(d)

        ##### Merge everything together #####
        df_model = df_model.replace('wakiso', 'wasiko')
        df_model = pd.merge(df_model, df_rainfall, how='left', on=['time', 'district']) #join outer here? such that districts without discharge but with rain or the other way around are still included?
        df_model = pd.merge(df_model, df_soil, how='left', on=['time', 'district'])#same here?
        #df_model = df_model[df_model['time'] > (impact_floods.time.min() - dt.timedelta(days=7))]    #filtering the df to date after the first observed event
        #df_model = df_model[df_model['time'] < (impact_floods.time.max() + dt.timedelta(days=7))]    #filtering the df to date before the last observed event
        df_model['Country'] = country

        df_model['Month'] = pd.to_datetime(df_model['time'])
        df_model['Month'] = df_model['Month'].dt.month
        df_model['season'] = df_model['Month'].apply(get_season)
        df_model = df_model.drop(columns=['Month'])
        df_model = df_model.replace('wasiko', 'wakiso') #Fix error

        if country == 'Uganda':
            df_discharge_uga = df_discharge
            impact_floods_uga = impact_floods
            df_model_uga = df_model
        elif country == 'Kenya':
            df_discharge_ken = df_discharge
            impact_floods_ken = impact_floods
            df_model_ken = df_model

    ##### Merge the data from Kenya and Uganda #####
    df_total = pd.concat([df_model_ken, df_model_uga])
    for dis in df_total['district'].unique():  # Remove districts without impact data from data
        df_dis = df_total[df_total['district'] == dis]
        if df_dis['flood'].sum() == 0:
            df_total = df_total[df_total['district'] != dis]

    df_total = df_total.rename(columns = {'max': 'rainfall'})
    df_total.to_csv('df_total_mediumqual.csv')

    return df_total

#result = load_data()

if __name__ == "__load_data__":
    import plac
    plac.call(load_data)


