### FUNCTION THAT IS USED TO CREATE NEW VARIABLES
import pandas as pd
import numpy as np
import os
from pathlib import Path

def relative_change(df):
    changes = ((df[0] - df[1])/df[1]) * 100
    return changes

def create_features(df_d, district_level, agg_interval, window, n_ahead):
    n_days = 5

    #Remove unneccessary columns
    max_cols = df_d.filter(regex='max|Unnamed').columns
    df_d = df_d.drop(columns = max_cols)
    df_d = df_d.drop(columns = ['dis', 'station_dis', 'station'])
    drop_cols = df_d.filter(regex='dis_station|rainfall').columns

    #0) Introduce the cross-product of rainfall and soil moisture level 1
    df_d['rainSM_L1'] = df_d['SoilMoi0_10cm_inst_mean']*df_d['rainfall']

    #1) Introduce the Cross-product of two soilmoisture variables (level 2 + level 4) & drop original soilm columns
    soil_df = df_d.filter(regex='Soil')
    df_d['soilM_2x4'] = soil_df['SoilMoi10_40cm_inst_mean']*soil_df['SoilMoi100_200cm_inst_mean']
    df_d = df_d.drop(columns = soil_df.columns)

    #2) Season indicator
    season_dum = pd.get_dummies((df_d['season'].astype('category')))
    df_d = pd.concat([df_d, season_dum], axis=1).drop(columns=['fall', 'season'])

    if district_level == False:
        # Create extra features for panel level model
        my_local_path = str(Path(os.getcwd()))

        # 1) Add mountaineous information for this district
        df_ken = pd.read_csv(my_local_path + '/kenya/input/_vegetation/_vegetation_Kenya.csv')
        df_uga = pd.read_csv(my_local_path + '/uganda/input/_vegetation/_vegetation_Uganda.csv')
        curr_district = df_d['district'].iloc[0]

        if curr_district in df_ken['ADMIN'].str.lower().unique():
            df_ken['district'] = df_ken['ADMIN'].str.lower()
            veg_d = df_ken[df_ken['district'] == curr_district]
            veg_d['time'] = pd.to_datetime(veg_d['Year'] * 10000 + veg_d['Month'] * 100 + veg_d['Day'], format='%Y%m%d')
            df_d = pd.merge(df_d, veg_d[['EVI_mean', 'district', 'time']], how='left', on=['district', 'time'])

        elif curr_district in df_uga['District'].str.lower().unique():
            df_uga['district'] = df_uga['District'].str.lower()
            veg_d = df_uga[df_uga['district'] == curr_district]
            veg_d['time'] = pd.to_datetime(veg_d['Year'] * 10000 + veg_d['Month'] * 100 + veg_d['Day'], format='%Y%m%d')
            df_d = pd.merge(df_d, veg_d[['EVI_mean', 'district', 'time']], how='left', on=['district', 'time'])
        elif curr_district == 'wakiso':
            df_uga['district'] = df_uga['District'].str.lower()
            df_uga["district"].replace({"wasiko": "wakiso"}, inplace=True)
            veg_d = df_uga[df_uga['district'] == curr_district]
            veg_d['time'] = pd.to_datetime(veg_d['Year'] * 10000 + veg_d['Month'] * 100 + veg_d['Day'], format='%Y%m%d')
            df_d = pd.merge(df_d, veg_d[['EVI_mean', 'district', 'time']], how='left', on=['district', 'time'])

        else:
            print("No information on vegetation for district "+curr_district)

        # Interpolate vegetation
        ts = df_d.set_index('time')['EVI_mean']
        interpol = ts.interpolate(method='spline', order=2)  # interpolate
        df_d['EVI_mean'] = interpol.reset_index()['EVI_mean']

        # Fill Na's at the start with the first value available
        df_d['EVI_mean'] = df_d['EVI_mean'].bfill()

        # 2) Add water surface information for this district
        df_ken_water = pd.read_csv(my_local_path + '/kenya/input/water_surface/waterstats_ken.csv')
        df_uga_water = pd.read_csv(my_local_path + '/uganda/input/water_surface/waterstats_uga.csv')

        if curr_district in df_ken_water['name'].str.lower().unique():
            df_ken_water['district'] = df_ken_water['name'].str.lower()
            df_d = pd.merge(df_d, df_ken_water[['district', 'occurrence']], how='left', on='district')
        elif curr_district in df_uga_water['ADM1_EN'].str.lower().unique():
            df_uga_water['district'] = df_uga_water['ADM1_EN'].str.lower()
            df_d = pd.merge(df_d, df_uga_water[['district', 'occurrence']], how='left', on='district')
        else:
            print("No information on the water surface for district "+curr_district)


        # 3) Add information about how mountaineous the district is
        df_ken_slope = pd.read_csv(my_local_path + '/kenya/input/elevation/slope_ken.csv')
        df_uga_slope = pd.read_csv(my_local_path + '/uganda/input/elevation/slope_uga.csv')
        if curr_district in df_ken_slope['name'].str.lower().unique():
            df_ken_slope['district'] = df_ken_slope['name'].str.lower()
            df_d = pd.merge(df_d, df_ken_slope[['district', 'mean']], how='left', on='district')
        elif curr_district in df_uga_slope['ADM1_EN'].str.lower().unique():
            df_uga_slope['district'] = df_uga_slope['ADM1_EN'].str.lower()
            df_d = pd.merge(df_d, df_uga_slope[['district', 'mean']], how='left', on='district')
        else:
            print("No information on elevation for district "+curr_district)

        # 4) Create features that will be used to create spatial information
        df_d['NB_flood14days'] = df_d['flood'].rolling(14).max()

        glofas_cols = df_d.filter(regex='dis_').columns
        for station in glofas_cols:
            df_d['NB_max'+station] = df_d[station].rolling(14).max()


        # Rename these columns
        df_d = df_d.rename(columns={'mean': 'avg_slope', 'occurrence':'perc_water'})

    # Drop the rows containing NAs created through the rolling windows
    df_d.dropna(inplace=True)

    # Also return the column names of the continuous variables that should be taken into account for WT
    df_d['flood'] = df_d['flood'].astype('category')
    float_cols = df_d.columns[(df_d.dtypes.values == np.dtype('float64'))]

    return df_d, float_cols
