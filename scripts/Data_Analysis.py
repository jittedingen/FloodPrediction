from Load_Data import load_data
from Load_Data import get_season
from Load_Data import get_district_level
from Load_Data import get_impact_data
from Load_Data import get_season
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pathlib import Path
import math
import datetime
import geopandas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def normalize_s(series):
    max_value = series.max()
    min_value = series.min()
    series = (series - min_value) / (max_value - min_value)
    return series

def relative_change(df):
    changes = ((df[0] - df[1])/df[1]) * 100
    return changes

my_local_path = str(Path(os.getcwd()))
df_total = pd.read_csv(my_local_path + '/df_total_mediumqual.csv')

#### CREATE MAPS #####
heatmaps = True
if heatmaps == True:
    time = 'avg_per_season' #season/None/year/avg_per_year/avg_per_season
    variable = 'n_floods'
    #Possible variables to fill in above: n_floods, rainfall, SoilMoi0_10cm_inst_mean, SoilMoi10_40cm_inst_mean,
    # SoilMoi40_100cm_inst_mean, SoilMoi100_200cm_inst_mean
    # dis, area, EVI_mean, NDVI_mean
    for country in ['Kenya', 'Uganda']:
        path = my_local_path + '/' + country + '/'

        #for Uganda activate the following lines :
        if country == 'Uganda':
            Admin= path + 'input/Admin/uga_admbnda_adm1_UBOS_v2.shp'   # for Uganda
            Admin_col = 'ADM1_EN'  # column name of the Admin name in the shapefile of Uganda
            col_veg = 'District'
            ct_code = 'uga'
        elif country == 'Kenya':
            Admin = path + 'input/Admin/KEN_adm1_mapshaper_corrected.shp'  # for Kenya
            Admin_col = 'name'  # column name of the Admin name in the shapefile for Kenya
            col_veg = 'ADMIN'
            ct_code = 'ken'

        # Open the district admin 1 shapefile - Contains all districts of the country
        district= gpd.read_file(Admin)
        district[Admin_col] = district[Admin_col].str.replace(u"é", "e").str.lower()

        if variable == 'n_floods':
            # Create dataframe with total number of floods for each district & merge with file of all districts
            df_impact = pd.read_csv(path + 'input/%s_impact_data.csv' %ct_code, encoding='latin-1')
            if country == 'Kenya':
                df_impact = df_impact.rename(columns = {'County': 'district'})
                format = '%m/%d/%Y'
            elif country == 'Uganda':
                df_impact = df_impact.rename(columns = {'Area': 'district'})
                format = '%d/%m/%Y'

            df_impact['Date'] = pd.to_datetime(df_impact['Date'], format=format)
            df_impact = df_impact.query("Date >= '2000-01-01' ")

            if ct_code == 'uga':
                df_impact = df_impact[df_impact['data_quality_score'] > 3]
            elif ct_code == 'ken':
                df_impact = df_impact[df_impact['data_quality'] > 0]

            df_impact['Date'] = pd.to_datetime(df_impact['Date'], format=format)
            df_impact = df_impact[df_impact['district'].notna()] #only keep data where district is not nan
            df_impact['district'] = df_impact['district'].str.lower()
            df_impact = df_impact.dropna(subset = ['district'])
            all_districts = df_impact['district'].unique()

            df_variable = pd.DataFrame([(d, df_impact[df_impact['district'] == d]['flood'].sum()) for d in all_districts], columns = ['district', 'n_floods'])
            df_impact['year'] = df_impact['Date'].dt.year
            years = df_impact['year'].unique()
            df_impact['Month'] = df_impact['Date'].dt.month
            df_impact['season'] = df_impact['Month'].apply(get_season)
            df_impact['year+season'] = df_impact['year'].apply(str)+ ' ' + df_impact['season']
            years_seasons = df_impact['year+season'].unique()
            seasons = df_impact['season'].unique()
            df_variable_avg_year = pd.DataFrame([(year, d, df_impact[(df_impact['district'] == d) & (df_impact['year'] == year)]['flood'].sum()) for d in all_districts for year in years], columns = ['year', 'district', 'n_floods'])
            df_variable_avg_season = pd.DataFrame([(year_season, d, df_impact[(df_impact['district'] == d) & (df_impact['year+season'] == year_season)]['flood'].sum()) for d in all_districts for year_season in years_seasons], columns = ['year+season', 'district', 'n_floods'])
            df_variable_raw = df_impact.rename(columns = {'Date': 'time'})

            part_title = 'Number of floods'
            rounding = 5.0
            minimal_val = 0
            maximal_val = 100

        elif variable == 'area':
            df_variable = pd.read_csv(path + 'input/Admin/%s_areas_districts.csv' %ct_code, encoding='latin-1')
            df_variable['area'] = df_variable['area']/1000000
            df_variable = df_variable.rename(columns = {'name': 'district'})
            df_variable = df_variable[['district', 'area']]
            part_title = 'Size of district in squared km'
            maximal_val = df_variable['area'].max()
            minimal_val = df_variable['area'].min()
            rounding = 100

        elif variable == 'EVI_mean':
            df_variable = pd.read_csv(path + 'input/_vegetation/_vegetation_%s.csv' %country, encoding='latin-1')
            df_variable = df_variable.rename(columns = {col_veg: 'district'})
            all_districts = df_variable['district'].unique()
            df_variable_raw = df_variable
            df_variable = pd.DataFrame([(d, df_variable[df_variable['district'] == d][variable].mean()) for d in all_districts], columns = ['district', variable])

            part_title = 'Average EVI per district'
            maximal_val = df_variable[variable].max()
            minimal_val = df_variable[variable].min()
            rounding = 100

        elif variable == 'NDVI_mean':
            df_variable = pd.read_csv(path + 'input/_vegetation/_vegetation_%s.csv' %country, encoding='latin-1')
            df_variable = df_variable.rename(columns = {col_veg: 'district'})
            all_districts = df_variable['district'].unique()
            df_variable_raw = df_variable
            df_variable = pd.DataFrame([(d, df_variable[df_variable['district'] == d][variable].mean()) for d in all_districts], columns = ['district', variable])

            part_title = 'Average NDVI per district'
            maximal_val = df_variable[variable].max()
            minimal_val = df_variable[variable].min()
            rounding = 100

        elif variable == 'rainfall':
            #get dataframe with rainfall data
            df_variable = df_total[df_total['Country'] == country]
            df_variable['district'] = df_variable['district'].str.lower()
            df_variable = df_variable.dropna(subset = ['district'])
            df_variable_raw = df_variable
            all_districts = df_variable['district'].unique()
            df_variable = pd.DataFrame([(d, df_variable[df_variable['district'] == d]['rainfall'].mean()) for d in all_districts], columns = ['district', 'rainfall'])
            part_title = 'Average amount of rainfall'
            format = '%Y-%m-%d'
            rounding = 5.0
            minimal_val = 0
            maximal_val = int(math.ceil(df_variable[variable].max()/rounding)) * int(rounding)

        elif 'Soil' in variable:
            #get dataframe with soil data
            df_variable = df_total[df_total['Country'] == country]
            df_variable['district'] = df_variable['district'].str.lower()
            df_variable = df_variable.dropna(subset = ['district'])
            df_variable_raw = df_variable
            all_districts = df_variable['district'].unique()
            df_variable = pd.DataFrame([(d, df_variable[df_variable['district'] == d][variable].mean()) for d in all_districts], columns = ['district', variable])
            part_title = 'Average ' + variable
            format = '%Y-%m-%d'
            rounding = 5.0
            minimal_val = int(math.floor(df_variable[variable].min()/rounding)) * int(rounding)
            maximal_val = int(math.ceil(df_variable[variable].max()/rounding)) * int(rounding)

        elif variable == 'dis':
            df_variable = df_total[df_total['Country'] == country]
            df_variable['district'] = df_variable['district'].str.lower()
            df_variable = df_variable.dropna(subset = ['district'])
            df_variable_raw = df_variable
            all_districts = df_variable['district'].unique()
            all_stations = df_variable['station'].unique()

            #Normalize discharge data per station
            for station in all_stations:
                df_station = df_variable[df_variable['station'] == station]
                df_station['dis'] = pd.DataFrame(df_station['dis']).apply(normalize_s)
                df_variable[df_variable['station'] == station] = df_station

            format = '%Y-%m-%d'
            df_variable['time'] = pd.to_datetime(df_variable['time'], format=format)
            df_variable['year'] = df_variable['time'].dt.year
            years = df_variable['year'].unique()
            df_variable['Month'] = df_variable['time'].dt.month
            df_variable['season'] = df_variable['Month'].apply(get_season)
            df_variable['year+season'] = df_variable['year'].apply(str)+ ' ' + df_variable['season']
            years_seasons = df_variable['year+season'].unique()
            seasons = df_variable['season'].unique()
            df_variable_avg_year = pd.DataFrame([(year, d, df_variable[(df_variable['district'] == d) & (df_variable['year'] == year)][variable].var()) for d in all_districts for year in years], columns = ['year', 'district', 'variance_' + variable])
            df_variable_avg_season = pd.DataFrame([(year_season, d, df_variable[(df_variable['district'] == d) & (df_variable['year+season'] == year_season)][variable].var()) for d in all_districts for year_season in years_seasons], columns = ['year+season', 'district', 'variance_' + variable])

            df_variable = pd.DataFrame([(d, df_variable[df_variable['district'] == d][variable].mean()) for d in all_districts], columns = ['district', variable])

            part_title = 'Average normalized ' + variable
            rounding = 0.005
            minimal_val = math.floor(df_variable[variable].min()/rounding) * rounding
            maximal_val = math.ceil(df_variable[variable].max()/rounding) * rounding


        if time == None:
            #Merged_ contains the stuff that is plotted
            # lower case the district column in both file and merge
            title = part_title
            vmax = maximal_val
            vmin = minimal_val
            df_variable['district']= df_variable['district'].str.lower()
            merged= district.set_index(Admin_col).join(df_variable.set_index('district'), lsuffix = 'double')

            fig, (ax1) = plt.subplots(1, 1, figsize=(16, 16))
            divider_ax1 = make_axes_locatable(ax1)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            cax1 = divider_ax1.append_axes("right", size="5%", pad=0.2)
            cax1.tick_params(axis='both', labelsize=40)

            #ax1.set_title(part_title + ' per district in ' + country, fontsize=16)
            merged.plot(ax=ax1, color='grey', edgecolor='black')
            merged.plot(ax=ax1, column=variable, legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax1)

            fig.savefig(path + 'output/Data_Analysis/%s_%s.png' % (variable, country))

        elif time == 'season':
            #Get the variable per season
            if (variable == 'EVI_mean') or (variable == 'NDVI_mean'):
                df_variable_raw['season'] = df_variable_raw['Month']
            else:
                df_variable_raw['season'] = pd.to_datetime(df_variable_raw['time'], format = format).dt.month
                #Remove impact events that do not include a date (8 for Kenya)
                df_variable_raw = df_variable_raw.dropna(subset = ['time', 'district'])

            df_variable_raw['season'] = df_variable_raw['season'].apply(get_season)

            if variable == 'n_floods':
                df_season = pd.DataFrame([(s, d, df_variable_raw[(df_variable_raw['district'] == d) & (df_variable_raw['season'] == s)]['flood'].sum()) for d in all_districts for s in ('summer', 'fall', 'spring', 'winter')], columns=['season', 'district', variable])
            elif variable == 'rainfall':
                df_season = pd.DataFrame([(s, d, df_variable_raw[(df_variable_raw['district'] == d) & (df_variable_raw['season'] == s)]['rainfall'].mean()) for d in all_districts for s in ('summer', 'fall', 'spring', 'winter')], columns=['season', 'district', variable])
            elif 'Soil' in variable:
                df_season = pd.DataFrame([(s, d, df_variable_raw[(df_variable_raw['district'] == d) & (df_variable_raw['season'] == s)][variable].mean()) for d in all_districts for s in ('summer', 'fall', 'spring', 'winter')], columns=['season', 'district', variable])
            elif (variable == 'dis') or (variable == 'EVI_mean') or (variable == 'NDVI_mean'):
                df_season = pd.DataFrame([(s, d, df_variable_raw[(df_variable_raw['district'] == d) & (df_variable_raw['season'] == s)][variable].mean()) for d in all_districts for s in ('summer', 'fall', 'spring', 'winter')], columns=['season', 'district', variable])

            df_season['district'] = df_season['district'].str.lower()
            df_season = df_season.set_index(['district', 'season']).unstack('season').reset_index()
            df_season.columns = df_season.columns.droplevel(level=0)
            df_season = df_season.rename(columns={'': 'district'})
            df_season['district']= df_season['district'].str.lower()
            df_merged = district.set_index(Admin_col).join(df_season.set_index('district'))
            merged = df_merged

            vmin = minimal_val
            max = df_season[['fall', 'summer', 'winter', 'spring']].max().max()
            vmax = int(math.ceil(max/rounding))*int(rounding)
            #vmax = 55

            if variable == 'dis':
                vmax = math.ceil(max / rounding) * rounding

            #### Make plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
            #fig.suptitle(part_title + ' per district per season in ' + '%s' % country, fontsize=16,
                         #fontweight='bold', x=0.5, y=0.94)
            divider_ax1 = make_axes_locatable(ax1)
            divider_ax2 = make_axes_locatable(ax2)
            divider_ax3 = make_axes_locatable(ax3)
            divider_ax4 = make_axes_locatable(ax4)

            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
            ax3.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)
            ax4.yaxis.set_visible(False)

            cax1 = divider_ax1.append_axes("right", size="5%", pad=0.2)
            cax1.tick_params(axis='both', labelsize=40)
            cax2 = divider_ax2.append_axes("right", size="5%", pad=0.2)
            cax2.tick_params(axis='both', labelsize=40)
            cax3 = divider_ax3.append_axes("right", size="5%", pad=0.2)
            cax3.tick_params(axis='both', labelsize=40)
            cax4 = divider_ax4.append_axes("right", size="5%", pad=0.2)
            cax4.tick_params(axis='both', labelsize=40)


            ax1.set_title('Spring', fontsize=40)
            merged.plot(ax=ax1, color='grey', edgecolor='black')
            merged.plot(ax=ax1, column='spring', legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax1)

            ax2.set_title('Summer', fontsize=40)
            merged.plot(ax=ax2, color='grey', edgecolor='black')
            merged.plot(ax=ax2, column='summer', legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax2)

            ax3.set_title('Fall', fontsize=40)
            merged.plot(ax=ax3, color='grey', edgecolor='black')
            merged.plot(ax=ax3, column='fall', legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax3)

            ax4.set_title('Winter', fontsize=40)
            merged.plot(ax=ax4, color='grey', edgecolor='black')
            merged.plot(ax=ax4, column='winter', legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax4)

            fig.savefig(path + 'output/Data_Analysis/Season_%s_%s.png' % (variable, country))

        elif time == 'year':
            df_variable_raw['year'] = pd.to_datetime(df_variable_raw['time'], format = format).dt.year

            #Remove impact events that do not include a date (8 for Kenya)
            df_variable_raw = df_variable_raw.dropna(subset = ['time', 'district'])
            if variable == 'n_floods':
                df_year = pd.DataFrame([(y, df_variable_raw[(df_variable_raw['year'] == y)]['flood'].sum()) for y in range(2000, 2020)], columns=['year', variable])
            elif variable == 'rainfall':
                df_year = pd.DataFrame([(y, df_variable_raw[(df_variable_raw['year'] == y)]['rainfall'].mean()) for y in range(2000, 2020)], columns=['year', variable])
            elif 'Soil' in variable:
                df_year = pd.DataFrame([(y, df_variable_raw[(df_variable_raw['year'] == y)][variable].mean()) for y in range(2000, 2020)], columns=['year', variable])
            elif variable == 'dis':
                df_year = pd.DataFrame([(y, df_variable_raw[(df_variable_raw['year'] == y)][variable].mean()) for y in range(2000, 2020)], columns=['year', variable])

            df_year = df_year.sort_values(by='year')

            ###### POSSIBILITY TO GET THIS FOR EVERY DISTRICT SEPARATELY
            # get dataframe with specified variable per year for each district
            #df_year_district = pd.DataFrame([(y, d, df_variable_raw[(df_variable_raw['district'] == d) & (df_variable_raw['year'] == y)]['flood'].sum()) for y in range(2000, 2020) for d in all_districts], columns = ['year', 'district', 'n_floods'])

            # get data frame which calculates the yearly average number of floods per district
            #df_district_avg = pd.DataFrame([(d, df_year_district[df_year_district['district'] == d]['n_floods'].mean()) for d in all_districts], columns = ['district', 'avg_yearly_floods'])

            plt.figure()
            plt.plot(df_year['year'], df_year[variable])
            max_variable = df_year[variable].max()
            rounded_max_variable = int(math.ceil(max_variable / rounding)) * int(rounding)
            if variable == 'dis':
                rounded_max_variable = math.ceil(max_variable / rounding) * rounding

            plt.axis([2000, 2020, minimal_val, rounded_max_variable])
            plt.xticks(np.arange(2000, 2020, 2))
            plt.savefig(path + 'output/Data_Analysis/' + part_title + '_%s.png' %country)


        elif time == 'avg_per_year':
            if variable == 'dis':
                df_avg_year_district = pd.DataFrame([(d, df_variable_avg_year[df_variable_avg_year['district'] == d]['variance_dis'].mean()) for d in all_districts], columns=['district', variable + time])
            else:
                df_avg_year_district = pd.DataFrame([(d, df_variable_avg_year[df_variable_avg_year['district'] == d][variable].mean()) for d in all_districts], columns = ['district', variable + time])
            #vmin = minimal_val
            #vmax = np.ceil(df_avg_year_district[variable+time].max())
            vmax = 6
            vmin = 0
            #vmax = 0.040
            merged = district.set_index(Admin_col).join(df_avg_year_district.set_index('district'))

            fig, (ax1) = plt.subplots(1, 1, figsize=(16, 16))
            divider_ax1 = make_axes_locatable(ax1)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            cax1 = divider_ax1.append_axes("right", size="5%", pad=0.2)
            cax1.tick_params(axis='both', labelsize=40)

            # ax1.set_title(part_title + ' per district in ' + country, fontsize=16)
            merged.plot(ax=ax1, color='grey', edgecolor='black')
            merged.plot(ax=ax1, column=variable+time, legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax1)

            fig.savefig(path + 'output/Data_Analysis/%s_%s.png' % (variable+time, country))

        elif time == 'avg_per_season':
            df_variable_avg_season['season'] = df_variable_avg_season['year+season'].str.split(" ", n = 1, expand = True)[1]
            seasons = ['spring', 'summer', 'winter', 'fall']
            df_avg_season_district = pd.DataFrame([(season, d, df_variable_avg_season[(df_variable_avg_season['district'] == d) & (df_variable_avg_season['season'] == season)]['n_floods'].mean()) for d in all_districts for season in seasons], columns = ['season', 'district', variable + time])


            #vmin = minimal_val
            #vmax = np.ceil(df_avg_season_district[variable+time].max())
            vmin = 0
            vmax = 3
            #vmax = df_avg_season_district[variable + time].max()
            df_avg_season_district = df_avg_season_district.set_index(['district', 'season']).unstack('season').reset_index()
            df_avg_season_district.columns = df_avg_season_district.columns.droplevel(level=0)
            df_avg_season_district = df_avg_season_district.rename(columns={'': 'district'})
            df_avg_season_district['district']= df_avg_season_district['district'].str.lower()
            merged = district.set_index(Admin_col).join(df_avg_season_district.set_index('district'))

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
            # fig.suptitle(part_title + ' per district per season in ' + '%s' % country, fontsize=16,
            # fontweight='bold', x=0.5, y=0.94)
            divider_ax1 = make_axes_locatable(ax1)
            divider_ax2 = make_axes_locatable(ax2)
            divider_ax3 = make_axes_locatable(ax3)
            divider_ax4 = make_axes_locatable(ax4)

            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
            ax3.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)
            ax4.yaxis.set_visible(False)

            cax1 = divider_ax1.append_axes("right", size="5%", pad=0.2)
            cax1.tick_params(axis='both', labelsize=40)
            cax2 = divider_ax2.append_axes("right", size="5%", pad=0.2)
            cax2.tick_params(axis='both', labelsize=40)
            cax3 = divider_ax3.append_axes("right", size="5%", pad=0.2)
            cax3.tick_params(axis='both', labelsize=40)
            cax4 = divider_ax4.append_axes("right", size="5%", pad=0.2)
            cax4.tick_params(axis='both', labelsize=40)

            ax1.set_title('Spring', fontsize=40)
            merged.plot(ax=ax1, color='grey', edgecolor='black')
            merged.plot(ax=ax1, column='spring', legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax1)

            ax2.set_title('Summer', fontsize=40)
            merged.plot(ax=ax2, color='grey', edgecolor='black')
            merged.plot(ax=ax2, column='summer', legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax2)

            ax3.set_title('Fall', fontsize=40)
            merged.plot(ax=ax3, color='grey', edgecolor='black')
            merged.plot(ax=ax3, column='fall', legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax3)

            ax4.set_title('Winter', fontsize=40)
            merged.plot(ax=ax4, color='grey', edgecolor='black')
            merged.plot(ax=ax4, column='winter', legend=True, vmin=vmin, vmax=vmax, cmap='coolwarm', cax=cax4)

            fig.savefig(path + 'output/Data_Analysis/%s_%s.png' % (variable+time, country))

        #create a shapefile out of the uga_affected_area_stations.csv file:
        #df_dg = pd.read_csv(path + 'input/%s_affected_area_stations.csv' %ct_code)
        #AffArea_station =district.set_index(Admin_col).join(df_dg) #joins left
        #AffArea_station.to_file(path + 'output/Data_Analysis/AffDistrict_%s.shp' %ct_code, index=True)

##### Create map of countries with GloFAS stations and rivers to explain data #####
for country in ['Kenya', 'Uganda']:

    ## Get shapefile of glofas stations
    Gl_stations = pd.read_csv(my_local_path + '/africa/glofas/Glofaspoints_Africa_510.csv')  # do not change
    Gl_stations = Gl_stations[Gl_stations['CountryNam'] == country]
    Gl_stations['station'] = Gl_stations['ID']
    Gl_stations = Gl_stations[['ID', 'station', 'Stationnam', 'CountryNam', 'XCorrected', 'YCorrected']].set_index(
        'ID').rename(
        columns={'Stationnam': 'location', 'CountryNam': 'Country', 'XCorrected': 'lon', 'YCorrected': 'lat'})
    Gl_stations = Gl_stations[['Country', 'station', 'location', 'lon', 'lat']]

    # Only include stations which are present in the final data set
    all_stations = df_total[df_total['Country'] == country]['station'].unique()
    Gl_stations = Gl_stations[Gl_stations['station'].isin(all_stations)]

    gdf = geopandas.GeoDataFrame(Gl_stations, geometry=geopandas.points_from_xy(Gl_stations.lon, Gl_stations.lat))
    gdf.to_file('gdf_' + country, driver='ESRI Shapefile')

    ### Get shapefile of districts & rivers
    if country == 'Uganda':
        Admin = my_local_path + '/' + country.lower() + '/' + 'input/Admin/uga_admbnda_adm1_UBOS_v2.shp'  # for Uganda
        Admin_col = 'ADM1_EN'  # column name of the Admin name in the shapefile of Uganda
        ct_code = 'uga'
        rivers = gpd.read_file(my_local_path + '/' + country.lower() + '/' + 'input/hydroshed/hydroshed2.shp')
    elif country == 'Kenya':
        Admin = my_local_path + '/' + country.lower() + '/' + 'input/Admin/KEN_adm1_mapshaper_corrected.shp'  # for Kenya
        Admin_col = 'name'  # column name of the Admin name in the shapefile for Kenya
        ct_code = 'ken'
        rivers = gpd.read_file(my_local_path + '/' + country.lower() + '/' + 'input/hydroshed/hydroshed2.shp')
    district = gpd.read_file(Admin)
    district[Admin_col] = district[Admin_col].str.replace(u"é", "e").str.lower()
    rivers = rivers[rivers['UP_CELLS'] >= 800] #Only get main rivers

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    rivers.plot(ax = ax, linewidth = 0.5, color = 'blue', edgecolor = 'black')
    district.plot(ax = ax, color = 'white', edgecolor = 'black', linewidth = 0.5)
    gdf.plot(ax = ax, marker='o', color='red', markersize=20)
    ax.axis('off')
    plt.savefig(my_local_path + '/' + country.lower() + '/' + 'output/Data_Analysis/RiverStations_%s.png' % country, dpi = 300)
    #plt.show()


#### RIVER DISCHARGE + RAINFALL VS TIME PLOTS
#dis = 'bungoma'
dis = 'butaleja'
df_district = df_total[df_total['district'] == dis]
dis_stations = df_district['station'].unique()

for station in dis_stations:
    fig, ax = plt.subplots()
    df_station = df_district[df_district['station'] == station]
    ax2 = ax.twinx()

    ax.plot(pd.to_datetime(df_station['time']), df_station['dis'], 'r', label = 'River discharge')
    ax.plot(pd.to_datetime(df_station[df_station['flood'] == 1]['time']), df_station[df_station['flood'] == 1]['dis'], 'ko')
    ax2.plot(pd.to_datetime(df_station['time']), df_station['max'], 'b', label = 'Rainfall')
    ax.set_xlim([datetime.date(2006, 11, 21), datetime.date(2006, 12, 14)])
    ax.set_ylim([200,600])
    ax2.set_ylim([0,120])
    ax.set_ylabel('River discharge')
    ax2.set_ylabel('Rainfall')
    ax.legend(loc = 2)
    ax2.legend(loc = 0)
    plt.savefig(my_local_path + '/%s_disrainfall.png' %station)
    #plt.show()

#### RIVER DISCHARGE + SOIL MOISTURE VS TIME PLOTS
for station in dis_stations:
    fig, ax = plt.subplots()
    df_station = df_district[df_district['station'] == station]
    ax2 = ax.twinx()

    ax.plot(pd.to_datetime(df_station['time']), df_station['dis'], 'r', label = 'River discharge')
    ax.plot(pd.to_datetime(df_station[df_station['flood'] == 1]['time']), df_station[df_station['flood'] == 1]['dis'], 'ko')
    ax2.plot(pd.to_datetime(df_station['time']), df_station['SoilMoi40_100cm_inst_mean'], 'b', label = 'Soil Moisture 40-100cm')
    ax.set_xlim([datetime.date(2006, 11, 21), datetime.date(2006, 12, 14)])
    ax.set_ylim([200,600])
    ax2.set_ylim([170,220])
    ax.set_ylabel('River discharge')
    ax2.set_ylabel('Soil Moisture')
    ax.legend(loc = 2)
    ax2.legend(loc = 0)
    plt.savefig(my_local_path + '/%s_dissoil40-100.png' %station)
    #plt.show()


#### PLOT DIFFERENT DISCHARGE STATIONS OVER EACH OTHER ####
# Do they follow the same pattern?

dis = 'bungoma'
df_district = df_total[df_total['district'] == dis]
dis_stations = df_district['station'].unique()
fig, ax = plt.subplots()#figsize=(40,25))
for station in dis_stations:
    df_station = df_district[df_district['station'] == station]
    df_station['dis'] = pd.DataFrame(df_station['dis']).apply(normalize_s)
    plt.plot(pd.to_datetime(df_station['time']), df_station['dis'], label = 'Station: '+ station, linewidth = 1)

#df_station['max'] = pd.DataFrame(df_station['max']).apply(normalize_s)
#ax.plot(pd.to_datetime(df_station['time']), df_station['max'], label = 'Rainfall', linewidth = 1)
ax.plot(pd.to_datetime(df_station[df_station['flood'] == 1]['time']), df_station[df_station['flood'] == 1]['dis'], 'ko')
plt.xlim(datetime.date(2016, 4, 1), datetime.date(2016, 6, 28))
plt.ylabel('River discharge')
plt.legend(loc = 0)
plt.savefig(my_local_path + '/%s_stations.png' % dis)





