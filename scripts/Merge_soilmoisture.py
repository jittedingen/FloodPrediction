############# MERGE SOIL_MOISTURE DATA #############
import pandas as pd

veg = True
if veg == True:
    # kenya
    df_1 = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\_vegetation\MODIS_2000-2005.csv')
    df_2 = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\_vegetation\MODIS_2006-2010.csv')
    df_3 = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\_vegetation\MODIS_2011-2015.csv')
    df_4 = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\_vegetation\MODIS_2016-2019.csv')

    df_total = pd.concat([df_1, df_2, df_3, df_4])

    # uganda
    df_1U = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\_vegetation\MODIS_2000-2005.csv')
    df_2U = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\_vegetation\MODIS_2006-2010.csv')
    df_3U = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\_vegetation\MODIS_2011-2015.csv')
    df_4U = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\_vegetation\MODIS_2016-2019.csv')

    df_totalUga = pd.concat([df_1U, df_2U, df_3U, df_4U])

    df_total.to_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\_vegetation\_vegetation_Kenya.csv')
    df_totalUga.to_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\_vegetation\_vegetation_Uganda.csv')

soil = False
if soil == True:
    #kenya
    soil_1 = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\soil_moisture\GDLAS_Kenyadata2000-2005.csv')
    soil_2 = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\soil_moisture\GDLAS_Kenyadata2006-2010.csv')
    soil_3 = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\soil_moisture\GDLAS_Kenyadata2011-2015.csv')
    soil_4 = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\soil_moisture\GDLAS_Kenyadata2016-2019.csv')

    soil_total = pd.concat([soil_1, soil_2, soil_3, soil_4])

    #uganda
    soil_1U = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\soil_moisture\GDLAS_data2000-2005.csv')
    soil_2U = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\soil_moisture\GDLAS_data2006-2010.csv')
    soil_3U = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\soil_moisture\GDLAS_data2011-2015.csv')
    soil_4U = pd.read_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\soil_moisture\GDLAS_data2016-2019.csv')

    soil_totalUga = pd.concat([soil_1U, soil_2U, soil_3U, soil_4U])

    soil_total.to_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\kenya\input\soil_moisture\soil_moisture_Kenya.csv')
    soil_totalUga.to_csv('~\PycharmProjects\Floods\IBF_TriggerModel_Flood-rainfall_v12\_uganda\input\soil_moisture\soil_moisture_Uganda.csv')

