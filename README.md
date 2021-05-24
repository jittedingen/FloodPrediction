# Flood prediction model for Kenya & Uganda 
## Thesis Jitte Dingenouts

This repository contains the code that was used in Jitte Dingenouts' thesis about flood prediction in Kenya and Uganda. It also includes her thesis for further explanation. Note that sometimes in the script you will see: "Panel level model" which is a synonym for the "Country Level model" as described in the thesis.

### Folders
The location specific folders (e.g. kenya) consists of an input and output folder. The input folder contains information about the location such as river discharge, rainfall and soil moisture. The output folder contains some data analysis results. Besides that, the Results folder contains the results for several models & scripts contains all scripts that are used in this thesis. The other folders come from the original benchmark of 510 & are barely used.

### Scripts
In the scripts folder you can find all the scripts that were used in this thesis. 
- Main.py: this is the main script which should be run to get the results. In this script, several other scripts are called.
- Interpolation.py: contains a function which interpolates the missing values that are present in the data
- Extra_Features.py: contains a function which constructs additional features from the data (cross-product of rainfall and soilmoisture closest to the surface / cross-product of the soil moisture level 4 (deepest in the ground) and level 2 / season indicator / for country level models include additional information on vegetation, percentage of water in district, indicator for how mountaineous a district or county is, the maximum river discharge over 2 weeks & a variable which indicates whether a flood occurred in the last 2 weeks.)
- Wavelet_Transform.py: contains a function which performs the Wavelet Transform to the data
- Imbalance.py: contains a function which makes the data more balanced
- District_Model.py: contains a function which performs a Logistic Regression and Support Vector Machine including cross-validation and returns the flood prediction results.
- Panel_ModelCV.py: contains a function which performs an LSTM model to the data and returns the flood prediction results for both overall performance, and performance per district. For the latter, the same model is used as for the first but we look at the performances of each district separately.
- Load_Data.py: contains a function which combines all the data that is available for Kenya and Uganda. It also contains a function get_district_level which returns all the information for a particular district. This function is for example used in the Interpolation.py script. In addition, the data quality is restricted (see explanation below about the df_totalhighqual.csv file for example) 
- BenchMark_Model.py: contains the modified version of the original benchmark model of 510. The original model is modified for a few reasons. More information about this can be found in the thesis.
- prep_WT.py: a script which can visualize the wavelet transform
- Data_Analysis.py: a script which can make several maps and visuals about the data such as heatmaps on the amount of floods in each district and a plot which shows the river discharge stations.
- Merge_soilmoisture.py: a script which merges the downloaded batches of soilmoisture.
- Results_Visualization.py: a messy script with many possibilities to make visualizations
- Retrieving_ExtraData.py: script that can extract data from GEE
- Sign_Test.py: script that performs the nonparametric sign test to the results to check significance. 
- gee_utils.py: script that can extract information from GEE that consists of several images
- GEE_get_individual_image.py: script that can extract information from GEE that consists of one image only.
- Plot_waves.R: script that plots waves that are used as explanation in the thesis.
- Scatter_NA_R.R: script that visualizes the missing and imputed values (see thesis)

Other scripts originate from the original benchmark model of 510 which I haven't used directly.


### Important data files
- SpatialMatrix.csv: a CSV document which contains only zeros and ones depending on whether districts are located next to each other. If two districts/counties are neighbours a one is denoted in their intersection cells.
- prep_WT_df.csv: a CSV document which can be used to visualize the wavelet transform in the prep_WT.py script.
- df_totalhighqual.csv: dataframe with all information about Kenya and Uganda that is used for modelling where the Uganda certainty of the flood occurrence (data_quality_score) is larger than 3 & for Kenya larger (data_quality) than 1.
- df_totalmediumqual.csv: dataframe with all information about Kenya and Uganda that is used for modelling where the Uganda certainty of the flood occurrence (data_quality_score) is larger than 3 & for Kenya larger (data_quality) than 0.
- df_total.csv: dataframe with all information about Kenya and Uganda that is used for modelling where no restriction is set to the certainty of the flood.
- uga_impact_data.csv: contains information on the impact of the floods that occurred in Uganda. In this file, you can find the certainty of the flood (data_quality_score).
- ken_impact_data.csv: contains information on the impact of the floods that occurred in Kenya. In this file, you can find the certainty of the flood (data_quality).

### Important remarks
If you want to extract information from Google Earth Engine (GEE), you need an account. 
Go to "https://code.earthengine.google.com/" and check whether you can sign in. If not, fill in the form to create an account. 

Besides that, you will need a GloFAS credentials file (settings.cfg), which you can get from Jacopo. Please set this file in the main folder (where you can also find the spatial matrix file.)
