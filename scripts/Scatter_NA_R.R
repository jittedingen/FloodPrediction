##### MASTER THESIS - SCATTER OF MISSING VALUES ######
rm(list = ls())
library(VIM)

#df = read.csv("C:\\Users\\Jitte\\PycharmProjects\\Floods\\IBF_TriggerModel_Flood-rainfall_v12\\df_total.csv", header=T, sep=",")

df = read.csv("C:\\Users\\Jitte\\PycharmProjects\\Floods\\IBF_TriggerModel_Flood-rainfall_v12\\df_panel.csv", header=T, sep=",")
df_imp = read.csv("C:\\Users\\Jitte\\PycharmProjects\\Floods\\IBF_TriggerModel_Flood-rainfall_v12\\df_interpolated.csv", header=T, sep=",")
df_district = subset(df, df$district == 'kilifi')
df_district_imp = subset(df_imp, df_imp$district == 'kilifi')

df_district = df_district[-1,] #remove first row with NA's for soilM
df_district_imp = df_district_imp[-1,] #remove first row with NA's for soilM

selection = c('dis_station1', 'flood', 'rainfall', 'SoilMoi0_10cm_inst_mean', 'SoilMoi10_40cm_inst_mean', 'SoilMoi40_100cm_inst_mean', 'SoilMoi100_200cm_inst_mean')
df_district = subset(df_district, select = selection)
df_district_imp = subset(df_district_imp, select = selection)


# Get logical TRUE/FALSE indicating if value was imputed or not
df_district_imp_log = is.na(df_district)
colnames(df_district_imp_log) <- paste(colnames(df_district_imp_log), 'imp', sep = "_")
df_scatter = cbind(df_district_imp, df_district_imp_log)


#Scatter matrix on district level for imputed values
scattmatrixMiss(df_scatter, delimiter = '_imp',  highlight = c('SoilMoi0_10cm_inst_mean'), col = c("skyblue", "red", "orange"))

#Scatter matrix on district level for missing values
scattmatrixMiss(df_district, highlight = 'SoilMoi0_10cm_inst_mean')

#Scatter matrix on panel level
#scattmatrixMiss(df, highlight = 'SoilMoi0_10cm_inst_mean')



 ######### TRY OUT
data(sleep, package = "VIM")
x_imp <- kNN(sleep[, 1:5])
x_imp[,c(1,2,4)] <- log10(x_imp[,c(1,2,4)])
scattmatrixMiss(x_imp, delimiter = "_imp", highlight = "Dream")