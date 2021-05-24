### FUNCTION THAT IS USED TO INTERPOLATE

from Load_Data import get_district_level
import pandas as pd

def interpolate(district, df_total):
    df_district = get_district_level(district, df_total)
    df_district = df_district.reset_index(drop=True)
    df_district['time'] = pd.to_datetime(df_district['time'], format='%Y-%m-%d')

    # Get columns with NA
    cols_NA = df_district.columns[df_district.isna().any()]
    n_NA = df_district[cols_NA].isna().sum()
    sorted_cols = n_NA.sort_values(ascending=True).index #first interpolate the ones with the least amount of NaNs

    for col in sorted_cols:
        ts = df_district.set_index('time')[col]
        interpol = ts.interpolate(method='spline', order=3) #interpolate
        df_district[col] = interpol.reset_index()[col]
    return df_district

if __name__ == "__interpolate__":
    import plac
    plac.call(interpolate)