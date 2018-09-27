import os
import pandas as pd
import numpy as np

os.getcwd()
os.chdir('D:\Sukriti\Knowledge\Python')
#######################################################################################################################
################Reading Datasets for Crop Production, Temp, Rainfall and Soil PH Level################
crop_yield = pd.read_csv('crop_production.csv')
crop_yield.isnull().values.any()
crop_yield = crop_yield.fillna(0)
crop_yield['STATE_ID'] = crop_yield['STATE_ID'].astype(np.int64)
crop_yield = crop_yield.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  ## Stripping of blank spaces from 'object' datatype columns

rain_district = pd.read_excel('Rainfall_dataset_district.xlsx')
rain_district.isnull().values.any()
rain_district = rain_district.fillna(0)
rain_district = rain_district.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

temp = pd.read_excel('Temp_dataset.xlsx')
temp.isnull().values.any()
temp = temp.fillna(0)
temp = temp.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

soil = pd.read_excel('Soil_PH_data.xlsx')
soil.isnull().values.any()
soil = soil.fillna(0)
soil = soil.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

#######################################################################################################################
#################Subset of Rain Data#################

rain_District_State_Kharif = rain_district.loc[:, 'JUL':'OCT']
rain_District_State_Rabi = rain_district.loc[:, ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR']]
rain_District_State_Summer = rain_district.loc[:, ['APR', 'MAY', 'JUN']]
rain_District_State_Kharif['Total_Rain_Kharif'] = rain_District_State_Kharif.sum(axis=1)
rain_District_State_Rabi['Total_Rain_Rabi'] = rain_District_State_Rabi.sum(axis=1)
rain_District_State_Summer['Total_Rain_Summer'] = rain_District_State_Summer.sum(axis=1)
rain_District_State = pd.concat(
    [rain_District_State_Kharif['Total_Rain_Kharif'], rain_District_State_Rabi['Total_Rain_Rabi'],
     rain_District_State_Summer['Total_Rain_Summer']], axis=1)
rain_District_State['STATE_ID'] = rain_district.loc[:, 'STATE_ID']
rain_District_State['STATE'] = rain_district.loc[:, 'STATE']
rain_District_State['DISTRICT'] = rain_district.loc[:, 'DISTRICT']
rain_District_State['DISTRICT_ID'] = rain_district.loc[:, 'DISTRICT_ID']
rain_District_State['YEAR'] = rain_district.loc[:, 'YEAR']

rain_District_State = rain_District_State[
    (rain_District_State.T != 0).any()]  ###delete all rows where all column values are 0.

##########################################De-Normalizing temp and rain dataset based on Seasons in India##############

temp_test = temp.loc[:, ['STATE_ID', 'STATE', 'YEAR', 'Apr-Jun', 'Jul-Oct', 'Oct-Mar']]

Final_Temp = pd.melt(temp_test, id_vars=["STATE_ID", 'STATE', 'YEAR'],
                     var_name="Season", value_name="Avg_Temp")  # De-Normalized data for Temperature

temp_test = temp.loc[:, ['STATE_ID', 'STATE', 'DISTRICT_ID', 'DISTRICT', 'YEAR', 'Apr-Jun', 'Jul-Oct', 'Oct-Mar']]

Final_Rain = pd.melt(rain_District_State, id_vars=["STATE_ID", 'STATE', 'DISTRICT_ID', 'DISTRICT', 'YEAR'],
                     var_name="Season", value_name="Total_Rain")


#######################################################################################################################
################ Defining Season ID for Temp levels################
def season_id(row):
    if row["Season"] == 'Apr-Jun':
        return 1
    elif row["Season"] == 'Jul-Oct':
        return 2
    elif row["Season"] == 'Oct-Mar':
        return 3
    else:
        return 0


Final_Temp = Final_Temp.assign(SEASON_ID=Final_Temp.apply(season_id, axis=1))


def season_id1(row):
    if row["Season"] == 'Total_Rain_Summer':
        return 1
    elif row["Season"] == 'Total_Rain_Kharif':
        return 2
    elif row["Season"] == 'Total_Rain_Rabi':
        return 3
    else:
        return 0


Final_Rain = Final_Rain.assign(SEASON_ID=Final_Rain.apply(season_id1, axis=1))


#######################################################################################################################
################ Defining Drought Level during monsoon season################
def drought_con(row):
    if row["Total_Rain"] <= 750.00 and row["SEASON_ID"] == 2:
        return 1
    else:
        return 0


Final_Rain = Final_Rain.assign(DROUGHT=Final_Rain.apply(drought_con, axis=1))

#######################################################################################################################
#############################Selection of Rice as Crop###############################
Rice_yield = crop_yield[(crop_yield['CROP'] == "Rice")| (crop_yield['CROP'] == "Wheat")]
Rice_yield.dtypes
Rice_yield.info()
Rice_yield['DISTRICT_ID'] = Rice_yield['DISTRICT_ID'].astype(np.int64)  ###Changing the datatype to Int


################ Defining Season ID for Rice Yield ################
def season_id2(row):
    if row["SEASON"] == 'Apr-Jun':
        return 1
    elif row["SEASON"] == 'Jul-Oct':
        return 2
    elif row["SEASON"] == 'Oct-Mar':
        return 3
    else:
        return 0


Rice_yield = Rice_yield.assign(SEASON_ID=Rice_yield.apply(season_id2, axis=1))

################ Rice Yield State-Wise################
Total_Rice_yield = Rice_yield.groupby(['STATE_ID', 'YEAR', 'SEASON', 'STATE', 'CROP'], as_index=False)[
    ['PRODUCTION', 'AREA']].sum(axis=0)

Rice_yield_sort = pd.DataFrame(Total_Rice_yield.groupby(['STATE'], as_index=False)[[
    'PRODUCTION']].max())  ###Identify in which State Rice yield was maximum in last 5 years
Rice_yield_sort

################ ################ ################ ################ ################ ################ ################
################ Rice Yield yearly################
Rice_yield_year = Total_Rice_yield[(Total_Rice_yield['YEAR'] == 2017)]
Rice_yield_year_sort = pd.DataFrame(Rice_yield_year.groupby(['STATE'], as_index=False)[[
    'PRODUCTION']].max())  ###Identify in which State Rice yield was maximum for a particular year

ps = Rice_yield_year_sort.loc[[Rice_yield_year_sort['PRODUCTION'].idxmax()]]
ps
Rice_yield_State = ps.loc[:, ['STATE']]
St = str(Rice_yield_State.values).strip('[]').strip('\'')
############################################################################################################################################

#####Merging of Soil, Temp and Rain Datasets#######

Temp_Rain = pd.merge(Final_Rain, Final_Temp, how='inner', on=['STATE_ID', 'YEAR', 'SEASON_ID'])[
    ['STATE_ID', 'STATE_x', 'DISTRICT_ID', 'SEASON_ID', 'YEAR', 'Total_Rain', 'DROUGHT', 'Avg_Temp']]
Soil_Temp_Rain = pd.merge(Temp_Rain, soil, how='inner', on='STATE_ID')[
    ['STATE_ID', 'STATE_x', 'DISTRICT_ID', 'SEASON_ID', 'YEAR', 'Total_Rain', 'DROUGHT', 'Avg_Temp', 'Soil_Type',
     'Ph Level']]
Soil_Temp_Rain.dtypes
Soil_Temp_Rain.info()
Soil_Temp_Rain.isnull().values.any()
Soil_Temp_Rain = Soil_Temp_Rain.fillna(0)

#####Creation of final data with all the required parameters#######

Crop_Yield_overall = pd.merge(Rice_yield, Soil_Temp_Rain, how='inner', on=['STATE_ID', 'DISTRICT_ID', 'YEAR', 'SEASON_ID'])[
        ['STATE_ID', 'STATE_x', 'DISTRICT_ID', 'SEASON_ID', 'YEAR', 'Total_Rain', 'DROUGHT', 'Avg_Temp', 'Soil_Type',
         'Ph Level', 'AREA', 'PRODUCTION']]
Crop_Yield_overall.dtypes
Crop_Yield_overall.info()
Crop_Yield_overall.to_csv('Crop_Yield_overall.csv')

###################Categorising the Soil Type######################
Crop_Yield_overall.drop(['STATE_x', 'Ph Level'], axis=1, inplace=True)
cleanup_Soil = {"Soil_Type": {"Red": 1, "Alluvial": 2, "Laterite": 3, "Forest": 4,
                              "Saline/Alkaline": 5, "Mountain/Forest Soil": 6, "Desert/Red": 7, "Black": 8,
                              "Alluvial/Black": 9, "Red/Black": 10}}
Crop_Yield_overall.replace(cleanup_Soil, inplace=True)
#############################################################################

from sklearn import preprocessing

norm_prod = Crop_Yield_overall[['AREA', 'PRODUCTION']]
# mean=norm_prod.mean()
mean_AREA = norm_prod['AREA'].mean()
# std=norm_prod.std()
std_AREA = norm_prod['AREA'].std()
normalized_df = (
                        norm_prod - norm_prod.mean()) / norm_prod.std()  ## normalizing the Area and Production value based on z-score algorithm
df = Crop_Yield_overall.loc[:, ['SEASON_ID', 'YEAR', 'Total_Rain', 'DROUGHT', 'Avg_Temp', 'Soil_Type', ]]
norm_df = pd.concat([df, normalized_df], axis=1)

X = norm_df.loc[:, ['Avg_Temp', 'SEASON_ID', 'Soil_Type', 'AREA', 'Total_Rain']]
Y = norm_df.loc[:, 'PRODUCTION']

norm_df.describe()
norm_df.to_csv('Normalized_crop_prod_overall.csv')

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
accuracy = regressor.score(X_test,Y_test)


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
regressor.coef_[3]
y_pred = regressor.predict(X_test)
Eval_test = pd.DataFrame({'Actual Production': Y_test, 'Predicted Production': y_pred})
Evaluation = pd.concat([Crop_Yield_overall['YEAR'], Crop_Yield_overall['AREA'], Eval_test], axis=1)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print('Coefficient of Determination(R2):', metrics.r2_score(Y_test, y_pred))
print('Accuracy is :',accuracy*100,'%')