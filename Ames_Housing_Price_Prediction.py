import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

##Importing data set 
raw_train = pd.read_csv("./train.csv")
raw_test  = pd.read_csv("./test.csv") 

train = raw_train
test  = raw_test 
            
##Adding Variable classification (categorical/discret/continous) information to input data
var_classification = pd.read_csv("./data_analysis.csv")
        
#Changing data types of year and month column from 'int64' to 'object'
#This change is necessary as we are dealing year and month as ordinal variables
for col in var_classification.loc[var_classification.iloc[:,1] == 'categorical','Column_Name']:
    if(train[col].dtypes not in ['object']):
        train[col] = train[col].astype('object')
        
for col in var_classification.loc[var_classification.iloc[:,1] == 'categorical','Column_Name']:
    if(test[col].dtypes not in ['object']):
        test[col] = train[col].astype('object')
        
#Handling null values in data
#Replacing null values with mean of the column for numerical features
for col in train.columns.values: 
    if(train[col].isnull().sum()):
        if(train[col].dtypes in ('float64','int64')):
            train[col] = train[col].fillna(train[col].mean(skipna = True))
        else: 
            train[col] = train[col].fillna(str("New_Cat"+"_"+col))
            #print(col,' : ', train[col].dtypes, str("New_Cat"+"_"+col))
            
for col in test.columns.values: 
    if(test[col].isnull().sum()):
        if(test[col].dtypes in ('float64','int64')):
            test[col] = test[col].fillna(test[col].mean(skipna = True))
        else: 
            test[col] = test[col].fillna(str("New_Cat"+"_"+col))
        
            
#Finding ordinal variables which are having null values 
#These null values are being replcaed by a new category which is at the top of the pyramid
# for col in var_classification[var_classification.Categorical_Type == 'ordinal'].Column_Name:
#     if(train[col].str.match('New_Cat').sum() != 0): 
#         print(col)

#For each nominal variable, low frequent elements (5% rule) are being merged into a single new category
#This method is effective in reducing the categories and hence making our model more effective
nominal_mod_col = []
freq_count = 0.05
for col in var_classification[var_classification.Categorical_Type == 'nominal'].Column_Name:
    frequency_arr = pd.DataFrame(train[col].value_counts()/train[col].value_counts().sum())
    count = frequency_arr[~(frequency_arr[col] > freq_count)].index.values
    train.loc[train[col].isin(count),col] = str('New_Cat'+'_'+col)
    
    frequency_arr = pd.DataFrame(test[col].value_counts()/test[col].value_counts().sum())
    count = frequency_arr[~(frequency_arr[col] > freq_count)].index.values
    test.loc[test[col].isin(count),col] = str('New_Cat'+'_'+col)
    
    
for col in var_classification[var_classification.Categorical_Type == 'nominal'].Column_Name:
    #converting noinal variables to numerical
    le = LabelEncoder()
    train[col+'_'+'le_col']       = le.fit_transform(train[col])
    le_col_names     = le.classes_
    #print(le_col_names)
    nominal_mod_col.append(str(col+'_'+'le_col'))
    
    #then converting these categorical variables from numerical to one hot vectors
    ohe = OneHotEncoder()
    ohe_labels       = list(col+'_'+str(x) for x in le.classes_)
    #print(ohe_labels)
    ohe_features_arr = ohe.fit_transform(train[[col]]).toarray()
    ohe_features     = pd.DataFrame(ohe_features_arr, columns=ohe_labels)
    
    train = pd.concat([train, ohe_features], axis = 1)
    
    le = LabelEncoder()
    test[col+'_'+'le_col']       = le.fit_transform(test[col])
    le_col_names     = le.classes_
    #print(le_col_names)
    
    #then converting these categorical variables from numerical to one hot vectors
    ohe = OneHotEncoder()
    ohe_labels       = list(col+'_'+str(x) for x in le.classes_)
    #print(ohe_labels)
    ohe_features_arr = ohe.fit_transform(test[[col]]).toarray()
    ohe_features     = pd.DataFrame(ohe_features_arr, columns=ohe_labels)
    
    test = pd.concat([test, ohe_features], axis = 1)
    #test_nominal_mod_col.append(str(col+'_'+'le_col'))
    
##Handling Categorical Variables 

#Training data
# Alley
train.Alley.replace({'Grvl':1, 'Pave':2, 'New_Cat_Alley':3}, inplace=True)

# Lot Shape
train.LotShape.replace({'Reg':1, 'IR1':2, 'IR2':3, 'IR3':4}, inplace=True)

# Land Contour
train.LandContour.replace({'Low':1, 'HLS':2, 'Bnk':3, 'Lvl':4}, inplace=True)

# Utilities
train.Utilities.replace({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}, inplace=True)

# Land Slope
train.LandSlope.replace({'Sev':1, 'Mod':2, 'Gtl':3}, inplace=True)

# Exterior Quality
train.ExterQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

# Exterior Condition
train.ExterCond.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

# Basement Quality
train.BsmtQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_BsmtQual':6}, inplace=True)

# Basement Condition
train.BsmtCond.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5,'New_Cat_BsmtCond':6}, inplace=True)

# Basement Exposure
train.BsmtExposure.replace({'No':1, 'Mn':2, 'Av':3, 'Gd':4, 'New_Cat_BsmtExposure':5}, inplace=True)

# Finished Basement 1 Rating
train.BsmtFinType1.replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6, 'New_Cat_BsmtFinType1':7}, inplace=True)

# Finished Basement 2 Rating
train.BsmtFinType2.replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6, 'New_Cat_BsmtFinType2':8}, inplace=True)

# Heating Quality and Condition
train.HeatingQC.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

# Kitchen Quality
train.KitchenQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

# Home functionality
train.Functional.replace({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8}, inplace=True)

# Fireplace Quality
train.FireplaceQu.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_FireplaceQu':6}, inplace=True)

# Garage Finish
train.GarageFinish.replace({'Unf':1, 'RFn':2, 'Fin':3,'New_Cat_GarageFinish':4}, inplace=True)

# Garage Quality
train.GarageQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5,'New_Cat_GarageQual':6}, inplace=True)

# Garage Condition
train.GarageCond.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_GarageCond':6}, inplace=True)

# Paved Driveway
train.PavedDrive.replace({'N':1, 'P':2, 'Y':3}, inplace=True)

# Pool Quality
train.PoolQC.replace({'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_PoolQC':6}, inplace=True)


#handling Test categorical ordinal variables
# Alley
test.Alley.replace({'Grvl':1, 'Pave':2, 'New_Cat_Alley':3}, inplace=True)

# Lot Shape
test.LotShape.replace({'Reg':1, 'IR1':2, 'IR2':3, 'IR3':4}, inplace=True)

# Land Contour
test.LandContour.replace({'Low':1, 'HLS':2, 'Bnk':3, 'Lvl':4}, inplace=True)

# Utilities
test.Utilities.replace({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4, 'New_Cat_Utilities':5}, inplace=True)

# Land Slope
test.LandSlope.replace({'Sev':1, 'Mod':2, 'Gtl':3}, inplace=True)

# Exterior Quality
test.ExterQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

# Exterior Condition
test.ExterCond.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

# Basement Quality
test.BsmtQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_BsmtQual':6}, inplace=True)

# Basement Condition
test.BsmtCond.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5,'New_Cat_BsmtCond':6}, inplace=True)

# Basement Exposure
test.BsmtExposure.replace({'No':1, 'Mn':2, 'Av':3, 'Gd':4, 'New_Cat_BsmtExposure':5}, inplace=True)

# Finished Basement 1 Rating
test.BsmtFinType1.replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6, 'New_Cat_BsmtFinType1':7}, inplace=True)

# Finished Basement 2 Rating
test.BsmtFinType2.replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6, 'New_Cat_BsmtFinType2':8}, inplace=True)

# Heating Quality and Condition
test.HeatingQC.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

# Kitchen Quality
test.KitchenQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_KitchenQual':6}, inplace=True)

# Home functionality
test.Functional.replace({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8, 'New_Cat_Functional':9}, inplace=True)

# Fireplace Quality
test.FireplaceQu.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_FireplaceQu':6, 'New_Cat_FireplaceQu':7}, inplace=True)

# Garage Finish
test.GarageFinish.replace({'Unf':1, 'RFn':2, 'Fin':3,'New_Cat_GarageFinish':4}, inplace=True)

# Garage Quality
test.GarageQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5,'New_Cat_GarageQual':6}, inplace=True)

# Garage Condition
test.GarageCond.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_GarageCond':6}, inplace=True)

# Paved Driveway
test.PavedDrive.replace({'N':1, 'P':2, 'Y':3}, inplace=True)

# Pool Quality
test.PoolQC.replace({'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'New_Cat_PoolQC':6}, inplace=True)


#Now Transforming ordinal variables to one-hot vectors
for col in var_classification[var_classification.Categorical_Type == 'ordinal'].Column_Name:
    #converting noinal variables to numerical
    le = LabelEncoder()
    temp       = le.fit_transform(train[col])
    le_col_names     = le.classes_
    #print(le_col_names)
    
    #then converting these categorical variables from numerical to one hot vectors
    ohe = OneHotEncoder()
    ohe_labels       = list(col+'_'+str(x) for x in le.classes_)
    #print(ohe_labels)
    ohe_features_arr = ohe.fit_transform(train[[col]]).toarray()
    ohe_features     = pd.DataFrame(ohe_features_arr, columns=ohe_labels)
    
    train = pd.concat([train, ohe_features], axis = 1)
    
    le = LabelEncoder()
    temp       = le.fit_transform(test[col])
    le_col_names     = le.classes_
    #print(le_col_names)
    
    #then converting these categorical variables from numerical to one hot vectors
    ohe = OneHotEncoder()
    ohe_labels       = list(col+'_'+str(x) for x in le.classes_)
    #print(ohe_labels)
    ohe_features_arr = ohe.fit_transform(test[[col]]).toarray()
    ohe_features     = pd.DataFrame(ohe_features_arr, columns=ohe_labels)
    
    test = pd.concat([test, ohe_features], axis = 1)
    
    
    
##Now data is prepared...we need to select relvant features for model training and testing (i.e filtering non-nuerical features)
non_numerical_col = []
non_numerical_col.extend(var_classification[var_classification.Categorical_Type == 'nominal'].Column_Name.values)
non_numerical_col.extend(var_classification[var_classification.Categorical_Type == 'ordinal'].Column_Name.values)
non_numerical_col.extend(nominal_mod_col)
non_numerical_col.append('SalePrice')
non_numerical_col.append('Id')
non_numerical_col

train_mod = train.loc[:,~(np.isin(train.columns.values, non_numerical_col))]
test_mod  = test.loc[:,~(np.isin(test.columns.values, non_numerical_col))]

x_train = train_mod
y_train = train.loc[:,train.columns == 'SalePrice']
x_test  = test_mod

####Spliting train data to training and validation data 

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3, random_state = 50)

#Visualizing the distribution of SalesPrice before normalizing 
sns.set(rc={'figure.figsize':(10,10)})
sns.distplot(y_train, bins=30)
plt.show()

non_numerical_col = []
non_numerical_col.extend(var_classification[var_classification.Categorical_Type == 'nominal'].Column_Name.values)
non_numerical_col.extend(var_classification[var_classification.Categorical_Type == 'ordinal'].Column_Name.values)
non_numerical_col.extend(nominal_mod_col)
#non_numerical_col.append('SalePrice')
non_numerical_col.append('Id')
non_numerical_col

train_mod1         = train.loc[:,~(np.isin(train.columns.values, non_numerical_col))]
correlation_matrix = train_mod1.corr().round(2)
SalesPrice_Corr    = correlation_matrix['SalePrice']
relevant_var       = SalesPrice_Corr[abs(SalesPrice_Corr) >= 0.25]
relevant_var_corr_mat = correlation_matrix.loc[(abs(SalesPrice_Corr) > 0.25),relevant_var.index.values]
relevant_feat = np.delete(relevant_var.index.values, np.where(relevant_var.index.values == 'SalePrice'))

x_train_final = x_train.loc[:,relevant_feat].reindex()
#x_val_final   = x_val.loc[:,relevant_feat].reindex()
x_test_final  = x_test.loc[:,relevant_feat].reindex()

for col in x_test_final.columns.values: 
    if(x_test_final[col].isnull().sum()):
            x_test_final[col] = 0
            
model = LinearRegression()
model.fit(x_train, y_train)

#Prdeciting the values from test data 
y_predicted = model.predict(x_train)
y_predicted = np.reshape(y_predicted, (-1,1))

#Error of the model (sanity check of our model)
mse = mean_squared_error(y_train, y_predicted)
mae = mean_absolute_error(y_train, y_predicted)
evs = explained_variance_score(y_train, y_predicted)

print("mse : {0} | mae : {1} | evs : {2}".format(mse,mae,evs))

#Visualizing the results of the model 

#Plotting residual Vs predicted Values
residual = (y_predicted-y_train)
# plt.plot([0,0,0],'r-')
# plt.axis([np.min(y_Predicted), np.max(y_Predicted), np.min(residual), np.max(residual)])
plt.scatter(y_predicted, residual)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

y_predicted = model.predict(x_test)
np.savetxt('Prediction_With_RandomForestTress.csv', y_predicted)