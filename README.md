# Ames_House_Price-Prediction-Using-Linear-Regression
In this project, I have tried to predict house prices using Linear Regression Model. 

## 1. Dataset 
I have used [Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf). This dataset contains 79 variables describing almost every aspect of residential homes in Ames, Iowa, United States. Dataset contains categorical(nominal & ordinal), discrete and continuous variables and also have missing values for some variables for train & test dataset. So, first step is to pre-process the available data. 

### 1.1 Data pre-processing
Since dataset contains categorical, discrete & continuous variables, so I have created a [csv file] (https://github.com/tomar840/Ames_House_Price-Prediction-Using-Linear-Regression/blob/master/data_analysis.csv) containing the information for each variable. This file has helped me in handling missing values for each variables and also for handling categorical variables. 

#### 1.1.1 Handling Missing Values 
* Missing values for continuous & discrete variables are replaced by **mean value** of that variable while the same have been replaced by a new category for categorical variables. If you wish to change this methadology then change [here] (https://github.com/tomar840/Ames_House_Price-Prediction-Using-Linear-Regression/blob/master/Ames_Housing_Price_Prediction.py#L33)

#### 1.1.2 Handling Categorical Variables
* Categorical variables in this dataset have two types *i.e.* **nominal** (doesn't have ordering sense) & **ordinal** (have ordering sense *e.g.* rating of furniture varying from poor to excelent). 

If a nominal categorical variable holds large number of categories, then it will unnecessarily increase the complexity of the model. To overcome this problem, I have used **Frequency Reduction** technique to club all the categories having frequency less than the defined frequency. If you want to change this methadology then you can change it [here](https://github.com/tomar840/Ames_House_Price-Prediction-Using-Linear-Regression/blob/master/Ames_Housing_Price_Prediction.py#L59). Nominal variables are then encoded to numerical values using [sklearn LabelEncoder] (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

Ordinal variables are dealt with manual identification of their order. This has been done [here] (https://github.com/tomar840/Ames_House_Price-Prediction-Using-Linear-Regression/blob/master/Ames_Housing_Price_Prediction.py#L104)

Now, all the variables are in numeric form but categorical variables which are converted to numbers might mislead the model. So, it's important to convert *transformed categorical variables* to one-hot encoded vectors using [sklearn OneHotEncoder] (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

Now, we have transformed all the variables to their final form to feed into the model. 

## 2. Model
I have used the very basic [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) to predict the price of houses. 

This  problem can also be found on [kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## 3. Scope of Improvement
You can use principal component analysis to find the relevant continuous features. I will update this portion after applying advanced regression techniques. 
