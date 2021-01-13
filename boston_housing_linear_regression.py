from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression



# probably the same as https://www.kaggle.com/vikrishnan/boston-house-prices
# and https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

# 14 coulmns
# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's


# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
ds_boston = load_boston()
X = ds_boston["data"]
y = ds_boston["target"]
print("ds_boston['data'].shape : ",X.shape)
df_boston = pd.DataFrame (X , columns = ds_boston["feature_names"] )
df_boston['MEDV'] = y 
print(df_boston.head())

def plots():
    # pairplot
    # takes too long and what is shown is not clear sns.pairplot(df_boston) 
    #plt.show()

    plt.plot(df_boston["LSTAT"],y,'o')
    plt.title("LSTAT vs MEDV")
    plt.show()

    plt.plot(df_boston["B"],y,'o')
    plt.title("B vs MEDV")
    plt.show()


    # how can i control the width
    # sns.barplot(x="PTRATIO", y="MEDV", data=df_boston)
    # plt.show()


    plt.plot(df_boston["PTRATIO"],y,'o')
    plt.title("PTRATIO vs MEDV")
    plt.show()

    plt.plot(df_boston["TAX"],y,'o')
    plt.title("TAX vs MEDV")
    plt.show()

    plt.plot(df_boston["RAD"],y,'o')
    plt.title("RAD vs MEDV")
    plt.show()


    plt.plot(df_boston["DIS"],y,'o')
    plt.title("DIS vs MEDV")
    plt.show()

    plt.plot(df_boston["AGE"],y,'o')
    plt.title("AGE vs MEDV")
    plt.show()

    plt.plot(df_boston["RM"],y,'o')
    plt.title("RM vs MEDV")
    plt.show()

    plt.plot(df_boston["NOX"],y,'o')
    plt.title("NOX vs MEDV")
    plt.show()


    plt.plot(df_boston["CHAS"],y,'o')
    plt.title("CHAS vs MEDV")
    plt.show()


    plt.plot(df_boston["INDUS"],y,'o')
    plt.title("INDUS vs MEDV")
    plt.show()


    plt.plot(df_boston["ZN"],y,'o')
    plt.title("ZN vs MEDV -> what is it , why so much on 0 ?????")
    plt.show()

    plt.plot(np.log10(df_boston["CRIM"]),y,'o')
    plt.title("log10('CRIM') vs MEDV")
    plt.show()


# plots()
reg = LinearRegression().fit(X, y) # same with LinearRegression(normalize="True")
print("score : {:.2f} (1 is the best)".format(reg.score(X, y)))

# reg = LinearRegression(normalize="True").fit(X, y)
# print(reg.score(X, y))

#print("reg.coef_ : ",reg.coef_)
print("reg.predict(X[0,:])  / y[0] : " , reg.predict([X[0,:]]) / y[0])
print("reg.predict(X[1,:])  / y[1] : " , reg.predict([X[1,:]]) / y[1])
print("reg.predict(X[2,:])  / y[2] : " , reg.predict([X[2,:]]) / y[2])