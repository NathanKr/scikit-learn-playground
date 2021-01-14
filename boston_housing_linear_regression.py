from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




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

    fig, axs = plt.subplots(2,2)
    axLSTAT = axs[0,0]
    axB  = axs[1,0]
    axNOX = axs[0,1]
    axCHAS = axs[1,1]


    fig.suptitle('features vs output - MEDV')
    axLSTAT.plot(df_boston["LSTAT"],y,'.')
    axLSTAT.set_title("percentage lower status of the population - LSTAT")

    axB.plot(df_boston["B"],y,'.')
    axB.set_title("proportion of blacks by town - B")

    axNOX.plot(df_boston["NOX"],y,'.')
    axNOX.set_title("nitric oxides concentration (parts per 10 million) - NOX")


    axCHAS.plot(df_boston["CHAS"],y,'.')
    axCHAS.set_title(" Charles River dummy variable - CHAS")
    plt.show()

    fig, axs = plt.subplots(2,2)
    fig.suptitle('features vs output - MEDV')

    axPTRATIO = axs[0,0]
    axTAX = axs[0,1]
    axRAD = axs[1,0]
    axDIS = axs[1,1]

    axPTRATIO.plot(df_boston["PTRATIO"],y,'.')
    axPTRATIO.set_title("pupil-teacher ratio by town - PTRATIO")

    axTAX.plot(df_boston["TAX"],y,'.')
    axTAX.set_title("full-value property-tax rate per $10,000 - TAX")

    axRAD.plot(df_boston["RAD"],y,'.')
    axRAD.set_title("index of accessibility to radial highways - RAD")


    axDIS.plot(df_boston["DIS"],y,'.')
    axDIS.set_title("weighted distances to five Boston employment centres - DIS")
    plt.show()


    fig, axs = plt.subplots(2,2)
    fig.suptitle('features vs output - MEDV')
    
    axAGE = axs[0,0]
    axINDUS = axs[0,1]
    axCRIM= axs[1,0]
    axRM = axs[1,1]

    axAGE.plot(df_boston["AGE"],y,'.')
    axAGE.set_title("proportion of owner-occupied units built prior to 1940 - AGE")

    axRM.plot(df_boston["RM"],y,'.')
    axRM.set_title("average number of rooms per dwelling - RM")

    axINDUS.plot(df_boston["INDUS"],y,'.')
    axINDUS.set_title("proportion of non-retail business acres per town. - INDUS")

    
    axCRIM.plot(np.log10(df_boston["CRIM"]),y,'.')
    axCRIM.set_title("per capita crime rate by town log10('CRIM')")
    plt.show()


    plt.plot(df_boston["ZN"],y,'.')
    plt.title("proportion of residential land zoned for lots over 25,000 sq.ft - ZN")
    plt.show()


def all_training():
    print("******** all_training")
    reg = LinearRegression().fit(X, y) # same with LinearRegression(normalize="True")
    print("score : {:.2f} (1 is the best)".format(reg.score(X, y)))

    # reg = LinearRegression(normalize="True").fit(X, y)
    # print(reg.score(X, y))

    #print("reg.coef_ : ",reg.coef_)
    print("reg.predict(X[0,:])  / y[0] : " , reg.predict([X[0,:]]) / y[0])
    print("reg.predict(X[1,:])  / y[1] : " , reg.predict([X[1,:]]) / y[1])
    print("reg.predict(X[2,:])  / y[2] : " , reg.predict([X[2,:]]) / y[2])


def train_test():
    print("******** train_test")
    # random_state=42 => to get same result every run , you can pick other number
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)
    reg = LinearRegression().fit(X_train, y_train) 
    print("score train : {:.2f} (1 is the best)".format(reg.score(X_train, y_train)))
    print("score test : {:.2f} (1 is the best)".format(reg.score(X_test, y_test)))

def learning_curves():
    # random_state=42 => to get same result every run , you can pick other number
    train_score_lc = []
    test_score_lc = []
    i_lc =[]
    step = 5
    i = step
    while i < y.size:
        X_lc = X[:i,]
        y_lc = y[:i]
        X_train, X_test, y_train, y_test = train_test_split(X_lc,y_lc ,random_state=42)
        reg = LinearRegression().fit(X_train, y_train) 
        train_score_lc.append(reg.score(X_train, y_train))
        test_score_lc.append(reg.score(X_test, y_test))
        i_lc.append(i)
        i += step


    plt.plot(i_lc,train_score_lc,'red',i_lc,test_score_lc,'green')
    plt.title('learning curves : score (1 is best) train - red , test - green')
    plt.xlabel('data set points')
    plt.ylabel('score')
    plt.ylim((0, 1)) # i did this to remove huge numbers , but why does it happens
    plt.show()


plots()
all_training()
train_test()
learning_curves()