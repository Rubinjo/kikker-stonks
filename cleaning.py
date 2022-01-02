import pandas as pd
import numpy as np
from pandas.core.dtypes.generic import ABCDataFrame

def cleanData():
    data = pd.read_excel('./data/main/top10_30m_1y_v2.xlsx', header=0)
    data.rename(columns={'Unnamed: 0':'dateTime'}, inplace=True)
    data.rename(columns=lambda x: x.split('=')[0], inplace=True)

    data = data.loc[:, (data.isnull().sum() <= 2000)]

    for colname in data.columns:
        if colname!='dateTime':
            # calculating log return
            data[f'r{colname}'] = np.log(data[f'{colname}']/data[f'{colname}'].shift(-1)).dropna()
            # dropping original
            data.drop(f'{colname}', axis=1, inplace=True)
            # dropping last column because empty
            data.drop(data.tail(1).index, inplace=True)

    print(data.describe())

    data.to_csv('./data/main/top10_30m_1y_clean.csv')

def selectingXRows(lower, df, n):
    df.rename(columns={'Unnamed: 0':'index'}, inplace=True)

    data = df.loc[(df['index'] > lower) & (df['index'] <= lower + n)]

    neutralRange = .001 # up and down this is neutral
    labelValue = df.loc[df['index'] == lower].rBTC.values[0]

    if labelValue < 0 - neutralRange:
        label = 0
    elif labelValue > 0 + neutralRange:
        label = 2
    else:
        label = 1

    btc = data.rBTC.values.tolist()
    eth = data.rETH.values.tolist()
    ada = data.rADA.values.tolist()
    xrp = data.rXRP.values.tolist()
    dot = data.rDOT.values.tolist()
    avax = data.rAVAX.values.tolist()
    doge = data.rDOGE.values.tolist()

    dataList = ([btc, eth, ada, xrp, dot, avax, doge], label)
    return dataList

def splitingData():
    SPLITPERCENTAGE = .2
    df = pd.read_csv('./data/main/top10_30m_1y_clean.csv')

    n=32
    eindbaasList = []
    for i in range(len(df) - n + 1):
        correct = True
        rowItem = selectingXRows(i, df, n)
        for ticker in range(len(rowItem[0])):
            # for item in range(len(rowItem[0][ticker])):

            if (len(rowItem[0][ticker]) != n):
                correct = False
                print("falsy")
                break
        if (correct):
            eindbaasList.append(rowItem)
    return eindbaasList

if __name__=='__main__': 
    splitingData()


