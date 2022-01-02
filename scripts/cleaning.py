import pandas as pd
import numpy as np

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

    neutralRange = .0019 # up and down this is neutral
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

    n=32 # amount of historic datapoints fed to network
    eindbaasList = []
    dataDict = {0: [], 1: [], 2: []}
    for i in range(len(df) - n + 1):
        correct = True
        rowItem = selectingXRows(i, df, n)

        
        # validating row
        for ticker in range(len(rowItem[0])):
            # for item in range(len(rowItem[0][ticker])):

            if (len(rowItem[0][ticker]) != n):
                correct = False
                break
        if (correct):
            eindbaasList.append(rowItem)

            tickers, label = rowItem
            dataDict[label].append(tickers)

    negCount = len(dataDict[0])
    neutralCount = len(dataDict[1])
    positiveCount = len(dataDict[2])

    minValue = min(negCount, neutralCount, positiveCount)

    newDict = {0: [], 1: [], 2: []}
    newDict[0] = dataDict[0][:minValue] 
    newDict[1] = dataDict[1][:minValue] 
    newDict[2] = dataDict[2][:minValue]  

    eindbaasList = []
    for label in range(3):
        for item in newDict[label]:
            eindbaasList.append((item, label))
    
    # df = pd.DataFrame.from_dict(dataDict)
    # df.to_csv('dataTest.csv')
    return eindbaasList

if __name__=='__main__': 
    splitingData()


