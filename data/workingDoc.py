import pandas as pd
import numpy as np

def cleanData():
    data = pd.read_excel('./crypto/data/top10_30m_1y_v2.xlsx', header=0)
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

    data.to_csv('./crypto/data/top10_30m_1y_clean.csv')


def calculateMeans():
    data = pd.read_csv('./crypto/data/top10_30m_1y_clean.csv', index_col=0)
    data['averageReturn1'] = data.mean(axis=1)
    return data


def calculateMeanDifference(data):
    print(data.dtypes)
    for colname in data.columns:
        if (colname != 'dateTime') & (colname != 'averageReturn1'):
            print(colname)
            # calculate return difference 1
            data[f'd{colname}1'] = data[f'{colname}'] - data['averageReturn1']
            data[f'd{colname}2'] = data[f'{colname}'] - data['averageReturn1'].shift(-1)
            data[f'd{colname}3'] = data[f'{colname}'] - data['averageReturn1'].shift(-2)
    print(data)
    print(data.describe())


data = calculateMeans()
calculateMeanDifference(data)