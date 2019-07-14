import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignment 2\data.csv',header=None)
df.columns = ['Wife', 'Husband', 'Rvalue']
df['Pvalue'] = np.nan
df['Wife'] = pd.to_numeric(df['Wife'])
df['Husband'] = pd.to_numeric(df['Husband'])
rad = 2.0

for b in range(2,10):
    TPositive = 0
    TNegative = 0
    FPositive = 0
    FNegative = 0
    s1=0
    s2=0
    for index, rows in df.iterrows():
        distance = {}
        swife = rows['Wife']
        shusband = rows['Husband']
        for i, row in df.iterrows():
            w1 = row['Wife']
            h1 = row['Husband']
            d = pow(((w1 - swife) ** 2 + (h1 - shusband) ** 2), 0.5)
            if d != 0:
                distance[i] = d
        n = sorted(distance.values())
        nei = {}
        q = {v: k for k, v in distance.items()}
        for i in range(b):
            ele = n[i]
            nei[ele] = q[ele]
        val = []
        for i in nei.keys():
            val.append(nei[i])
        for i, d2 in df.iterrows():
            if i in val and d2['Rvalue'] == 1 and distance[i] < rad:
                s1 += distance[i]
            elif i in val and d2['Rvalue'] == 0 and distance[i] < rad:
                s2 += distance[i]
        if s1 > s2:
            rows['Pvalue'] = 1
        else:
            rows['Pvalue'] = 0

        if rows['Pvalue'] == rows['Rvalue'] and rows['Pvalue'] == 1:
            TPositive += 1
        elif rows['Pvalue'] == rows['Rvalue'] and rows['Pvalue'] == 0:
            TNegative += 1
        elif rows['Pvalue'] != rows['Rvalue'] and rows['Pvalue'] == 1:
            FPositive += 1
        elif rows['Pvalue'] != rows['Rvalue'] and rows['Pvalue'] == 0:
            FNegative += 1
    acc=(TPositive+TNegative)/399
    print(b)
    print(TPositive)
    print(TNegative)
    print(FPositive)
    print(FNegative)
    print(acc)


