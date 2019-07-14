import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import neighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
data = pd.read_csv(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignment 2\data.csv',header=None)
data.columns = ['Wife', 'Husband', 'Rvalue']
data['Wife'] = pd.to_numeric(data['Wife'])
data['Husband'] = pd.to_numeric(data['Husband'])
data = shuffle(data)
df_ref= pd.DataFrame(data.iloc[:300, :])
df_test = pd.DataFrame(data.iloc[300:, :])
df_test['Pvalue'] = 0
sensitivity = []
specificity=[]
ppv=[]
npv=[]
acc=[]
Hit_rate = []
TPositive = 0
TNegative = 0
FPositive = 0
FNegative = 0

for b in range(2,10):
    TPositive = 0
    TNegative = 0
    FPositive = 0
    FNegative = 0
    for index, rows in df_test.iterrows():
        distance = {}
        swife = rows['Wife']
        shusband = rows['Husband']
        for i, row in df_ref.iterrows():
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
        s = 0
        for ind, d2 in df_ref.iterrows():
            if ind in val:
                s += d2['Rvalue']
        if s > float(b>2):
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
    Hit_rate.append((TNegative + TPositive) / 99)
    sensitivity.append(TPositive / (TPositive + FNegative))
    specificity.append(TNegative / (FPositive + TNegative))
    ppv.append(TPositive / (TPositive + FPositive))
    npv.append(TNegative / (TNegative + FNegative))

names = ["Hit Rate", "Sensitivity", "Specificity", "PPV", "NPV"]
Ac_hit_rate = max(Hit_rate)
x = Hit_rate.index(Ac_hit_rate)
Ac_sensitivity = sensitivity[x]
Ac_specificity = specificity[x]
Ac_ppv = ppv[x]
Ac_npv = npv[x]
measures = [Ac_hit_rate, Ac_sensitivity, Ac_specificity, Ac_ppv, Ac_npv]
x +=2
ind = range(len(measures))
plt.bar(ind, measures, 0.3)
plt.xlabel('Metrics')
plt.ylabel('Metric Value')
plt.xticks(ind, names, fontsize=6)
plt.title('Bar Graph for Test Data, with k=%i' %x )
plt.show()


X = pd.DataFrame(df_ref.iloc[:, :2])
Y = pd.DataFrame(df_ref.iloc[:, 2])
n_neighbors = x
h=0.1
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y.iloc[:, 0], cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(" Classification (k = %i)" %n_neighbors)
plt.show()


















