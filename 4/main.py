import re

import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def ip_filter(string):
    if bool(re.search("^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$", string)):
        return int(''.join([bin(int(x)+256)[3:] for x in string.split('.')]),2)
    return string

df = pd.read_csv('../NF-UQ-NIDS-v2 250000.csv')
df.drop(labels=[df.columns[0],"Attack","Dataset"], axis=1, inplace=True)
df_sklearn = df.copy()
columns = []
[("ADDR" in x) and columns.append(x) for x in df.columns]
for column in columns:
    df_sklearn[column] = df_sklearn[column].apply(ip_filter)
train, test = train_test_split(df_sklearn, test_size= 0.5 )

def sig(net):
    return 1/(1+np.exp(-net))
def sig_der(x):
    return sig(x)*(1-sig(x))
def scalar_product(_str, k, k0):
    str = _str.copy()
    str.pop("Label")
    metka = sum([x*y for x,y in zip(str,k)])+k0
    return metka


m_precision_l=[]
m_Fscore_l=[]
m_recall_l=[]
k=[0 for _ in range(len(df.columns)-1)]
k0 = random.uniform(-1,1)
for i in range(3):
    l=[]
    m=[]

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    b=1
    norma = np.exp(-i)*0.5+0.01
    for index, row in train.iterrows():
        sp = scalar_product(row, k, k0)
        label = np.round(sig(sp))
        result = row['Label']

        # error: 0, -1, 1
        error=[-(result-label)*norma*x*sig_der(sp) for x in row]
        error_k0 = -(result-label)*norma*sig_der(sp)
        l.append(label)
        if (label == result):
            if (label == 1):
                tp += 1
            else:
                tn += 1
        else:

            if (label == 0) and (result == 1):
                fn += 1
            if (label == 1) and (result == 0):
                fp += 1
        if ((tp+fp+tn+fn)%1000)==0:
            print(tp+fp+tn+fn)
        k = [k-x for k,x in zip(k,error)]
        k0-=error_k0
    precision_l = tp / (tp + fp)
    recall_l = tp / (tp + fn)
    if (((1 + b) ^ 2) * tp + (b ^ 2) * fn + fp)==0:
        print('')
    Fscore_l = (((1 + b) ** 2) * tp) / (((1 + b) ** 2) * tp + (b ** 2) * fn + fp)
    print("Коэфф.",k0,*k)
    print("precision",precision_l)
    print("recall",recall_l)
    print("Fscore",Fscore_l)
    m_precision_l.append(precision_l)
print(*["="*100]*3,sep='\n')


tp = 0
tn = 0
fp = 0
fn = 0
b=1

for index, row in test.iterrows():
    sp = scalar_product(row, k, k0)
    label = np.round(sig(sp))
    result = row['Label']
    # error: 0, -1, 1
    l.append(label)
    if (label == result):
        if (label == 1):
            tp += 1
        else:
            tn += 1
    else:

        if (label == 0) and (result == 1):
            fn += 1
        if (label == 1) and (result == 0):
            fp += 1
recall_l = tp / (tp + fn)
if (((1 + b) ^ 2) * tp + (b ^ 2) * fn + fp)==0:
    print('')
Fscore_l = (((1 + b) ** 2) * tp) / (((1 + b) ** 2) * tp + (b ** 2) * fn + fp)
print("precision",precision_l)
print("recall",recall_l)
print("Fscore",Fscore_l)


newDF=pd.DataFrame({'precision1':m_precision_l})
plt.figure(1)
sns.lineplot(data=newDF)
plt.title("Оценка precision")
plt.xlabel("Эпохи")
plt.ylabel("precision")
plt.show()

