import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('Раздаточный материал ПЗ-5.csv')
df.drop(labels=[df.columns[0]], axis=1, inplace=True)
df_sklearn = df.copy()
# apply normalization techniques
for i in range(1,29):
    column = "A-"+str(i)
    df_sklearn[column] = MinMaxScaler().fit_transform(np.array(df_sklearn[column]).reshape(-1, 1))
train, test = train_test_split(df_sklearn, test_size= 0.5 )

def calc_sig(_str, k, k0):
    str = _str.copy()
    str.pop("S")
    metka = sum([x*y for x,y in zip(str,k)])+k0
    formula = 1/(1+np.exp(-metka))
    return np.round(formula)

m_precision_l=[]
m_Fscore_l=[]
m_recall_l=[]
k=[0 for _ in range(len(df.columns)-1)]
k0 = random.uniform(-1,1)
for i in range(10):
    l=[]
    m=[]

    tp = tn = fp = fn = 0
    b=1
    norma = np.exp(-i)
    for index, row in train.iterrows():
        label = calc_sig(row, k, k0)
        result = row['S']
        # error: 0, -1, 1
        error=[(result-label)*norma*x for x in row]
        error_k0 = (result-label)*norma
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
        k = [k+x for k,x in zip(k,error)]
        k0+=error_k0
    precision_l = tp / (tp + fp)
    recall_l = tp / (tp + fn)
    if (((1 + b) ^ 2) * tp + (b ^ 2) * fn + fp)==0:
        print('')
    Fscore_l = (((1 + b) ** 2) * tp) / (((1 + b) ** 2) * tp + (b ** 2) * fn + fp)
    print("Коэфф.",k0,*k)
    print("result",result)
    print("result_",l)
    print("fp,fn,tp,tn",fp,fn,tp,tn)
    print("precision",precision_l)
    print("recall",recall_l)
    print("Fscore",Fscore_l)
    m_precision_l.append(precision_l)
    m_recall_l.append(precision_l)
    m_Fscore_l.append(precision_l)
    precision_l = tp / (tp + fp)
print(*["="*100]*3,sep='\n')


tp = 0
tn = 0
fp = 0
fn = 0
b=1
for index, row in test.iterrows():
    label = calc_sig(row, k, k0)
    result = row['S']
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
print("result",result)
print("result_",l)
print("fp,fn,tp,tn",fp,fn,tp,tn)
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

newDF1=pd.DataFrame({'recall1':m_recall_l})
plt.figure(2)
sns.lineplot(data=newDF1)
plt.title("Оценка recall")
plt.xlabel("Эпохи")
plt.ylabel("recall")
plt.show()

newDF2=pd.DataFrame({'Fscore1':m_Fscore_l})
plt.figure(3)
sns.lineplot(data=newDF2)
plt.title("Оценка Fscore")
plt.xlabel("Эпохи")
plt.ylabel("Fscore")
plt.show()