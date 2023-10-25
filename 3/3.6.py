import numpy as np
import random
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

result0 = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]

st1 = [-5, 1, 5, 12, -2, 32, 2, 3, 4, 5]
st2 = [-5, 2, 5, -2, -1, 23, 24, 6, 8, 9]
def calc_sig(str1,str2,k1,k2,k0):

    metka = str1 * k1 + str2 * k2+k0

    formula = 1/(1+math.exp(-metka))

    return np.round(formula)

def calc(str1,str2,k1,k2,k0):

    metka = str1 * k1 + str2 * k2+k0

    if(metka>=1):
        return 1
    else:
         return 0





k0_l=random.uniform(-1,1)
k1_l = random.uniform(-1,1)
k2_l = random.uniform(-1,1)
norma=0.5

m_precision_l=[]
m_Fscore_l=[]
m_recall_l=[]
for i in range(100):
    l=[]
    m=[]

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    b=1
    for j in range(len(st1)):
        label = calc(st1[j], st2[j], k1_l, k2_l,k0_l)
        # error: 0, -1, 1
        error0=(result0[j]- label)*norma
        error1 = (result0[j]- label)*norma*st1[j]
        error2 = (result0[j]- label ) * norma * st2[j]
        l.append(label)
        if (label == result0[j]):
            if (label == 1):
                tp += 1
            else:
                tn += 1
        else:

            if (label == 0) and (result0[j] == 1):
                fn += 1
            if (label == 1) and (result0[j] == 0):
                fp += 1

        k1_l+= error1
        k2_l+= error2
        k0_l+=error0
    precision_l = tp / (tp + fp)
    recall_l = tp / (tp + fn)
    if (((1 + b) ^ 2) * tp + (b ^ 2) * fn + fp)==0:
        print('')
    Fscore_l = (((1 + b) ** 2) * tp) / (((1 + b) ** 2) * tp + (b ** 2) * fn + fp)
    print("Коэфф.",k0_l,k1_l,k2_l)
    print("result0",result0)
    print("result_",l)
    print("fp,fn,tp,tn",fp,fn,tp,tn)
    print("precision",precision_l)
    print("recall",recall_l)
    print("Fscore",Fscore_l)
    m_precision_l.append(precision_l)
    m_recall_l.append(precision_l)
    m_Fscore_l.append(precision_l)

print("======================================================================================"
      "======================================================================================"
      "======================================================================================")

k0_s=random.uniform(-1,1)
k1_s = random.uniform(-1,1)
k2_s = random.uniform(-1,1)

m_precision_s=[]
m_Fscore_s=[]
m_recall_s=[]
for i in range(100):
    l = []
    m = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    b = 1
    for j in range(len(st1)):
        label = calc_sig(st1[j], st2[j], k1_s, k2_s,k0_s)
        # error: 0, -1, 1
        error0=(result0[j]- label)*norma
        error1 = (result0[j]- label)*norma*st1[j]
        error2 = (result0[j]- label ) * norma * st2[j]
        l.append(label)
        if (label == result0[j]):
            if (label == 1):
                tp += 1
            else:
                tn += 1
        else:

            if (label == 0) and (result0[j] == 1):
                fn += 1
            if (label == 1) and (result0[j] == 0):
                fp += 1
        k1_s+= error1
        k2_s+= error2
        k0_s+=error0
    precision_s = tp / (tp + fp)
    recall_s = tp / (tp + fn)
    if (((1 + b) ^ 2) * tp + (b ^ 2) * fn + fp) == 0:
        print('')
    Fscore_s = (((1 + b) ** 2) * tp) / (((1 + b) ** 2) * tp + (b ** 2) * fn + fp)

    print("Коэфф.",k0_s,k1_s,k2_s)
    print("result0",result0)
    print("result_",l)
    print("fp,fn,tp,tn", fp, fn, tp, tn)
    print("precision",precision_s)
    print("recall",recall_s)
    print("Fscore",Fscore_s)
    m_precision_s.append(precision_s)
    m_Fscore_s.append(Fscore_s)
    m_recall_s.append(precision_s)

newDF=pd.DataFrame({'precision1':m_precision_l,'precision2':m_precision_s})
plt.figure(1)
sns.lineplot(data=newDF)
plt.title("Оценка precision")
plt.xlabel("Эпохи")
plt.ylabel("precision")
plt.show()

newDF1=pd.DataFrame({'recall1':m_recall_l,'recall2':m_recall_s})
plt.figure(2)
sns.lineplot(data=newDF1)
plt.title("Оценка recall")
plt.xlabel("Эпохи")
plt.ylabel("recall")
plt.show()

newDF2=pd.DataFrame({'Fscore1':m_Fscore_l,'Fscore2':m_Fscore_s})
plt.figure(3)
sns.lineplot(data=newDF2)
plt.title("Оценка Fscore")
plt.xlabel("Эпохи")
plt.ylabel("Fscore")
plt.show()



