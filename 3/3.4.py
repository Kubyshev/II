import numpy as np
import random
import math

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





k0=0
k1 = random.uniform(-1,1)
k2 = random.uniform(-1,1)
norma=0.5


for i in range(100):

    l = []
    m = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    b=1
    for j in range(len(st1)):
        label = calc(st1[j], st2[j], k1, k2,k0)
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

        k1+= error1
        k2+= error2
        k0+=error0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (((1 + b) ^ 2) * tp + (b ^ 2) * fn + fp)==0:
        print('')
    Fscore = (((1 + b) ** 2) * tp) / (((1 + b) ** 2) * tp + (b ** 2) * fn + fp)
    print("Коэфф.",k1,k2)
    print("result0",result0)
    print("result_",l)
    print("fp,fn,tp,tn",fp,fn,tp,tn)
    print("precision",precision)
    print("recall",recall)
    print("Fscore",Fscore)


print("=======================================")


for i in range(100):
    l = []
    m = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    b = 1
    for j in range(len(st1)):
        label = calc_sig(st1[j], st2[j], k1, k2,k0)
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
        k1+= error1
        k2+= error2
        k0+=error0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (((1 + b) ^ 2) * tp + (b ^ 2) * fn + fp) == 0:
        print('')
    Fscore = (((1 + b) ** 2) * tp) / (((1 + b) ** 2) * tp + (b ** 2) * fn + fp)

    print("Коэфф.",k1,k2)
    print("result0",result0)
    print("result_",l)
    print("fp,fn,tp,tn", fp, fn, tp, tn)
    print("precision",precision)
    print("recall",recall)
    print("Fscore",Fscore)







