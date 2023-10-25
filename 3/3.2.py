import numpy as np
import random

result0 = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]

st1 = [-5, 1, 5, 12, -2, 32, 2, 3, 4, 5]
st2 = [-5, 2, 5, -2, -1, 23, 24, 6, 8, 9]
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
    l=[]
    for j in range(len(st1)):
        label = calc(st1[j], st2[j], k1, k2,k0)
        # error: 0, -1, 1
        error0=(result0[j]- label)*norma
        error1 = (result0[j]- label)*norma*st1[j]
        error2 = (result0[j]- label ) * norma * st2[j]
        l.append(label)
        k1+= error1
        k2+= error2
        k0+=error0
    print("Коэфф.",k1,k2)
    print("result0",result0)
    print("result_",l)




