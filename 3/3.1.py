#import pandas as pd

#rating = pd.read_csv('NF-UQ-NIDS-v2.csv')
#rating[:200000].to_csv('workII.csv', index=False)

import numpy as np
test = [-5,1,5,12,-2,32,2,3,4,5]
k1= 1
vector = np.array(test)
test2 = [-5,2,5,-2,-1,23,24,6,8,9]
k2 = -1
vector2 = np.array(test2)
print ("А1   "+"А2   "+"  Класс ")
for i in range (0,10):
    result= vector[i]*k1+vector2[i]*k2
    if (result<=0):
        result=0
    else:
        result=1
    print(str(str(vector[i])+"    "+str(vector2[i]))+"   "+str(result))

q1=0
q2=0
err=0
for i in range(0,100):


    for i in range(0, 10):
        metka = vector[i] * q1 + vector2[i] * q2
        if (metka <= 0):
            metka = 0
        else:
            metka = 1

        if(metka-result!=0):
            err=err+1

    q1=q1+1
    q2=q2+1
print (q1,"   ",q2,"  ",err)

