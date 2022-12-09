import numpy as np
from math import exp as e
import random 
from matplotlib import pyplot as plt
def random_weight(n):
    randomweight=[]
    for i in range(n):
        randomweight.append(random.random())
    return np.array(randomweight)
def act_sigmoid(x):
    return 1/(1+e(-x))

def active_out(x,w):
    X=inner_product(w,x)
    return act_sigmoid(X)

def def_sigmoid(x):
    return (2*e(-x))/(1+e(-x))**2
def inner_product(x,y):
    out=0
    for i in range(len(x)):
        out+=x[i]*y[i]
    return out
def update_weight1(w11,z,y,yp,x,eta):
    W11=w11+eta*2*yp*(y-yp)*(1-yp)*z*(1-z)*x/3
    return W11
def update_weight2(w21,y,yp,z,eta):
    W21=w21+eta*2*(y-yp)*yp*(1-yp)*z/3
    return W21
w11=random_weight(3)
w12=random_weight(3)
w21=random_weight(3)
x1=np.array([0,1,0,1])
x2=np.array([0,0,1,1])
y=np.array([0,1,1,0])
eta=0.3
yp=[0,0,0]
MSE=[]
j=0
E=1
while E>0.01:
    E=0
    for i in range(3):
        z1=active_out(np.array([x1[i],x2[i],1]),w11)
        z2=active_out(np.array([x1[i],x2[i],1]),w12)
        Z=np.array([z1,z2,1])
        yp[i]=act_sigmoid(inner_product(Z,w21))
        E=E+0.5*(y[i]-yp[i])**2
        w11=update_weight1(w11,Z,y[i],yp[i],np.array([x1[i],x2[i],1]),eta)
        w12=update_weight1(w12,Z,y[i],yp[i],np.array([x1[i],x2[i],1]),eta)
        w21=update_weight2(w21,y[i],yp[i],Z,eta)
    print(E)
    MSE.append(E)
    j+=1
MSE=np.array(MSE)
print(f'weight2 is {w11} weight12 is {w12} weight21 is {w21}')
testx=np.array([1,1,1])
testh=np.dot(testx,w11)
testh2=np.dot(testx,w12)
testz=np.array([1/(1+e(-testh)),1/(1+e(-testh2)),1])
testyp=1/(1+e(np.dot(testz,w21)))
print('yp is:',yp)
print('test output:',testyp)
plt.plot(MSE)
plt.title('MSE')
plt.show()


