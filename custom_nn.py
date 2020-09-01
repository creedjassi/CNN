# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:49:19 2020

@author: jasro
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import pyplot
from os import listdir
import cv2
import seaborn as sns
import tensorflow as tf
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
def data_extraction(path1):
    c=0
    image_set=list()
    for filename in listdir(path1):
        im_data=image.imread(path1+'/' + filename)
        image_set.append(im_data)
        # print('> loaded %s %s' % (filename, im_data.size))
        if c==0:
            imag=cv2.resize(im_data, (64,64), interpolation = cv2.INTER_AREA)
            n2=np.array(imag)
            n1=np.array(n2.reshape(-1))
            n=n1.reshape(n1.shape[0],-1)
            c+=1
            g=n
            continue
        else:
            imag=cv2.resize(im_data, (64,64), interpolation = cv2.INTER_AREA)
            n2=np.array(imag)
            n1=np.array(n2.reshape(-1))
            n=n1.reshape(n1.shape[0],-1)
            g=np.concatenate((g,n),axis=1)
    print(f'Matrix Dim: {g.shape}')  
    print('Example of an image')
    pyplot.imshow(im_data)   
    return(g)
    #np.savetxt("normal.csv", j, delimiter=",")
x_1=data_extraction('D:/Dataset COVID-19 Augmented/COVID-19') 
y_1=np.ones((1,912))
#x_1.shape
#x_2=data_extraction('D:/COVID-19 Radiography Database/NORMAL') 
x_3=data_extraction('D:/Dataset COVID-19 Augmented/Non-COVID-19') 
y_3=np.zeros((1,912))  
Y=np.concatenate((y_1,y_3),axis=1) 
#np.savetxt("Y.csv", Y, delimiter=",")  
X=np.concatenate((x_1,x_3),axis=1)
#np.savetxt("X.csv", X, delimiter=",")  
#np.savetxt("covid.csv", x_1, delimiter=",") 
#my_data = genfromtxt('normal.csv', delimiter=',')
#to access each image column
'''s=x_1[:,1]
s=s.reshape(s.shape[0],1)
s.shape'''
def xavier(n_i,n_i_1):
    np.random.seed(0)
    return(np.random.randn(n_i,n_i_1)*np.sqrt(6/(n_i + n_i_1)))
def initilaize_parameters():
    w1=np.zeros((2,60,4096))
    w1[0]=np.random.randn(60,4096)*np.sqrt(2/4096)
    w1[1]=np.random.randn(60,4096)*np.sqrt(1/4096)
    b1=np.zeros((2,60,1))
   # W1=xavier(60,60)
    W1=np.random.randn(60,60)*np.sqrt(2/60)
    B1=np.zeros((60,1))
    w2=np.zeros((2,30,60))
    w2[0]=np.random.randn(30,60)*np.sqrt(2/60)
    w2[1]=np.random.randn(30,60)*np.sqrt(1/60)
    #w2[0]=xavier(30,60)
    #w2[1]=xavier(30,60)
    b2=np.zeros((2,30,1))
    W2=np.random.randn(30,30)*np.sqrt(2/30)
    #W2=xavier(30,30)
    B2=np.zeros((30,1))
    w3=np.zeros((2,16,30))
    w3[0]=np.random.randn(16,30)*np.sqrt(2/30)
    w3[1]=np.random.randn(16,30)*np.sqrt(1/30)
    #w3[0]=xavier(16,30)
    #w3[1]=xavier(16,30)
    b3=np.zeros((2,16,1))
    W3=np.random.randn(16,16)*np.sqrt(2/16)
    #W3=xavier(16,16)
    B3=np.zeros((16,1))
    w4=np.zeros((2,8,16))
    w4[0]=np.random.randn(8,16)*np.sqrt(2/16)
    w4[1]=np.random.randn(8,16)*np.sqrt(1/16)
    #w4[0]=xavier(8,16)
    #w4[1]=xavier(8,16)
    b4=np.zeros((2,8,1))
    W4=np.random.randn(8,8)*np.sqrt(2/8)
   #W4=xavier(8,8)
    B4=np.zeros((8,1))
    w5=np.zeros((2,1,8))
    w5[0]=np.random.randn(1,8)*np.sqrt(2/8)
    w5[1]=np.random.randn(1,8)*np.sqrt(1/8)
    #w5[0]=xavier(1,8)
    #w5[1]=xavier(1,8)
    b5=np.zeros((2,1,1))
    W5=np.random.randn(1,1)*np.sqrt(2/1)
    #W5=xavier(1,1)
    B5=np.zeros((1,1))
    parameters={'w1':w1,'b1':b1,'W1':W1,'B1':B1,'w2':w2,'b2':b2,'W2':W2,'B2':B2,'w3':w3,'b3':b3,'W3':W3,'B3':B3,
                'w4':w4,'b4':b4,'W4':W4,'B4':B4,'w5':w5,'b5':b5,'W5':W5,'B5':B5}
    return(parameters)
def forward_prop(X,Y,parameters):
    w1=parameters['w1']
    b1=parameters['b1']
    W1=parameters['W1']
    B1=parameters['B1']
    w2=parameters['w2']
    b2=parameters['b2']
    W2=parameters['W2']
    B2=parameters['B2']
    w3=parameters['w3']
    b3=parameters['b3']
    W3=parameters['W3']
    B3=parameters['B3']
    w4=parameters['w4']
    b4=parameters['b4']
    W4=parameters['W4']
    B4=parameters['B4']
    w5=parameters['w5']
    b5=parameters['b5']
    W5=parameters['W5']
    B5=parameters['B5']
    beta=0.25
    alpha_1=0.05
    gamma=0.2
    epsilon=0.0001
    y_p=[]
    dw5=np.zeros((2,8,1))
    dW4=np.zeros((8,8))
    dW5=dB5=0
    dw4=np.zeros((2,8,16))
    dB4=np.zeros((8,1))
    dW3=np.zeros((16,16))
    dw3=np.zeros((2,16,30))
    dB3=np.zeros((16,1))
    db5=np.zeros((2,1,1))
    db4=np.zeros((2,8,1))
    db3=np.zeros((2,16,1))
    dW2=np.zeros((30,30))
    dB2=np.zeros((30,1))
    dw2=np.zeros((2,30,60))
    db2=np.zeros((2,30,1))
    dW1=np.zeros((60,60))
    dB1=np.zeros((60,1))
    dw1=np.zeros((2,60,4096))
    db1=np.zeros((2,60,1))
    m=1600
    dZ5=0
    for i in range(1600):
        X_1=X[:,i]
        X_1=X_1.reshape(X_1.shape[0],1)
        X_1=X_1/255
        z1_1=np.dot(w1[0],X_1) + b1[0]
        a1_1=np.array(tf.nn.leaky_relu(z1_1))
        z1_2=np.dot(w1[1],X_1) + b1[1]
        a1_2=np.array(tf.math.tanh(z1_2))
        Zagg_1=beta*a1_1 + (1-beta)*a1_2
        Z1=np.dot(W1,Zagg_1) + B1
        A1=np.array(tf.nn.relu(Z1))
        z2_1=np.dot(w2[0],A1) + b2[0]
        a2_1=np.array(tf.nn.leaky_relu(z2_1))
        z2_2=np.dot(w2[1],A1) + b2[1]
        a2_2=np.array(tf.math.tanh(z2_2))
        Zagg_2=beta*a2_1 + (1-beta)*a2_2
        Z2=np.dot(W2,Zagg_2)  + B2
        A2=np.array(tf.nn.relu(Z2))
        z3_1=np.dot(w3[0],A2) + b3[0]
        a3_1=np.array(tf.nn.leaky_relu(z3_1))
        z3_2=np.dot(w3[1],A2) + b3[1]
        a3_2=np.array(tf.math.tanh(z3_2))
        Zagg_3=beta*a3_1 + (1-beta)*a3_2
        Z3=np.dot(W3,Zagg_3) + B3
        A3=np.array(tf.nn.relu(Z3))
        z4_1=np.dot(w4[0],A3) + b4[0]
        a4_1=np.array(tf.nn.leaky_relu(z4_1))
        z4_2=np.dot(w4[1],A3) + b4[1]
        a4_2=np.array(tf.math.tanh(z4_2))
        Zagg_4=beta*a4_1 + (1-beta)*a4_2
        Z4=np.dot(W4,Zagg_4) + B4
        A4=np.array(tf.nn.relu(Z4))
        z5_1=np.dot(w5[0],A4) + b5[0]
        a5_1=np.array(tf.nn.leaky_relu(z5_1))
        z5_2=np.dot(w5[1],A4) + b5[1]
        a5_2=np.array(tf.math.tanh(z5_2))
        Zagg_5=beta*a5_1 + (1-beta)*a5_2
        A5=np.array(tf.nn.relu(Zagg_5))
        Z5=np.dot(W5,A5) + B5
        A6=np.array(tf.math.sigmoid(Z5))
        y_p.append(A6)
        dZ5+=(-1)*A6*(1-A6)
        #dZ5+=np.sqrt((Y[i]-A6))*(-1)*A6*(1-A6)
        dW5+=dZ5*A5
        dB5+=dZ5
        dw5[0]+=dZ5*W5*beta*A4
        dw5[1]+=dZ5*W5*(1-beta)*(1-(a5_2)**2)*A4
        db5[0]+=dZ5*W5*beta
        db5[1]+=dZ5*W5*(1-beta)*(1-(a5_2)**2)
        dZ4=dZ5*W5*beta*w5[0]
        dW4+=dZ4*Zagg_4
        dB4+=dZ4.T
        dw4[0]+=(np.dot(dZ4,W4)*beta*A3).T
        dw4[1]+=((np.dot(dZ4,W4).T*(1-beta)*(1-(a4_2)**2)).T*A3).T
        db4[0]=np.dot(dZ4,W4).T*beta
        db4[1]=np.dot(np.dot(dZ4,W4)*(1-beta),(1-(a4_2)**2))
        dZ3=np.dot(np.dot(dZ4,W4),beta*w4[0])
        dW3+=dZ3*Zagg_3
        dB3+=dZ3.T
        dw3[0]+=np.dot(np.dot(dZ3,W3).T,beta*A2.T)
        dw3[1]+=(np.dot(dZ3,W3)*(1-beta)*(1-(a3_2)**2).T*A2).T
        db3[0]+=np.dot(dZ3,W3).T*beta
        db3[1]+=np.dot(dZ3,W3).T*(1-beta)*(1-(a3_2)**2)
        dZ2=np.dot(np.dot(dZ3,W3)*beta,w3[0])
        dW2+=dZ2*Zagg_2
        dB2+=dZ2.T
        dw2[0]+=(np.dot(dZ2,W2)*beta*A1).T
        dw2[1]+=np.dot(np.dot(dZ2,W2).T*(1-beta)*(1-(a2_2)**2),A1.T)
        db2[0]+=np.dot(dZ2,W2).T*beta
        db2[1]+=np.dot(dZ2,W2).T*(1-beta)*(1-(a2_2)**2)
        dZ1=np.dot(np.dot(dZ2,W2)*beta,w2[0])
        dW1+=dZ1*Zagg_1
        dB1+=dZ1.T
        dw1[0]+=(np.dot(dZ1,W1)*beta*X_1).T
        dw1[1]+=(np.dot(dZ1,W1)*beta*(1-(a1_2)**2).T*X_1).T
        db1[0]+=np.dot(dZ1,W1).T*beta
        db1[1]+=np.dot(dZ1,W1).T*(1-beta)*(1-(a1_2)**2)
    #print(len(y_p))    
    #SW5=gamma*dW5 + (1-gamma)*((dW5)**2)
    W5=W5-alpha_1*(dW5/(m)**2)#/np.sqrt(SW5 + epsilon))
    #SB5=gamma*dB5 + (1-gamma)*((dB5)**2)
    B5=B5-alpha_1*(dB5/(m)**2)#/np.sqrt(SB5))
   # Sw5_0=gamma*dw5[0] + (1-gamma)*((dw5[0])**2)
    w5[0]=w5[0]-alpha_1*(dw5[0].T/(m)**2)#/np.sqrt(Sw5_0))
  #  Sw5_1=gamma*dw5[1] + (1-gamma)*((dw5[1])**2)
    w5[1]=w5[1]-alpha_1*(dw5[1].T/(m)**2)
    b5[0]=b5[0]-alpha_1*(db5[0]/(m)**2)
    b5[1]=b5[1]-alpha_1*(db5[1]/(m)**2)
    W4=W4-alpha_1*(dW4/(m)**2)
    B4=B4-alpha_1*(dB4/(m)**2)
    w4[0]=w4[0]-alpha_1*(dw4[0]/(m)**2)
    w4[1]=w4[1]-alpha_1*(dw4[1]/(m)**2) 
    b4[0]=b4[0]-alpha_1*(db4[0]/(m)**2)
    b4[1]=b4[1]-alpha_1*(db4[1]/(m)**2)
    W3=W3-alpha_1*(dW3/(m)**2)
    B3=B3-alpha_1*(dB3/(m)**2)
    w3[0]=w3[0]-alpha_1*(dw3[0]/(m)**2)
    w3[1]=w3[1]-alpha_1*(dw3[1]/(m)**2)
    b3[0]=b3[0]-alpha_1*(db3[0]/(m)**2)
    b3[1]=b3[1]-alpha_1*(db3[1]/(m)**2)
    W2=W2-alpha_1*(dW2/(m)**2)
    B2=B2-alpha_1*(dB2/(m)**2)
    w2[0]=w2[0]-alpha_1*(dw2[0]/(m)**2)
    w2[1]=w2[1]-alpha_1*(dw2[1]/(m)**2)        
    b2[0]=b2[0]-alpha_1*(db2[0]/(m)**2)
    b2[1]=b2[1]-alpha_1*(db2[1]/(m)**2)
    W1=W1-alpha_1*(dW1/(m)**2)
    B1=B1-alpha_1*(dB1/(m)**2)
    w1[0]=w1[0]-alpha_1*(dw1[0]/(m)**2)
    w1[1]=w1[1]-alpha_1*(dw1[1]/(m)**2)        
    b1[0]=b1[0]-alpha_1*(db1[0]/(m)**2)
    b1[1]=b1[1]-alpha_1*(db1[1]/(m)**2)
    #print(dw1)
    #print(dw2)
    #print(dw3)
    #print(dw4)
    #print(dw5)
    activation_values={'w1':w1,'b1':b1,'W1':W1,'B1':B1,'w2':w2,'b2':b2,'W2':W2,'B2':B2,'w3':w3,'b3':b3,'W3':W3,'B3':B3,
                'w4':w4,'b4':b4,'W4':W4,'B4':B4,'w5':w5,'b5':b5,'W5':W5,'B5':B5,'y_p':np.array(y_p)}
    np.savetxt("w1_0.csv", w1[0], delimiter=",") 
    np.savetxt("w1_1.csv", w1[1], delimiter=",") 
    np.savetxt("w2_0.csv", w2[0], delimiter=",") 
    np.savetxt("w2_1.csv", w2[1], delimiter=",") 
    np.savetxt("w3_0.csv", w3[0], delimiter=",") 
    np.savetxt("w3_1.csv", w3[1], delimiter=",") 
    np.savetxt("w4_0.csv", w4[0], delimiter=",") 
    np.savetxt("w4_1.csv", w4[1], delimiter=",")
    np.savetxt("w5_0.csv", w5[0], delimiter=",") 
    np.savetxt("w5_1.csv", w5[1], delimiter=",")
    np.savetxt("b1_0.csv", b1[0], delimiter=",") 
    np.savetxt("b1_1.csv", b1[1], delimiter=",") 
    np.savetxt("b2_0.csv", b2[0], delimiter=",") 
    np.savetxt("b2_1.csv", b2[1], delimiter=",") 
    np.savetxt("b3_0.csv", b3[0], delimiter=",") 
    np.savetxt("b3_1.csv", b3[1], delimiter=",") 
    np.savetxt("b4_0.csv", b4[0], delimiter=",") 
    np.savetxt("b4_1.csv", b4[1], delimiter=",")
    np.savetxt("b5_0.csv", b5[0], delimiter=",") 
    np.savetxt("b5_1.csv", b5[1], delimiter=",")
    np.savetxt("W1.csv", W1, delimiter=",") 
    np.savetxt("W2.csv", W2, delimiter=",") 
    np.savetxt("W3.csv", W3, delimiter=",") 
    np.savetxt("W4.csv", W4, delimiter=",") 
    np.savetxt("W5.csv", W5, delimiter=",") 
    np.savetxt("B1.csv", B1, delimiter=",") 
    np.savetxt("B2.csv", B2, delimiter=",")
    np.savetxt("B3.csv", B3, delimiter=",") 
    np.savetxt("B4.csv", B4, delimiter=",") 
    np.savetxt("B5.csv", B5, delimiter=",") 
    return(activation_values)
import math
def cost_cal(y_pred,y):
    J=0
    for j in range(1600):
        J+=((y[j]-y_pred[j])**2)
        #minus_y_p=1-y_pred[j]
        #if y_pred[j]==0:
         #   J+=(-1)*(1-y[j])*math.log(minus_y_p,10)
        #else:
         #   J+=(-1)*y[j]*math.log(y_pred[j],10)
    #loss=(-1/1642)*np.sum(np.multiply(np.log10(y_pred),y) + np.multiply((1-y),np.log10((1-y_pred))))
    #loss=np.squeeze(loss)
    J/=1600
    J=np.sqrt(J)
    return(J)
def model(X,Y):
    #x_1=data_extraction('D:/Dataset COVID-19 Augmented/COVID-19') 
    #y_1=np.ones((1,912))
    #x_3=data_extraction('D:/Dataset COVID-19 Augmented/Non-COVID-19') 
    #y_3=np.zeros((1,912))  
    #Y=np.concatenate((y_1,y_3),axis=1) 
    #np.savetxt("Y.csv", Y, delimiter=",")  
    #X=np.concatenate((x_1,x_3),axis=1)
    #np.savetxt("X.csv", X, delimiter=",")
    #parameters=initilaize_parameters()
    parameters=initialize_epoch_continuation()
   # gr=[]
    #g=[]
    for epoch in range(1000):
        parameters_trained=forward_prop(X,Y, parameters)
        #parameters=backward_prop(X, Y, parameters, activation, i)
        loss=cost_cal(parameters_trained['y_p'],Y)
        y_hat=parameters_trained['y_p']
       # print(y_hat.shape)
        cost=loss
        print(f'Cost after epoch {epoch} : {cost}')
        #gr.append(cost)
        #g.append(i)
    #plt.plot(g,gr)
    return(y_hat)    
    '''  count+=1
        if count==100:
            print(f'Cost after epoch {epoch + 1} : {cost}')
            count=0
        if epoch==999:
            return(np.array(y_hat))
        if epoch!=0:
            count+=1'''
def evaluation(y_final_pred,Y):
    print(f'Precision, recall, fscore : {precision_recall_fscore_support(Y, y_final_pred)}')
    print(f'Accuracy : {accuracy_score(Y, y_final_pred)}')
def initialize_epoch_continuation():
    w1_0= genfromtxt('w1_0.csv', delimiter=',')  
    w1_1= genfromtxt('w1_1.csv', delimiter=',')
    w2_0= genfromtxt('w2_0.csv', delimiter=',')  
    w2_1= genfromtxt('w2_1.csv', delimiter=',')
    w3_0= genfromtxt('w3_0.csv', delimiter=',')  
    w3_1= genfromtxt('w3_1.csv', delimiter=',')
    w4_0= genfromtxt('w4_0.csv', delimiter=',')  
    w4_1= genfromtxt('w4_1.csv', delimiter=',')
    w5_0= genfromtxt('w5_0.csv', delimiter=',')  
    w5_1= genfromtxt('w5_1.csv', delimiter=',')
    b1_0= genfromtxt('b1_0.csv', delimiter=',')  
    b1_1= genfromtxt('b1_1.csv', delimiter=',')
    b2_0= genfromtxt('b2_0.csv', delimiter=',')  
    b2_1= genfromtxt('b2_1.csv', delimiter=',')
    b3_0= genfromtxt('b3_0.csv', delimiter=',')  
    b3_1= genfromtxt('b3_1.csv', delimiter=',')
    b4_0= genfromtxt('b4_0.csv', delimiter=',')  
    b4_1= genfromtxt('b4_1.csv', delimiter=',')
    b5_0= genfromtxt('b5_0.csv', delimiter=',')  
    b5_1= genfromtxt('b5_1.csv', delimiter=',')
    W1= genfromtxt('W1.csv', delimiter=',')  
    W2= genfromtxt('W2.csv', delimiter=',')
    W3= genfromtxt('W3.csv', delimiter=',')  
    W4= genfromtxt('W4.csv', delimiter=',')
    W5= genfromtxt('W5.csv', delimiter=',')
    B1= genfromtxt('B1.csv', delimiter=',') 
    B1=B1.reshape(60,1)
    B2= genfromtxt('B2.csv', delimiter=',')
    B2=B2.reshape(30,1)
    B3= genfromtxt('B3.csv', delimiter=',')  
    B3=B3.reshape(16,1)
    B4= genfromtxt('B4.csv', delimiter=',')
    B4=B4.reshape(8,1)
    B5= genfromtxt('B5.csv', delimiter=',')  
    B5=B5.reshape(1,1)
    w1=np.zeros((2,60,4096))
    w1[0]=w1_0
    w1[1]=w1_1
    b1=np.zeros((2,60,1))
    b1[0]=b1_0.reshape(60,1)
    b1[1]=b1_1.reshape(60,1)
    w2=np.zeros((2,30,60))
    w2[0]=w2_0
    w2[1]=w2_1
    b2=np.zeros((2,30,1))
    b2[0]=b2_0.reshape(30,1)
    b2[1]=b2_1.reshape(30,1)
    w3=np.zeros((2,16,30))
    w3[0]=w3_0
    w3[1]=w3_1
    b3=np.zeros((2,16,1))
    b3[0]=b3_0.reshape(16,1)
    b3[1]=b3_1.reshape(16,1)
    w4=np.zeros((2,8,16))
    w4[0]=w4_0
    w4[1]=w4_1
    b4=np.zeros((2,8,1))
    b4[0]=b4_0.reshape(8,1)
    b4[1]=b4_1.reshape(8,1)
    w5=np.zeros((2,1,8))
    w5[0]=w5_0
    w5[1]=w5_1
    b5=np.zeros((2,1,1))
    b5[0]=b5_0.reshape(1,1)
    b5[1]=b5_1.reshape(1,1)
    updated_parameters={'w1':w1,'b1':b1,'W1':W1,'B1':B1,'w2':w2,'b2':b2,'W2':W2,'B2':B2,'w3':w3,'b3':b3,'W3':W3,'B3':B3,
                'w4':w4,'b4':b4,'W4':W4,'B4':B4,'w5':w5,'b5':b5,'W5':W5,'B5':B5}
    return(updated_parameters)
#X[0,:].shape    
Z=np.concatenate((X,Y),axis=0)
for i in range(1824):
    a=np.random.randint(0,1823)
    b=np.random.randint(0,1823)
    if a!=b:
        temp=Z[:,a]
        Z[:,a]=Z[:,b]
        Z[:,b]=temp
Z=Z        
Y_=Z[4096,:]
X_=Z[0:4096,:]
X_=X_
a_train=X_[0:4096,0:1600]
b_train=Y_[0:1600]
a_test=X_[0:4096,1600:1824]
b_test=Y_[1600:1824]
a_train.shape
b_train.shape
a_test.shape
b_test.shape
train_y=model(a_train,b_train)
train_y.shape
t=np.array(train_y.reshape(-1))
'''t.shape
np.savetxt("train_pre.csv", t, delimiter=",") 
my_data = genfromtxt('train_pre.csv', delimiter=',')
import pandas as pd
d2=pd.read_csv('C:/Users/jasro/train_pre.csv',low_memory=False)
d2['pred'].unique()
my_data
fi=[]
len(my_data)
for i in range(len(my_data)):
    if my_data[i]<0.519:
        fi.append(0)
    else:
        fi.append(1)
len(fi)        
np.savetxt("fff.csv", fi, delimiter=",")
import pandas as pd
d=pd.read_csv('C:/Users/jasro/fff.csv',low_memory=False)
d['pred'].value_counts()
np.savetxt("b_.csv", b_train, delimiter=",") 
d1=pd.read_csv('C:/Users/jasro/b_.csv',low_memory=False)
d1['pred'].value_counts()
from sklearn.metrics import confusion_matrix
confusion_matrix(d1['pred'].tolist(), d['pred'].tolist())
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(d1['pred'].tolist(), d['pred'].tolist(), average='macro')
from sklearn.metrics import accuracy_score
accuracy_score(d1['pred'].tolist(), d['pred'].tolist())
len(d)'''
