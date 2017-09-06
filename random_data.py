# -*- coding: utf-8 -*-
"""
Random_data.py module produces random distributed data for testing ML code. 
Can produce both classification and regression data."""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
class Reg():
    """
    Produce random dist. regresions data of diffrent shape. """

    def __init__(self,name="df"):
        self.name=name
    def gen_x(self,d,n,F=False):
        """
        Generates x data. if F is true the data will be random distibuted in x dir."""
        if F:
            self.x=np.sort(np.random.rand(n)*d).reshape(-1,1)
        else:
            self.x=np.linspace(0.1,d,n).reshape(-1,1)
        self.xr=np.c_(np.ones(self.x.shape),self.x)
    def gen_sqrdata(self,n=100,d=1,a=2,b=2,m=2,F=False):
        """
        Create sqr distributed data. n number of data points, d is the upper range. a,b,m are the 
        the coeff for eq y=ax**2+bx+m. If F ist true x will be random distributed in x dir"""
        self.gen_x(d,n,F)
        self.y1=a*self.x**2+b*self.x+m
        self.y=np.random.normal(self.y1)
    def gen_linedata(self,n=100,d=1,k=2,m=2,F=False):
        """
        Create linear distributed data. n number of data points, d is the upper range, k the slope and m the inters. 
        If F ist true x will be random distributed in x dir"""
        self.gen_x(d,n,F)
        self.y1=k*self.x+m
        self.y=np.random.normal(self.y1)
    def gen_sindata(self,n=100,d=8,s=0.3,F=False):
        """
        create test data with sinus shape.If  F ist true x will be random distributed in x dir"""
        self.gen_x(d,n,F)
        self.x=np.sort(np.random.rand(n)*d).reshape(-1,1)
        self.y=(1+self.x)*np.sin(self.x)+np.random.normal(0,scale=self.x*s)
        self.y1=(1+self.x)*np.sin(self.x)
    def gen_betadata(self,d=1,n=100,dis=0.05,F=False):
        """
        Create cum-beta distributed data for fit function. If F ist true x will be random distributed in x dir"""
        self.gen_x(d,n,F)
        b=beta(4,3,scale=d)
        self.y=np.zeros(n)
        self.y1=b.cdf(self.x)
        for i in range(self.y.shape[0]):
            self.y[i]=np.random.normal(self.y1[i],scale=dis+dis*self.y1[i])
        self.y[np.where(self.y<0)]=0
        self.y[np.where(self.y>1)]=1
    def plot(self,x=None,y=None):
        """
        plot random distributed regesion data"""

        plt.plot(self.x,self.y1)
        plt.plot(self.x,self.y,'.k')
        if y is not None and x is not None:
            plt.plot(x,y,linewidth=2)
        elif y is not None:
            plt.plot(self.x,y,linewidth=2)
        plt.show()
        plt.close()
class Classify():
    """
    Produce random dist. classification data of diffrent shape. """

    def __init__(self,name="df"):
        self.name=name
    def gen_traindata(self,n=20,k=2,sig=3,F=False):
        """
        Generate random training data for classification problems. n is 
        number of points per class and k is number of classes. sig is the standrad dev 
        Two fixed center points if F is True"""
        from itertools import combinations
        d0=np.inf
        if F:
            print ("fixed center")
            s=np.array([[5,3],[5,7]])
        else:
            s=np.random.rand(k,2)*10
        for i in combinations(range(k),2):
            d=np.linalg.norm(s[i[0]]-s[i[1]])
            if d<d0:
                d0=d
        self.sig=d0/(sig*2)
        self.s=s
        print(s)
        self.k=k
        self.X_train,self.y_train=self.rand_gen(n)
    def gen_testdata(self,n=20):
        """
        Generate random test data for classification problems. n is 
        number of points per class."""
        self.X_test,self.y_test=self.rand_gen(n)
    def rand_gen(self,n):
        x=np.zeros((n*self.k,2))
        y=np.zeros(n*self.k)
        for i in range(self.k):
            x[n*i:n*(i+1)]=np.random.normal(x[n*i:n*(i+1)],scale=self.sig)+self.s[i]
            y[n*i:n*(i+1)]=i
        return x,y
    def plot(self,typ='brain'):
        """
        plots the randaom distributed class data.
        typ can either be 'train','test' or 'both' which plot train test or both data"""
        if typ=='train' or typ=='both':
            for i in np.unique(self.y_train):
                    plt.plot(self.X_train[self.y_train==i,0],self.X_train[self.y_train==i,1],'.')
        if typ=='test' or typ=='both':
            for i in np.unique(self.y_test):
                    plt.plot(self.X_test[self.y_test==i,0],self.X_test[self.y_test==i,1],'.')
        plt.show()
        plt.close()
if __name__=='__main__':
    C=Classify()
    C.gen_traindata(n=300,sig=2,)
    C.gen_testdata(n=40)
    C.plot(typ='both')
