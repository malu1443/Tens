import numpy as np
from sklearn.model_selection import train_test_split
class Data():
    """Read data """
    def ___init__(self,name="sa"):
        self.name=name
    def load_data(self,frac=None):
        self.X1=np.load("X1.npy")
        self.X2=np.load("X2.npy")
        self.y1=np.load("y1.npy")
        self.y2=np.load("y2.npy")
        if frac:
            self.X2,xd,self.y2,yd=train_test_split(self.X2,self.y2,train_size=frac)
        self.m=self.X1.shape[0]
        self.n=self.X1.shape[1]
    def set_batch_size(self,n=50):
        self.batch_size=n
        self.loops=self.m//self.batch_size
        self.i=0
        self.X3=self.X1[10000:12000]
        self.y3=self.y1[10000:12000]
    def next_b(self):
        b0,b1=self.i*self.batch_size,(self.i+1)*self.batch_size
        self.i+=1
        if self.i >= self.loops:
            self.i=0
        return self.X1[b0:b1],self.y1[b0:b1]
if __name__=='__main__':
    print("start")
