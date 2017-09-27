import pandas as pd
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
class Data():
    """Read data """
    def ___init__(self,name="sa"):
        self.name=name
    def split_train(self,proc=0.3):
        self.X11,self.X12,self.y11,self.y12=train_test_split(self.X1,self.y1,test_size=proc,random_state=42)
    def pd2Xy(self,B,y_col,proc=0.25,split=None):
        """ Converst pandas Dataframe into train and test arrays. 
        y_col specifice the column that will serv as the target values"""
        y=B[y_col].values
        B.drop([y_col],axis=1,inplace=True)
        self.head=np.array(B.columns)
        if split:
            X1,self.X3,y1,self.y3=train_test_split(B.values,y,test_size=split)
            self.X1,self.X2,self.y1,self.y2=train_test_split(X1,y1,test_size=proc)
        else:
            self.X3=None
            self.X1,self.X2,self.y1,self.y2=train_test_split(B.values,y)
    def minmax(self):
        """Used to norm X_train and X_test"""
        from sklearn.preprocessing import MinMaxScaler
        X=np.append(self.X1,self.X2,axis=0)
        mm=MinMaxScaler().fit(X)
        self.X1=mm.transform(self.X1)
        self.X2=mm.transform(self.X2)
    def y_classify(self,val=-1,inplace=False):
        """ Transform regression y array into boolean classify array. val specifices value for catoff. 
        inplace updates the array"""
        y1=np.copy(self.y1)
        y2=np.copy(self.y2)
        if val==-1:
            val=y1.mean()
        print("y1/y2 min/mean/max:",y1.min(),y1.mean(),y1.max(),y2.min(),y2.mean(),y2.max())
        y1[y1<val]=0
        y1[y1>=val]=1
        y2[y2<val]=0
        y2[y2>=val]=1
        print("y1,y2 mean:",y1.mean(),y2.mean()) 
        if inplace:
            print("dsad")
            self.y1=y1
            self.y2=y2
    def fselect_unstat(self,prec=20):
        """ use p value to exclude variables. prec is precentige of features selceted"""
        from sklearn.feature_selection import SelectPercentile
        select=SelectPercentile(percentile=prec)
        select.fit(self.X1,self.y1)
        self.X1=select.transform(self.X1)
        self.X2=select.transform(self.X2)
        if self.X3:
            self.X3=select.transform(self.X3)
        print("Selected feat using unsata model:",self.head[select.get_support()],len(self.head[select.get_support()]))
        self.head=self.head[select.get_support()]
    def fselect_mb(self,nes=50,C=True):
        """ use random forest to exclude variables. nes is number of estimators. C true for classification problem."""
        from sklearn.feature_selection import SelectFromModel
        if C:
            print("RandomF Classifier selected")
            sel=SelectFromModel(RandomForestClassifier(n_estimators=nes,random_state=42),threshold='median')
        else:
            print("RandomF Regressor selected")
            sel=SelectFromModel(RandomForestRegressor(n_estimators=nes,random_state=42),threshold='median')
        sel.fit(self.X1,self.y1)
        self.X1=sel.transform(self.X1)
        self.X2=sel.transform(self.X2)
        if self.X3:
            self.X3=sel.transform(self.X3)
        print("Selected feat using model based:",self.head[sel.get_support()],len(self.head[sel.get_support()]))
        self.head=self.head[sel.get_support()]
    def fselect_IF(self,nes=50,fs=40,Classification=True):
        """ use random forest to exclude variables. nes is number of estimators and fs number of features selected"""
        from sklearn.feature_selection import RFE
        if Classification:
            print("RandomF Classifier selected")
            sel=RFE(RandomForestClassifier(n_estimators=nes,n_jobs=6,random_state=42),n_features_to_select=fs)
        else:
            print("RandomF Regressor selected")
            sel=RFE(RandomForestRegressor(n_estimators=nes,n_jobs=6,random_state=42),n_features_to_select=fs)
        sel.fit(self.X1,self.y1)
        self.X1=sel.transform(self.X1)
        self.X2=sel.transform(self.X2)
        if self.X3:
            self.X3=select.transform(self.X3)
        print("Selected feat using IF:",self.head[sel.get_support()],len(self.head[sel.get_support()]))
        self.head=self.head[sel.get_support()]
    def fselect_pca(self,proc):
        """ use principal compnent analysis to reduce number of features"""
        from sklearn.decomposition import PCA
        pca=PCA(n_components=proc).fit(self.X1)
        self.X1=pca.transform(self.X1)
        self.X2=pca.transform(self.X2)
        if self.X3:
            self.X3=pca.transform(self.X3)
    def plot_reg(self,D,Z):
        plt.plot(D,Z[:,0],label='Train')
        plt.plot(D,Z[:,1],label='Test')
        plt.show()
        plt.close()
    def plot_cl(self,C,Z,g):
        """Plot score for train and test samples as a function of C and gamma. see plotFeat for more inf."""
        for j in range(Z.shape[1]):
            plt.plot(C,Z[:,j,0],label=str(g[j]))
            plt.plot(C,Z[:,j,1],label=str(g[j]))
        plt.legend()
        plt.show()
        plt.close()
    def gridS(self,mod,para):
        """Apply grid search for diffrent functions"""
        grid=GridSearchCV(mod,param_grid=para,n_jobs=6).fit(self.X1,self.y1)
        print(grid.estimator.__class__.__name__,grid.score(self.X1,self.y1),grid.score(self.X2,self.y2),grid.best_params_)
        self.grid=grid
if __name__=="__main__":
    print("dsw")
