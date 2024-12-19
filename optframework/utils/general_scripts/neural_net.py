# -*- coding: utf-8 -*-
"""
Created on Mon Sep 1 09:01:15 2021

@author: xy0264
"""

import numpy as np
import scipy.special
import copy
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from extract_xls import extract_xls

def plot_grid(data):
    
    from general_scripts.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, c_KIT_orange, c_KIT_purple
    
    # Get number of individual NM components and also extrat their names
    num_comp=len(set(data['NM1']))
    name_comp=list(set(data['NM1']))
    min_ph=min(data['pH [-]'])-1
    max_ph=max(data['pH [-]'])+1
    min_I=min(data['I [mol/L]'])*0.5
    max_I=max(data['I [mol/L]'])*2
    
    # Set up plot
    scl=1.2
    plt.rc('mathtext', fontset='cm')
    #plt.rc('font', family='Latin Modern Roman')
    plt.rc('xtick', labelsize=10*scl)
    plt.rc('ytick', labelsize=10*scl)
    plt.rc('axes', labelsize=12*scl, linewidth=0.5*scl)
    plt.rc('legend', fontsize=10*scl, handlelength=3*scl)
    markerlist=['o','*','s','^','x']
    colorlist=[c_KIT_green,c_KIT_red,c_KIT_blue,c_KIT_orange,c_KIT_purple]
    
    # Close all and setup figure
    plt.close('all')
    fig, ax = plt.subplots(1,num_comp,figsize=np.array([6.4, 3.2])*2) 
    
    for i in range(num_comp):
        
        # Extrat pH and I values for component and plot them with the name as legend entry
        idx=[j for j, x in enumerate(data['NM1']) if x == name_comp[i]]
        ph=[data['pH [-]'][j] for j in idx]
        I=[data['I [mol/L]'][j] for j in idx]
        ax[i].scatter(I,ph,label=name_comp[i],color=colorlist[i],marker=markerlist[i]) 
        
        ax[i].set_xlim(left=min_I,right=max_I)
        ax[i].set_ylim(bottom=min_ph,top=max_ph)
        ax[i].set_ylabel('$pH$ / $-$')
        ax[i].set_xlabel('Ionic strength $I$ / $mol/L$')
        ax[i].legend()
        ax[i].grid(True)

def plot_regression(x,y,network,setName='',fig=None,idx=1):
    
    from general_scripts.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, c_KIT_orange, c_KIT_purple
    
    idx_plt=1
    if len(y.shape)>1:
        yp=network.predict(x)[:,idx_plt]
        y=y[:,idx_plt]
    else:
        yp=network.predict(x)
        
    A=np.ones((x.shape[0],2))
    A[:,0]=yp.reshape(x.shape[0])
    m,c=np.linalg.lstsq(A,y,rcond=None)[0]
    
    if fig==None:
        fig=plt.figure(figsize=np.array([6.4, 6.4])*2)
    ax=fig.add_subplot(2,2,idx)
    myTitle='%s: %f *x + %f' %(setName,m,c)
    ax.set_title(myTitle)
    ax.scatter(yp,y,marker='+',color=c_KIT_green)
    ax.set_xlabel('ANN Output')
    ax.set_ylabel('Data Set')
    alpha=min(yp.min(),y.min())
    omega=max(yp.max(),y.max())
    xPlot=np.linspace(alpha,omega,10)
    ax.plot(xPlot,xPlot,'k')
    ax.plot(xPlot,xPlot*m+c,'--',color=c_KIT_red)
    ax.set_xlim([alpha,omega])
    ax.set_ylim([alpha,omega])
    ax.grid(True)
    
    return(fig)
    
class myMLP:
    def __init__(self,hiddenlayer=(10,10),classification=False):
        self.hl=hiddenlayer
        self.classification=classification
        self.xMin=0.0
        self.xMax=1.0
        self.W=[]
        self._sigmoid=lambda x:scipy.special.expit(x)
        
    def _initWeights(self):
        self.W.append((np.random.rand(self.hl[0],self.il)-0.5))
        self.W.append((np.random.rand(self.hl[1],self.hl[0])-0.5))
        self.W.append((np.random.rand(self.ol,self.hl[1])-0.5))
        
    def _calOut(self,X):
        O1=self._sigmoid(self.W[0]@X.T)
        O2=self._sigmoid(self.W[1]@O1)
        y=(self.W[len(self.W)-1]@O2).T
        return(y)
    
    def predict(self,X):
        X=(X-self.xMin)/(self.xMax-self.xMin)
        X=np.hstack((X,np.ones(X.shape[0])[:,None]))
        y=self._calOut(X)
        if self.classification: y=np.round(y)
        return(y)
    
    def fit(self,X,Y,eta=0.75,maxIter=200,vareps=1e-3,scale=True,XT=None,YT=None):
        self.xMin=X.min(axis=0) if scale else 0
        self.xMax=X.max(axis=0) if scale else 1        
        X=(X-self.xMin)/(self.xMax-self.xMin)
        X=np.hstack((X,np.ones(X.shape[0])[:,None]))
        if len(Y.shape)==1:
            Y=Y[:,None]
        self.il=X.shape[1]
        self.ol=Y.shape[1]
        self._initWeights()
        (XVal, YVal, X, Y) = self._divValTrainSet(X,Y)
        
        self.train(X,Y,XVal,YVal,eta,maxIter,vareps,XT,YT)
        
    def train(self,X,Y,XVal=None,YVal=None,eta=0.75,maxIter=200,vareps=10**-3,XT=None,YT=None): 
        if XVal is None: (XVal, YVal, X, Y) = self._divValTrainSet(X,Y)
        if len(Y.shape)==1: Y=Y[:,None]
        if len(YVal.shape)==1: YVal=YVal[:,None]
        if self.il!=X.shape[1]: X=np.hstack((X,np.ones(X.shape[0])[:,None]))
        if self.il!=XVal.shape[1]: XVal=np.hstack((XVal,np.ones(XVal.shape[0])[:,None]))
        
        dW=[]
        for i in range(len(self.W)):
            dW.append(np.zeros_like(self.W[i]))
        yp=self._calOut(XVal)        
        if self.classification: yp = np.round(yp)
        meanE=(np.sum((YVal-yp)**2)/XVal.shape[0])/YVal.shape[1]
        minError=meanE
        minW=copy.deepcopy(self.W)
        self.errorVal=[]; self.errorTrain=[]; self.errorTest=[]
        mixSet=np.random.choice(X.shape[0],X.shape[0],replace=False)
        
        counter=0
        while meanE>vareps and counter<maxIter:
            counter+=1
            for m in range(self.ol):
                for i in mixSet:
                    x=X[i,:]
                    O1=self._sigmoid(self.W[0]@x.T)
                    O2=self._sigmoid(self.W[1]@O1)
                    temp=self.W[2]*O2*(1-O2)[None,:]
                    dW[2]=O2
                    dW[1]=temp.T@O1[:,None].T
                    dW[0]=(O1*(1-O1)*(temp@self.W[1])).T@x[:,None].T
                    yp=self._calOut(x)[m]
                    yfactor=np.sum(Y[i,m]-yp)
                    for j in range(len(self.W)):
                        self.W[j]+=eta*yfactor*dW[j]
                    
            yp=self._calOut(X)

            yp = self._calOut(XVal)
            if self.classification: yp = np.round(yp)
            meanE = (np.sum((YVal-yp)**2)/XVal.shape[0])/YVal.shape[1] #*\label{code:fullmlpbatch:6}
            self.errorVal.append(meanE)
            if meanE < minError: 
                minError = meanE
                minW = copy.deepcopy(self.W)      
                self.valChoise = counter
            
            if XT is not None:
                yp = self.predict(XT)
                if len(YT.shape) == 1: YT = YT[:,None]; 
                meanETest = (np.sum((YT-yp)**2)/XT.shape[0])/YT.shape[1]
                self.errorTest.append(meanETest)
                
                yp = self._calOut(X)
                if self.classification:
                    yp = np.round(yp)
                meanETrain = (np.sum((Y-yp)**2)/X.shape[0])/Y.shape[1]
                self.errorTrain.append(meanETrain)
                
        self.W=copy.deepcopy(minW)
    
    def _divValTrainSet(self, X,Y):
        self.ValSet    = np.random.choice(X.shape[0],int(X.shape[0]*0.25),replace=False)
        self.TrainSet  = np.delete(np.arange(0, Y.shape[0] ), self.ValSet) 
        XVal     = X[self.ValSet,:]
        YVal     = Y[self.ValSet]
        X        = X[self.TrainSet,:]
        Y        = Y[self.TrainSet]
        return (XVal, YVal, X, Y)
    
    def exportNet(self, filePrefix):
        np.savetxt(filePrefix+"MinMax.csv", np.array([self.xMin, self.xMax]), delimiter=",")
        np.savetxt(filePrefix+"W0.csv", self.W[0], delimiter=",")
        np.savetxt(filePrefix+"W1.csv", self.W[1], delimiter=",")
        np.savetxt(filePrefix+"W2.csv", self.W[2], delimiter=",")
    
    def importNet(self,filePrefix, classification=False):
        MinMax = np.loadtxt(filePrefix+'MinMax.csv',delimiter=",")
        W2 = np.loadtxt(filePrefix+'W2.csv',delimiter=",")
        W1 = np.loadtxt(filePrefix+'W1.csv',delimiter=",")    
        W0 = np.loadtxt(filePrefix+'W0.csv',delimiter=",") 
        self.W = [W0,W1,W2]
        #self.hl = (W0.shape[0], W2.shape[1])
        self.il = W0.shape[1]
        self.ol = W2.shape[0]
        self.xMin = MinMax[0] 
        self.xMax = MinMax[1]
        self.classification = classification

def mlp(hidden_layer_sizes, alpha):
    params_gbm = {}
    params_gbm['hidden_layer_sizes'] = round(hidden_layer_sizes)
    params_gbm['alpha'] = alpha
    params_gbm['activation'] = def_activation
    params_gbm['solver'] = def_solver
    # params_gbm['learning_rate_init'] = learning_rate_init
    scores = cross_val_score(MLPRegressor(max_iter = 2e+10, random_state=rs, **params_gbm,early_stopping=def_early_stopping,tol=def_tol),#, early_stopping = True, tol=1e-3
                             X_train, y_train, scoring='neg_root_mean_squared_error', cv=5).mean()
    
    score = scores.mean()
    
    return score
            
# %% MAIN    
if __name__ == '__main__':
    
    #np.random.seed(42)
        
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV    
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MaxAbsScaler
    from bayes_opt import BayesianOptimization
    from sklearn import svm
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn import datasets

    plt.close('all')
    
    # Define optimization and fallback ann model
    opt='grid' #'bay':Bayesian, 'grid': Gridsearch, else: simpleMLP
    standard_ann='mlp' #'mlp': Multilayer Perceptron, 'svm': Support vector machine
    
    # Define if outputs (alphas are beeing transformed to log-scale)
    transform_Y=True
    # Define a threshold for alpha. If Y<thr_Y --> Y=thr_Y
    thr_Y=1e-3
    
    # Fraction between Train and Test Dataset
    frac=0.8
    # Standard MLP parameters
    def_activation='logistic'
    def_solver='lbfgs'
    def_tol=1e-2
    def_early_stopping=False    
    # Should initial weights be random or fixed?
    rs=42 # Fixed: 42, random: None
    # Define hiddenlayer as tuple
    hl=(7)
    # Number of iterations
    fit_iter=1
    # Export ANN
    exp=True
    
    
    # XTrain=np.random.rand(2500,2)
    # YTrain=np.sin(2*np.pi*(XTrain[:,0]+0.5*XTrain[:,1]))+0.5*XTrain[:,1]
    # Noise=np.random.rand(YTrain.shape[0])-0.5
    # YTrain=(1+0.05*Noise)*YTrain
    
    # XTest=np.random.rand(500,2)
    # YTest=np.sin(2*np.pi*(XTest[:,0]+0.5*XTest[:,1]))+0.5*XTest[:,1]
    
    # myPredict=myMLP(hiddenlayer=(8,8))
    # myPredict.fit(XTrain,YTrain)
    # yp=np.squeeze(myPredict.predict(XTest))
    
    #filename_ANN_data=os.path.join(os.path.dirname( __file__ ),'..',"data\\neural_net_data\\ANN_training_data_OZ_211103.xlsx")
    #filename_ANN_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\neural_net_data\\ANN_OZ_211103.npy")
    filename_ANN_data=os.path.join(os.path.dirname( __file__ ),'..',"data\\neural_net_data\\ANN_training_data_ESB_211103.xlsx")
    filename_ANN_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\neural_net_data\\ANN_ESB_211103.npy")
    data=extract_xls(filename_ANN_data)
        
    #plot_grid(data)
    
    # Restructure ANN data
    ld=len(data['NM1'])
    name_comp=list(set(data['NM1']))
    X=np.ones((ld,3))    
    Y=np.zeros((ld,3))    
    
    for i in range(ld):        
        if data['NM1'][i]=='SF800': X[i,0]=0
        if data['NM1'][i]=='ZNO': X[i,0]=1 
        if data['NM1'][i]=='SF300': X[i,0]=2
        X[i,1]=data['pH [-]'][i]    
        X[i,2]=np.log10(data['I [mol/L]'][i])   
        Y[i,0]=data['ALPHA_N_NM1NM1'][i]
        Y[i,1]=data['ALPHA_N_NM1M'][i]
        Y[i,2]=data['ALPHA_N_MM'][i]   
    
    # Apply threshold
    Y[Y<thr_Y]=thr_Y
        
    # Transform Y on log axis
    if transform_Y: Y=np.log10(Y)
    
    # Scale/Standardize Data
    scaler=MaxAbsScaler().fit(X)
    X=scaler.transform(X)
    #Y=MaxAbsScaler().fit(Y).transform(Y)
    
    # Only use column i
    #Y=Y[:,1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1-frac, random_state=rs)
    
    # %% GRID SEARCH
    if opt=='grid': 
        param_grid={'hidden_layer_sizes':[]}
        num_hl=[1,2]
        #hl_size=[1,5,10,50,100,500]
        hl_size=np.arange(1,10,1)
        for i in range(len(num_hl)):
            if num_hl[i]>1:
                for j in range(len(hl_size)):
                    for k in range(len(hl_size)):
                        param_grid['hidden_layer_sizes'].append((hl_size[j],hl_size[k]))
            else:
                for j in range(len(hl_size)):
                    param_grid['hidden_layer_sizes'].append((hl_size[j]))
        
        # Grid Search
        mlp=MLPRegressor(random_state=rs,max_iter=1000,activation=def_activation, solver=def_solver,early_stopping=def_early_stopping,tol=def_tol)
        reg=GridSearchCV(mlp,param_grid,scoring='neg_root_mean_squared_error',verbose=2)
        reg.fit(X_train, y_train)
        
        print(reg.best_params_)
    
    # %% BAYESIAN OPTIMIZATION
    elif opt=='bay':
        params_gbm ={
            'hidden_layer_sizes':(3, 50), 
            'alpha':(0.01, 0.1)}
        gbm_bo = BayesianOptimization(mlp, params_gbm, random_state=rs)
        gbm_bo.maximize(init_points=100, n_iter=150)
        
        params_gbm = gbm_bo.max['params']    
        
        params_gbm['hidden_layer_sizes'] = round(params_gbm['hidden_layer_sizes'])
        params_gbm['alpha'] = params_gbm['alpha']
    
        print(params_gbm)
        reg = MLPRegressor(max_iter = 2e+10, random_state=rs, **params_gbm, activation=def_activation, solver=def_solver,early_stopping=def_early_stopping,tol=def_tol) 
        reg.fit(X_train, y_train)
    
    # %% SimpleMLP
    else:
        if standard_ann == 'mlp':
            reg = MLPRegressor(hl,max_iter = 1000, random_state=rs, activation=def_activation, solver=def_solver,early_stopping=def_early_stopping,tol=def_tol) 
            reg.fit(X_train, y_train)
        elif standard_ann == 'svm':
            svr = svm.SVR()
            reg = MultiOutputRegressor(svr)
            reg.fit(X_train, y_train)
        
    # %% GENERAL POST-PROCESSING / PLOTTING
    fig=plot_regression(X_train,y_train,reg,'Train Data')
    if frac!=1: fig=plot_regression(X_test,y_test,reg,'Test Data',fig,2)
    fig=plot_regression(X,Y,reg,'All Data',fig,3)

    print(reg.score(X_test, y_test))    
    
    if exp:
        np.save(filename_ANN_exp,{'ANN':reg,'scaler':scaler,'transform_Y':transform_Y})
    
    # %% SELF WRITTEN MODEL
    
    # # Only use column i
    # Y=Y[:,0]
    
    # # Split dataset up
    # TrainSet=np.random.choice(X.shape[0],int(X.shape[0]*frac),replace=False)
    # XTrain=X[TrainSet,:]
    # YTrain=Y[TrainSet]
    
    # if frac!=1:
    #     TestSet=np.delete(np.arange(0,X.shape[0]),TrainSet)
    #     XTest=X[TestSet,:]
    #     YTest=Y[TestSet]    
    
    # RMSE_min=1000
    # for i in range(fit_iter):
    #     myPredict=myMLP(hiddenlayer=hl)
    #     myPredict.fit(XTrain,YTrain,eta=0.75,maxIter=500)
        
    #     yp_All=np.squeeze(myPredict.predict(X))
    #     RMSE_All=np.sqrt(np.sum((yp_All-Y)**2)/len(yp_All))
    #     if RMSE_All<RMSE_min:
    #         RMSE_min=RMSE_All
    #         ANN_min=myPredict
        
    #     print(f'Iteration {i+1}/{fit_iter}.. RMSE_min={RMSE_min}')
    
    # myPredict=ANN_min
    
    # if exp: myPredict.exportNet(filename_ANN_exp)
    
    # yp_Train=np.squeeze(myPredict.predict(XTrain))
    # if frac!=1: yp_Test=np.squeeze(myPredict.predict(XTest))
    # yp_All=np.squeeze(myPredict.predict(X))
    
    # RMSE_Train=np.sqrt(np.sum((yp_Train-YTrain)**2)/len(yp_Train))
    # if frac!=1: RMSE_Test=np.sqrt(np.sum((yp_Test-YTest)**2)/len(yp_Test))
    # RMSE_All=np.sqrt(np.sum((yp_All-Y)**2)/len(yp_All))
        
    # fig=plot_regression(XTrain,YTrain,myPredict,'Train Data')
    # if frac!=1: fig=plot_regression(XTest,YTest,myPredict,'Test Data',fig,2)
    # fig=plot_regression(X,Y,myPredict,'All Data',fig,3)
    
    # if frac!=1: 
    #     print('RMSE Values: Train:',RMSE_Train,'| Test:',RMSE_Test,'| All:',RMSE_All)
    # else:        
    #     print('RMSE Values: Train:',RMSE_Train,'| All:',RMSE_All)
    
    # # Normalize X
    # X[:,:-1]=(X[:,:-1]-X[:,:-1].min(axis=0))/(X[:,:-1].max(axis=0)-X[:,:-1].min(axis=0)) 
    # # Normalize Y
    # #Y=(Y-Y.min(axis=0))/(Y.max(axis=0)-Y.min(axis=0)) 
    
    # # X, y = make_regression(n_samples=200, random_state=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y[:,1])
    # regr = MLPRegressor((5,5), max_iter=5000).fit(X_train, y_train)
    # #regr = svm.SVR().fit(X_train, y_train)
    # y_mod_test=regr.predict(X_test)
    # y_mod_ges=regr.predict(X)
    # print(regr.score(X_test,y_test))
    # # comp=np.zeros((len(y_test),2))
    # # comp[:,0]=y_mod_test[:,1]
    # # comp[:,1]=y_test[:,1]
    # # print(comp)
    
    # # comp2=np.zeros((len(Y),2))
    # # comp2[:,0]=y_mod_ges[:,1]
    # # comp2[:,1]=Y[:,1]
    # # print(comp2)
    
