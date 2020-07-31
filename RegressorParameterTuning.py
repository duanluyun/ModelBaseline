import RegressorConfig as Configure
import DataPreprocessing
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from hyperopt import fmin,tpe,STATUS_OK,Trials
from sklearn.preprocessing import scale,normalize

bestTemp=None
dataTemp=DataPreprocessing.load_data()
featuresTemp=dataTemp.data
label=dataTemp.target

def Regressor_hyperopt_train_test(params):
    global featuresTemp
    global label
    if 'scale' in params:
        if params['scale']==1:
            featuresTemp=scale(featuresTemp)
        del params['scale']
    if 'normalize' in params:
        if params['normalize']==1:
            featuresTemp=normalize(featuresTemp)
        del params['normalize']
    t=params['type']
    del params['type']
    if  t=='RandomForestRegressor':
        regressor=RandomForestRegressor(**params)
    elif t=='GradientBoostingRegressor':
        regressor=GradientBoostingRegressor(**params)
    elif t == 'SVR':
        kernel_cfg=params.pop('kernel')
        kernel_name=kernel_cfg.pop('name')
        params.update(kernel_cfg)
        params['kernel']=kernel_name
        regressor=SVR(**params)
    else:
        return 0
    return cross_val_score(regressor,featuresTemp,label,scoring='neg_mean_squared_error')

def f2(params):
    global  bestTemp
    paramsTemp=params.copy()
    tempResult=Regressor_hyperopt_train_test(params)
    mse=-tempResult.mean()
    std=tempResult.std()
    if bestTemp==None:
        bestTemp=mse
    if mse<bestTemp:
        bestTemp = mse
        print("the new best mse is {}  std is {}  using model {} parameters {}".format(mse,std, paramsTemp['type'], paramsTemp))
    return {'loss':mse,'status':STATUS_OK}


def RegressorChoice():
    trials=Trials()
    best=fmin(fn=f2,algo=tpe.suggest,space=Configure.RegressorSpace,max_evals=Configure.Parameter_tunning_max_evals,trials=trials)
    print("the best mse is {}".format(best))

