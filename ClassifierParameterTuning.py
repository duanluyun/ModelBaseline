import DataPreprocessing
import ClassifierConfig as Configure
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import  BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,KFold
from hyperopt import fmin,tpe,STATUS_OK,Trials
from sklearn.preprocessing import scale,normalize



bestTemp=None

dataTemp=DataPreprocessing.load_data()
features=dataTemp.data
label=dataTemp.target

def classfier_hyperopt_train_test(params):
    global  features
    global  label
    if 'scale' in params:
        if params['scale']==1:
            features=scale(features)
        del params['scale']
    if 'normalize' in params:
        if params['normalize']==1:
            features=normalize(features)
        del params['normalize']
    t=params['type']
    del params['type']
    if  t=='RandomForestClassifier':
        clf=RandomForestClassifier(**params)
    elif t=='KNeighborsClassifier':
        clf=KNeighborsClassifier(**params)
    elif t=='BernoulliNB':
        clf=BernoulliNB(**params)
    elif t=='SVC':
        kernel_cfg = params.pop('kernel')
        kernel_name = kernel_cfg.pop('name')
        params.update(kernel_cfg)
        params['kernel'] = kernel_name
        clf = SVC(**params)
    else:
        return 0
    return cross_val_score(clf,features,label).mean()

def f(params):
    global  bestTemp
    paramsTemp=params.copy()
    acc=classfier_hyperopt_train_test(params)
    if bestTemp==None:
        bestTemp=acc
    if acc>bestTemp:
        bestTemp=acc
        print("the new best acc is {}  using model {} parameters {}".format(acc,paramsTemp['type'],paramsTemp))
    return {'loss':-acc,'status':STATUS_OK}

def classfierChoice():
    trials=Trials()
    best=fmin(fn=f,space=Configure.classfiertSpace,algo=tpe.suggest,max_evals=Configure.Parameter_tunning_max_evals,trials=trials)
    print(best)






