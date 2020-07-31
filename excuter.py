import warnings
import itemConfig as Configure
import ClassifierParameterTuning
import RegressorParameterTuning



def run_master(conf):
    if conf.model=='Classifier':
        ClassifierParameterTuning.classfierChoice()
    elif  conf.model=='Regressor':
        RegressorParameterTuning.RegressorChoice()
    else:
        print("please choice the correct model")



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    run_master(Configure)