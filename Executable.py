#Here, we have to pass only the path of the data to get the result
#Evaluation metric used is ROC_AUC_SCORE and Accuracy Score

#import dependencies
import warnings #to eliminate the unwanted warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
import lightgbm as lgbm

plt.style.use('ggplot')
sns.set_color_codes(palette= 'muted')
sns.set(style= 'ticks', color_codes= True)

pd.options.display.max_columns= 199
pd.options.display.max_rows= 500

path= 'C://Users//sngupta//Documents//hackerearthhackathon//american_express_hiring_challenge//bank_management_data//bank_additional//bank-additional-full.csv'

def read_process_data(path, sep= ';'): #use to read and basic preprocessing of data before ml modeling
    
    df= pd.read_csv(path, sep= sep) #data reading
    print('The shape of the dataset: {}' .format(df.shape)) #print the shape of the data
    
    #Encoding of the categorical features
    mapping= {'yes': 1, 'no': 0, 'unknown': 2}
    df['default']= df['default'].map(mapping)
    df['housing']= df['housing'].map(mapping)
    df['loan']= df['loan'].map(mapping)
    df['y']= df['y'].map(mapping)
    df['month']= df['month'].map({'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'oct': 10, 'nov': 11, 'dec': 12, 'mar': 3, 'apr': 4, 'sep': 9})
    df['day_of_week']= df['day_of_week'].map({'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5})
    df['poutcome']= df['poutcome'].map({'nonexistent': 2, 'failure': 0, 'success': 1})
    df['education']= df['education'].map({'basic.4y': 1, 'high.school': 4, 'basic.6y': 2, 'basic.9y': 3,
       'professional.course': 6, 'unknown': 7, 'university.degree': 5,
       'illiterate': 0})
    
    df= pd.get_dummies(df) #used for the nominal categorical features
    
    #Call model function for the further execution
    prediction= run_LGBM(df.drop(['y'], axis= 1), df['y'])
    
    print(prediction.head(10))
    
    return prediction

#Try using the LightGBM tuned model for that
def run_LGBM(train, target):
    params = {
        'learning_rate': 0.011, #0.011,
        'objective':'binary',
        'max_depth': 8,
        'colsample_bytree': 0.7,
        'bagging_fraction':0.5,
        'feature_fraction':0.6,
        'bagging_seed':101,
        'num_threads': 7,
        'seed': 101,
        'booster': 'gbdt'
        }
    
    X_train, X_test, y_train, y_test= train_test_split(train, target, test_size= 0.2, random_state= 101)
    
    dtrain= lgbm.Dataset(X_train, y_train)
    dtest= lgbm.Dataset(X_test, y_test)
    
    model= lgbm.train(params, dtrain, num_boost_round= 3701, valid_sets= [dtrain, dtest], 
                      valid_names= ['Train', 'Valid'], verbose_eval= 100, early_stopping_rounds= 50)
    
    prediction= model.predict(X_test)
    
    #prediction_2= model.predict(test)
    
    for i in range(len(prediction)):
        if prediction[i]<0.5:
            prediction[i]= 0
        else:
            prediction[i]= 1
            
    
    #accuracy
    print('Testing accuracy: {:.4f}' .format(accuracy_score(y_test, prediction)))
    print('f1 score: {}' .format(f1_score(y_test, prediction)))
    print('ROC AUC: {}' .format(roc_auc_score(y_test, prediction)))
    
    print('Confusion Matrix: ')
    cm= confusion_matrix(y_test, prediction)
    print(cm)
    print('Classification Report: ')
    print(classification_report(y_test, prediction))
    
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print('True Positive Rate: {}' .format(TPR))
    print('True Negative Rate: {}' .format(TNR))
    print('False Positive Rate: {}' .format(FPR))
    print('False Negative Rate: {}' .format(FNR))
    print('Overall Accuracy: {}' .format(ACC))
    
    #plot the feature Importance
    lgbm.plot_importance(model, figsize= (16, 6))
    plt.show()
    
    result_df= pd.DataFrame()
    result_df['Actual']= y_test
    result_df['Predicted']= prediction
    
    result_df['Actual']= result_df['Actual'].map({0: 'no', 1: 'yes'})
    result_df['Predicted']= result_df['Predicted'].map({0: 'no', 1: 'yes'})
    
    return result_df


#Call read_process_data function
prediction= read_process_data(path)