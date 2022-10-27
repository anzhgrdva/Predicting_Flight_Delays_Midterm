import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost
from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



# Define a class
class supervised:
    """
    * Split the data
    * Scale the data
    * Perform the following for both raw and scaled data:
        * Random search to find best parameters
        * Print model evalutation metrics: 
            * recall
            * precision
            * F1
            * AUC score
        * Plot:
            * confusion matrix
            * ROC
    Params:
    * df: Dataframe with all data and target variable as last column.
    
    Returns:
    * Best model from random search
    """
    def __init__(self, df, estimator, param_distributions, model_name):
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=0)
        self.X_train_pre = X_train
        self.X_test_pre =  X_test
        self.y_train = y_train
        self.y_test =  y_test
        self.estimator = estimator
        self.params = param_distributions
        self.model_name = model_name

    def get_best_model(self,scaled=True):
        if scaled==True:
            scaler = MinMaxScaler()
            self.X_train = scaler.fit_transform(self.X_train_pre)
            self.X_test = scaler.transform(self.X_test_pre)
            print('**Data has been scaled.**')
        else:
            self.X_train = self.X_train_pre
            self.X_test = self.X_test_pre
            print('**Data not scaled**')
        search = RandomizedSearchCV(self.estimator, param_distributions=self.params, random_state=0,n_jobs=-2,scoring='recall')
        search.fit(self.X_train, self.y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(self.X_test)
        y_pred_train = best_model.predict(self.X_train)

        # Metrics for test data
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1score = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred)

        # Metrics for training data
        recall_train = recall_score(self.y_train, y_pred_train)
        precision_train = precision_score(self.y_train, y_pred_train)
        f1score_train = f1_score(self.y_train, y_pred_train)
        auc_train = roc_auc_score(self.y_train, y_pred_train)

        print(f'\n{self.model_name} evaluation metrics: \n\tTest data\tTraining data\t\tDifference')
        print(f'Recall: \t{100*recall:.2f}%\t\t{100*recall_train:.2f}%\t\t{100*(recall-recall_train):.2f}%')
        print(f'Precision: \t{100*precision:.2f}%\t\t{100*precision_train:.2f}%\t\t{100*(precision-precision_train):.2f}%')
        print(f'F1: \t\t{100*f1score:.2f}%\t\t{100*f1score_train:.2f}%\t\t{100*(f1score-f1score_train):.2f}%')
        print(f'AUC: \t\t{100*auc:.2f}%\t\t{100*auc_train:.2f}%\t\t{100*(auc-auc_train):.2f}%')
        
        print(f'Best model parameters from randomized search: {search.best_params_}')
        ConfusionMatrixDisplay.from_estimator(best_model, self.X_test, self.y_test)
        RocCurveDisplay.from_estimator(best_model, self.X_test, self.y_test)
        return best_model






# ## Silvia 2022-10-25 20:48 Example of how to call the custom function for supervised learning
# param_lr = {
#     # 'penalty': ['l1','l2', 'elasticnet'],
#     'C': C_list,
#     'max_iter' : max_iter_list,
#     'class_weight': [None, 'balanced']
# }

# lr = LogisticRegression(random_state=0)
# lr_attributes = supervised(df, lr, param_lr, model_name='logistical regression')
# best_lr = lr_attributes.get_best_model()

# # Save the model
# model = best_lr

# filename = 'model_best_lr.sav'
# pickle.dump(model, open(filename, 'wb'))
