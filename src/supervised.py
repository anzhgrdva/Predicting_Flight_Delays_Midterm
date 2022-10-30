# from re import X
from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

import seaborn as sns

# Define a class
class supervised:
    """
    * Split the data
    * Scale the data if specified
    * Perform the following data:
        * Random search to find best parameters

    Params:
    * df: Dataframe with all data and target variable as last column.
    
    Returns:
    * Best model from random search
    """
    
    def __init__(self, X, y, estimator, param_distributions, model_name):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8,random_state=0)
        self.X_train_pre = Xtrain
        self.X_test_pre =  Xtest
        self.y_train = ytrain
        self.y_test =  ytest
        self.estimator = estimator
        self.params = param_distributions
        self.model_name = model_name

    def get_best_model(self,scaled=False,plot=True):
        """
        * Train the best model from the RandomizedSearch.
        * Print model evalutation metrics: 
            * RMSE
            * Mean absolute error (MAE)
            * R^2 score
        Params:
        - scale (bool): To scale data, pass the argument scaled=True. Default is False.
            Should only be used if all variables are numeric.
        - plot (bool): If true, plot true vs. predicted values using test data set from train-test split.

        Returns: Best model from the RandomizedSearch.
        Attributes:
        - Same attributes as for the given estimator.
        - Evaluation metrics for train and test data subsets:
            - `.r2_train` and `.r2`
            - `.rmse_train` and `.rmse`
            - `.mean_abs_error_train` and `.mean_abs_error`
        """
        if scaled==True:
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train_pre)
            self.X_test = scaler.transform(self.X_test_pre)
            print('**Data has been scaled.**')
        else:
            self.X_train = self.X_train_pre
            self.X_test = self.X_test_pre
            print('**No scaling performed**')
        search = RandomizedSearchCV(self.estimator, param_distributions=self.params, n_iter=4, random_state=0,scoring=None)
        search.fit(self.X_train, self.y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(self.X_test)
        y_pred_train = best_model.predict(self.X_train)
    
       
        # Metrics for test data

        self.rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        self.mean_abs_error = mean_absolute_error(self.y_test, y_pred)
        self.r2 = r2_score(self.y_test, y_pred)

        # Metrics for training data

        self.rmse_train = mean_squared_error(self.y_train, y_pred_train)
        self.mean_abs_error_train = mean_absolute_error(self.y_train, y_pred_train)
        self.r2_train = r2_score(self.y_train, y_pred_train)

        print(f'\n{self.model_name} evaluation metrics: \n\tTest data\tTraining data\t\tDifference')
        print(f'RMSE: \t\t{self.rmse:.2f}\t\t{self.rmse_train:.2f}\t\t{(self.rmse - self.rmse_train):.2f}')
        print(f'MAE: \t\t{self.mean_abs_error:.2f}\t\t{self.mean_abs_error_train:.2f}\t\t{(self.mean_abs_error - self.mean_abs_error_train):.2f}')
        print(f'R^2: \t\t{self.r2:.2f}\t\t{self.r2_train:.2f}\t\t{(self.r2 - self.r2_train):.2f}')
        
        print(f'Best model parameters from randomized search: {search.best_params_}')

        if plot:
            self.fig = sns.scatterplot(x=self.y_test, y=y_pred)
            self.fig.set_xlabel('Predicted')
        
        return best_model

# # Example on how to call it for  Logistical regression
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
