### ONLY predictions with the choosen model..

# Importing libaries
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import GUI as g

def pred_new(model, x):
    x = np.array(x).reshape(1,-1)
    # scaling:
    x = scaler.transform(x)
    pred = model.predict(x)
    print(f'{model} prediction is: {pred}')
    return pred

def pred_new_poly(model, x, degree):
    poly_reg = PolynomialFeatures (degree = degree)
    x = np.array(x).reshape(1,-1)
    # scaling:
    x = scaler.transform(x)
    x = poly_reg.fit_transform(x)
    pred = model.predict(x)
    print(f'{model} prediction is: {pred}')
    return pred

### Preperation of data

#reading data
df = pd.read_csv(r"C:\Users\Orang\OneDrive - Ariel University\Desktop\שנה ד\סמסטר ב\Data mining\Project\ready_data.csv", sep=',')

# dropping first and second column (not relevnt)
df.drop(columns=['precise_change'], inplace=True)

# splitting target and features.
X = df.drop(columns=['j_integral', 'Model Properties change '], inplace=False)
X = X.iloc[:,0:12]
X.drop(columns=['nu_T'], inplace=True)
y = df.loc[:,['j_integral']]

# scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# splitting data to train and test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# building the model after grid search best parameters.
from xgboost import XGBRegressor
XGboost_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1,
             monotone_constraints='()', n_estimators=400, n_jobs=16,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
XGboost_model.fit(X, y)


                             
### Lets Predict values!

x = g.gui()
g.gui(pred_new(XGboost_model, x))



