# Importing libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

#reading data
df = pd.read_csv(r"C:\Users\Orang\OneDrive - Ariel University\Desktop\שנה ד\סמסטר ב\Data mining\Project\ready_data.csv", sep=',')
print(df.info())

# dropping first and second column (not relevnt)
df.drop(columns=['precise_change'], inplace=True)

# splitting target and features.
X = df.drop(columns=['j_integral', 'Model Properties change '], inplace=False)
y = df.loc[:,['j_integral']]

X = X.iloc[:,0:12]
X.drop(columns=['nu_T'], inplace=True) # after i dropped this correlated column much better results.

## checking for multicollinearity 
# heat map
import seaborn as sns
matrix = np.triu(X.corr(), 1)
plt.figure(figsize = (10, 10))
sns.heatmap(X.corr(), annot = True, square = True, annot_kws={"size": 7}, fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix', fontsize = 10)
plt.show


# colclutions: found conntections.
### Pair plots
new_df = df.loc[:,['E_T', 'nu_T', 'G_T', 'Model Properties change ']]
sns.pairplot(new_df, diag_kind="kde", hue="Model Properties change ")

new_df = df.loc[:,['G_{21} = G_{23}', 'nu_{21} = nu_{23}', 'Model Properties change ']]
sns.pairplot(new_df, diag_kind="hist", hue="Model Properties change ")

### influences plots.
df_2 = pd.read_csv(r"C:\Users\Orang\OneDrive - Ariel University\Desktop\שנה ד\סמסטר ב\Data mining\Project\another_data.csv", sep=',')
sns.barplot(data=df_2, x="values", y="j_integral", hue="Model Properties change ")

#### Predictions:

# scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# splitting data to train and test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# DF for summery table:
final_table = pd.DataFrame(columns=['Model', 'best_Parameters', 'MSE_CV_score', 'R^2', 'R_adj'])
final_table['Model'] = ['Linear regression', 'Polynomial',
                            'Ridge regression', 'Lasso', 'Random forest',
                            'Knn', 'XGboost', 'SVR']

### Linear regression:
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

#cross validation
from sklearn.model_selection import cross_val_score
cv = LeaveOneOut()
accuracies = cross_val_score(estimator=lin_reg, X=X_test, y=y_test, cv=cv, scoring='neg_mean_squared_error', n_jobs = -1)


final_table.iloc[0,1] = 'No_Par'
final_table.iloc[0,2] = accuracies.mean()
final_table.iloc[0,3] = r2_score(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print("linear regression")
print(f"mean of r2: {r2}")
print(f"rmse is: {rmse}\n")

### Polynomial regression:
from sklearn.preprocessing import PolynomialFeatures
# degree 2
poly_reg = PolynomialFeatures (degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y_train)
y_poly_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2 = r2_score(y_test, y_poly_pred)
dj_r2 = 1 - ((1 - r2) * (len(y) - 1) / (len(y) - X_test.shape[1] - 1))

accuracies = cross_val_score(estimator=lin_reg_2, X=X_test, y=y_test, cv=cv, scoring='neg_mean_squared_error', n_jobs = -1)


print("Polyminal regression degree 2: ")
print(f"mean of r2: {r2}")
print(f"rmse is: {rmse}\n")

# degree 3
poly_reg = PolynomialFeatures (degree = 3)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly,y_train)
y_poly_pred = lin_reg_3.predict(poly_reg.fit_transform(X_test))

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2 = r2_score(y_test, y_poly_pred)
dj_r2 = 1 - ((1 - r2) * (len(y) - 1) / (len(y) - X_test.shape[1] - 1))

accuracies = cross_val_score(estimator=lin_reg_3, X=X_test, y=y_test, cv=cv, scoring='neg_mean_squared_error', n_jobs = -1)

final_table.iloc[1,4] = dj_r2
final_table.iloc[1,3] = r2
final_table.iloc[1,2] = accuracies.mean()
final_table.iloc[1,1] = 'degree = 3'

print("Polyminal regression degree 3: ")
print(f"mean of r2: {dj_r2}")
print(f"rmse is: {rmse}\n")

# degree 4
poly_reg_4 = PolynomialFeatures (degree = 4)
X_poly = poly_reg_4.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y_train)
y_poly_pred = lin_reg_2.predict(poly_reg_4.fit_transform(X_test))

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2 = r2_score(y_test, y_poly_pred)
dj_r2 = 1 - ((1 - r2) * (len(y) - 1) / (len(y) - X_test.shape[1] - 1))

print("Polyminal regression degree 4: ")
print(f"mean of r2: {dj_r2}")
print(f"rmse is: {rmse}\n") # best polynom degree is 3.

### Ridge regression:
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from sklearn.linear_model import RidgeCV, MultiTaskLassoCV
ridge_model = RidgeCV(alphas=np.linspace(0.001, 5.0, num=400), cv=3)
# fit model
ridge_model.fit(X_train, y_train)
pred = ridge_model.predict(X_test)

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test, pred)
dj_r2 = 1 - ((1 - r2) * (len(y) - 1) / (len(y) - X_test.shape[1] - 1))

accuracies = cross_val_score(estimator=ridge_model, X=X_test, y=y_test, cv=cv, scoring='neg_mean_squared_error', n_jobs = -1)
print("Ridge regression")
print(f"best alpha is: {ridge_model.alpha_}") # best alpha for ridge
print(f"mean of r2: {dj_r2}")
print(f"rmse is: {rmse}\n")

final_table.iloc[2,4] = dj_r2
final_table.iloc[2,3] = r2
final_table.iloc[2,2] = accuracies.mean()
final_table.iloc[2,1] = f"best alpha is: {ridge_model.alpha_}\n"

### Lasso regression:
lasso_model = MultiTaskLassoCV(alphas=np.linspace(0.001, 5.0, num=400), cv=3, n_jobs = -1)
# fit model
lasso_model.fit(X_train, y_train)

pred = lasso_model.predict(X_test)

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test, pred)
dj_r2 = 1 - ((1 - r2) * (len(y) - 1) / (len(y) - X_test.shape[1] - 1))

accuracies = cross_val_score(estimator=lasso_model, X=X_test, y=y_test, cv=cv, scoring='neg_mean_squared_error', n_jobs = -1)
print("Lasso regression")
print(f"best alpha is: {lasso_model.alpha_}") # best alpha for ridge
print(f"mean of r2: {dj_r2}")

final_table.iloc[3,4] = dj_r2
final_table.iloc[3,3] = r2
final_table.iloc[3,2] = accuracies.mean()
final_table.iloc[3,1] = f"best alpha is: {lasso_model.alpha_}"


# Compute the cross-validation score with the default hyper-parameters
from sklearn.model_selection import cross_val_score
for Model in [Ridge, Lasso]:
    model = Model()
    print('%s: %s' % (Model.__name__,
                      cross_val_score(model, X_train, y_train).mean()))

# We compute the cross-validation score as a function of alpha, the strength of the regularization for Lasso and Ridge


alphas = np.linspace(0.01, 5.0, num=100)

plt.figure(figsize=(5, 3))

for Model in [Lasso, Ridge]:
    scores = [cross_val_score(Model(alpha), X_test, y_test, cv=3, n_jobs = -1).mean()
            for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()
print(max(scores))

### Random forest:
from sklearn.ensemble import RandomForestRegressor
regr_multirf = RandomForestRegressor(n_estimators=10, random_state=0)
regr_multirf.fit(X_train, y_train)

param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 1, stop = 80, num = 10)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [2,4],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False]}

from sklearn.model_selection import RandomizedSearchCV
model_ranforest = RandomizedSearchCV(estimator = regr_multirf, param_distributions = param_grid, cv = 3, verbose=2, n_jobs = -1)
model_ranforest.fit(X_train, y_train)

pred = model_ranforest.predict(X_test)

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test, pred)

accuracies = cross_val_score(estimator=model_ranforest, X=X_test, y=y_test, cv=cv, scoring='neg_mean_squared_error', n_jobs = -1)
print("Random Forest: ")
print(f"The best parameters:\n {model_ranforest.best_params_}")
print(f"mean of r2: {accuracies.mean()}")

final_table.iloc[4,3] = r2
final_table.iloc[4,2] = accuracies.mean()
final_table.iloc[4,1] = model_ranforest.best_params_['n_estimators']

### Knn Regressor:
from sklearn.neighbors import KNeighborsRegressor

algorithm = KNeighborsRegressor()   

hp_candidates = [{'n_neighbors': [2,3,4,5,6], 'weights': ['uniform','distance']}]
knn_model = RandomizedSearchCV(estimator=algorithm, param_distributions=hp_candidates, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
knn_model.fit(X_train, y_train)

pred = knn_model.predict(X_test)

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test, pred)

accuracies = cross_val_score(estimator=knn_model, X=X_test, y=y_test, cv=cv, scoring='neg_mean_squared_error', n_jobs = -1)
print("KNeighbors regression: ")
print(f"The best parameters:\n {knn_model.best_params_}")
print(f"mean of r2: {accuracies.mean()}")

final_table.iloc[5,3] = r2
final_table.iloc[5,2] = accuracies.mean()
final_table.iloc[5,1] = knn_model.best_params_['n_neighbors']


### XGBoost Regressor
from xgboost import XGBRegressor
XGboost_model = XGBRegressor()


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

## Hyper Parameter Optimization
# hyperparameter_grid = {
#     'n_estimators': [100, 400, 800],
#     'max_depth': [3, 6, 9],
#     'learning_rate': [0.05, 0.1, 0.20],
#     'min_child_weight': [1, 10, 100]
#     }
# # Set up the random search with 4-fold cross validation
# random_cv = RandomizedSearchCV(estimator=XGboost_model,
#             param_distributions=hyperparameter_grid,
#             cv=cv, n_iter=50,
#             scoring = 'neg_mean_absolute_error',n_jobs = 4,
#             verbose = 5, 
#             return_train_score = True,
#             random_state=42)

# random_cv.fit(X_train,y_train)
# random_cv.best_estimator_


pred = XGboost_model.predict(X_test)
accuracies = cross_val_score(XGboost_model, X_test, y_test, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# force scores to be positive
# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test, pred)

print("XGboost: ")
print(f"mean of r2: {accuracies.mean()}")

final_table.iloc[6,3] = r2
final_table.iloc[6,2] = accuracies.mean()


### SVM regressor:
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
svr_model = GridSearchCV(SVR(),param_grid,refit=True,verbose=2, n_jobs = -1)
svr_model.fit(X_train,y_train)

pred = svr_model.predict(X_test)

# cross validation and evaluation for degree level:
rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test, pred)

accuracies = cross_val_score(estimator=knn_model, X=X_test, y=y_test, cv=cv, scoring='neg_mean_squared_error', n_jobs = -1)
print("SVM regression (RBF kernel): ")
print(f"The best parameters:\n {knn_model.best_params_}")
print(f"mean of r2: {accuracies.mean()}")

final_table.iloc[7,3] = r2
final_table.iloc[7,2] = accuracies.mean()
final_table.iloc[7,1] = svr_model.best_estimator_

### FINAL SUMMARY of all models and what i choose for the data prediction:

print(final_table)

####### PREDICTIONS #######
