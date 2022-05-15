# Predictions of J_integral with given properties

The project is dealing with research that conducted by Mor Mega.
the researcher's goal is to find solutions for the industry regarding
finding the best values of properties to prevent fractures. 

## What the project contains?
3 python files.
# 1st file- comparing_models
the first step is split the data into train, test and evaluate the best models to predict j_integral values,
the files gives table with Neg MSE and R^2 evalutaion methods and comparing between the models.
# 2st file- GUI
we build simple GUI to let the user interact and use the predictions, the way it works
is simple the user enteres the properties values and gets the j_integral prediction
based on the chosen model.
# 3st file- Predictions
this file is training the model on the full data provided in the file,
and then runs automaticly the GUI, the user can run this file only
to get is predictions, the chosen model at the moment is XGboost,
and therefore the predictions in this file will base on XGboost model.

