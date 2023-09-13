import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = pd.read_csv(r"C:\Users\Usuario\personal_projects\machineLearning_projects\supervised_learning\XGBoost_projects\XGBoost_house_price_prediction\supervised_learning_projects\data\processed\processed_data.csv")

# Create X and y arrays
X = data.drop("price", axis=1).values
y = data["price"].values
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Instantiate a StandardScaler
scaler = StandardScaler()
# Scale the training and test features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate a PCA object
pca = PCA(n_components=4)
pca_X_train = pca.fit_transform(X_train_scaled)
pca_X_test = pca.transform(X_test_scaled)

# Setup the pipeline steps: steps
steps = [("xgb_model", xgb.XGBRegressor(max_depth = 2, objective = "reg:linear", n_estimators = 25, colsample_bytree = 1, eta = 0.100, reg_lambda = 1))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, pca_X_train, y_train, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))