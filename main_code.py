!pip install lazypredict
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import  mean_absolute_error, r2_score
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.express as px
import warnings
from datetime import datetime as dt
import numpy as np
from sklearn.compose import ColumnTransformer
warnings.simplefilter(action="ignore", category= UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)
df = pd.read_csv("/content/steel_strength.csv")
df.head(15)
df.drop(columns=["formula", "elongation", "tensile strength"], inplace=True)
df.shape
df.duplicated().sum()
df.dtypes
df.isna().sum()
df.describe().T
df.nunique()
df.hist(figsize=(17,12),color='green');
plt.figure(figsize=(15,8))
sns.boxplot(data=df.drop(columns=["yield strength"]), showfliers=True, orient="h")
plt.title("Boxplot of columns");
df.corr().style.background_gradient(axis=None)
df.corrwith(df["yield strength"])
fig = px.scatter(data_frame=df, x="yield strength", y= "co", size= "ti", color= "ni", labels={"ni":"niquel"}, color_continuous_scale="Viridis",
                 title="Yield Strenght vs Cobalt vs Titanium vs Nickel")
fig.update_layout(yaxis_title="cobalt (%)")
fig.show()
def drop_outliers_IQR(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    df_without_outliers = df[~((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))].dropna()
    return df_without_outliers
print("Initial shape:", df.shape)
print("Shape after dropping outliers:", drop_outliers_IQR(df).shape)
X = df.drop(columns=['yield strength'])
y = df['yield strength']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
y_mean = df["yield strength"].mean()
y_pred_mae = [y_mean] * len(df)
acc_baseline = mean_absolute_error(df["yield strength"], y_pred_mae)
print("Baseline MAE:", round(acc_baseline))
# MODEL 1
reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
models,predictions = reg.fit(X_train, X_test, y_train, y_test)
models.head()
# MODEL 2
X_train_sts, X_test_sts = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)
reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
models,predictions = reg.fit(X_train_sts, X_test_sts, y_train, y_test)
models.head()
params = {
    "n_estimators": range(450,1000,100),
    "max_depth": range(20,61,5),
    "criterion": ["squared_error", "absolute_error"],
    "min_samples_split": [2,4],
    "min_samples_leaf": [1,2,4]
}
# MODEL 3
model_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    params,
    cv=5,
    n_jobs=-1,
    n_iter=35,
    scoring=["neg_mean_absolute_error", "r2"],
    refit="neg_mean_absolute_error",
    verbose=1
)
model_rf.fit(X_train, y_train)
cv_results = pd.DataFrame(model_rf.cv_results_)
cv_results.sort_values("rank_test_neg_mean_absolute_error").head()
# Get feature names from training data
features = X_train.columns
# Extract importances from model
importances = model_rf.best_estimator_.feature_importances_
# Create a series with feature names and importances
feat_imp = pd.Series(importances, index=features)
# Plot 10 most important features
feat_imp.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");
mae_rf_train = mean_absolute_error(y_train, model_rf.predict(X_train))
mae_rf_test = mean_absolute_error(y_test, model_rf.predict(X_test))

print("Random Forest:")
print("Training Mean Absolute Error:", round(mae_rf_train, 4))
print("Test Mean Absolute Error:", round(mae_rf_test, 4))
print("Baseline Mean Absolute Error:", round(acc_baseline, 4))
r2_rf_train = r2_score(y_train, model_rf.predict(X_train))
r2_rf_test = r2_score(y_test, model_rf.predict(X_test))

print("Random Forest:")
print("Training R2:", round(r2_rf_train, 4))
print("Test R2:", round(r2_rf_test, 4))
params_br = {
    "n_estimators": range(5,50,5),
}

# MODEL 4
model_br = GridSearchCV(
    BaggingRegressor(random_state=42),
    params_br,
    cv=5,
    n_jobs=-1,
    scoring=["neg_mean_absolute_error", "r2"],
    refit="neg_mean_absolute_error",
    verbose=1
)

model_br.fit(X_train, y_train)
cv_results = pd.DataFrame(model_br.cv_results_)
cv_results.sort_values("rank_test_neg_mean_absolute_error").head()

mae_br_train = mean_absolute_error(y_train, model_br.predict(X_train))
mae_br_test = mean_absolute_error(y_test, model_br.predict(X_test))

print("Bagging Regressor:")
print("Training Mean Absolute Error:", round(mae_br_train, 4))
print("Test Mean Absolute Error:", round(mae_br_test, 4))
print("Baseline Mean Absolute Error:", round(acc_baseline, 4))

r2_br_train = r2_score(y_train, model_br.predict(X_train))
r2_br_test = r2_score(y_test, model_br.predict(X_test))

print("Bagging Regressor:")
print("Training R2:", round(r2_br_train, 4))
print("Test R2:", round(r2_br_test, 4))

params_et = {
    "n_estimators": range(100,1001,100),
    "max_depth": range(20,61,5),
    "criterion": ["squared_error", "absolute_error"],
    "min_samples_split": [2,3],
    "min_samples_leaf": [1,2,3]
}


# MODEL 5
model_et = RandomizedSearchCV(
    ExtraTreesRegressor(random_state=42),
    params_et,
    cv=5,
    n_jobs=-1,
    scoring=["neg_mean_absolute_error", "r2"],
    n_iter=35,
    refit="neg_mean_absolute_error",
    verbose=1
)


model_et.fit(X_train, y_train)
cv_results = pd.DataFrame(model_et.cv_results_)
cv_results.sort_values("rank_test_neg_mean_absolute_error").head()

mae_et_train = mean_absolute_error(y_train, model_et.predict(X_train))
mae_et_test = mean_absolute_error(y_test, model_et.predict(X_test))

print("Extra Trees Regressor:")
print("Training Mean Absolute Error:", round(mae_et_train, 4))
print("Test Mean Absolute Error:", round(mae_et_test, 4))
print("Baseline Mean Absolute Error:", round(acc_baseline, 4))


r2_et_train = r2_score(y_train, model_et.predict(X_train))
r2_et_test = r2_score(y_test, model_et.predict(X_test))

print("Extra Trees Regressor:")
print("Training R2:", round(r2_et_train, 4))
print("Test R2:", round(r2_et_test, 4))


pipe = VotingRegressor(estimators=[("rf", model_rf.best_estimator_),
                                  ("et" ,model_et.best_estimator_),
                                  ("br" ,model_br.best_estimator_)])


params_vr = {
    "weights": [None, [2,3,2], [2,1,2]]
}

model_vr = GridSearchCV(
    pipe,
    params_vr,
    cv=5,
    n_jobs=-1,
    scoring=["neg_mean_absolute_error", "r2"],
    refit="neg_mean_absolute_error",
    verbose=1
)


model_vr.fit(X_train, y_train)

cv_results = pd.DataFrame(model_vr.cv_results_)
cv_results.sort_values('rank_test_neg_mean_absolute_error').head()

mae_vr_train = mean_absolute_error(y_train, model_vr.predict(X_train))
mae_vr_test = mean_absolute_error(y_test, model_vr.predict(X_test))

print("Voting Regressor:")
print("Training Mean Absolute Error:", round(mae_vr_train, 4))
print("Test Mean Absolute Error:", round(mae_vr_test, 4))
print("Baseline Mean Absolute Error:", round(acc_baseline, 4))

r2_vr_train = r2_score(y_train, model_vr.predict(X_train))
r2_vr_test = r2_score(y_test, model_vr.predict(X_test))

print("Voting Regressor:")
print("Training R2:", round(r2_vr_train, 4))
print("Test R2:", round(r2_vr_test, 4))

fig = px.bar(y=["Extra Trees", "Random Forest", "Bagging", "Voting", "Baseline"],
             x=[mae_et_test, mae_rf_test, mae_br_test, mae_vr_test, acc_baseline],
             color=["Extra Trees", "Random Forest", "Bagging", "Voting", "Baseline"],
             color_discrete_map={"Extra Trees": "orange",
                                "Random Forest": "green",
                                "Bagging": "blue",
                                "Voting": "grey",
                                "Baseline": "pink"},
            title="Mean Absolute Error comparison (lower is better)")
fig.update_layout(yaxis={'categoryorder':'total descending'}, xaxis_title="MAE", yaxis_title="Models")
fig.show()


fig = px.bar(y=["Extra Trees", "Random Forest", "Bagging", "Voting"],
             x=[r2_et_test, r2_rf_test, r2_br_test, r2_vr_test],
             color=["Extra Trees", "Random Forest", "Bagging", "Voting"],
             color_discrete_map={"Extra Trees": "orange",
                                "Random Forest": "green",
                                "Bagging": "blue",
                                "Voting": "grey",},
            title="R2-Score comparison (higher is better)")
fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="R2-score", yaxis_title="Models")
fig.show()


