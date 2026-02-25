import os

import numpy as np
from sklearn.linear_model import Lasso, LassoCV

import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, TweedieRegressor, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import xgboost as xg
from os import scandir
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib as mpl
import shap



def shap_explain(model, X_train, X_test):
    features = (

            [   "h33_ct0",
                "h33_ct4",
                "h33_ct8",
                "h33_ct12",
                "h33_ct16",
                "h33_ct20",
                "k27ac_gb_ct0",
                "k27ac_gb_ct4",
                "k27ac_gb_ct8",
                "k27ac_gb_ct12",
                "k27ac_gb_ct16",
                "k27ac_gb_ct20",
                "k27ac_pm_ct0",
                "k27ac_pm_ct4",
                "k27ac_pm_ct8",
                "k27ac_pm_ct12",
                "k27ac_pm_ct16",
                "k27ac_pm_ct20",
                "k4me3_gb_ct0",
                "k4me3_gb_ct4",
                "k4me3_gb_ct8",
                "k4me3_gb_ct12",
                "k4me3_gb_ct16",
                "k4me3_gb_ct20",
                "k4me3_pm_ct0",
                "k4me3_pm_ct4",
                "k4me3_pm_ct8",
                "k4me3_pm_ct12",
                "k4me3_pm_ct16",
                "k4me3_pm_ct20",
                "k36me3_gb_ct0",
                "k36me3_gb_ct4",
                "k36me3_gb_ct8",
                "k36me3_gb_ct12",
                "k36me3_gb_ct16",
                "k36me3_gb_ct20",
                "k36me3_pm_ct0",
                "k36me3_pm_ct4",
                "k36me3_pm_ct8",
                "k36me3_pm_ct12",
                "k36me3_pm_ct16",
                "k36me3_pm_ct20",
                "k4me1_gb_ct0",
                "k4me1_gb_ct4",
                "k4me1_gb_ct8",
                "k4me1_gb_ct12",
                "k4me1_gb_ct16",
                "k4me1_gb_ct20",
                "k4me1_pm_ct0",
                "k4me1_pm_ct4",
                "k4me1_pm_ct8",
                "k4me1_pm_ct12",
                "k4me1_pm_ct16",
                "k4me1_pm_ct20",
                "k79me2_gb_ct0",
                "k79me2_gb_ct4",
                "k79me2_gb_ct8",
                "k79me2_gb_ct12",
                "k79me2_gb_ct16",
                "k79me2_gb_ct20",
                "k79me2_pm_ct0",
                "k79me2_pm_ct4",
                "k79me2_pm_ct8",
                "k79me2_pm_ct12",
                "k79me2_pm_ct16",
                "k79me2_pm_ct20",
            ]
    )

    #X_test_df = pd.DataFrame(X_test, columns=features)
    X_test_df = pd.DataFrame(X_test)
    explainer = shap.TreeExplainer(model)
    # Calculates the SHAP values - It takes some time
    exp = explainer(X_test_df)
    shap_values = exp.values
    #exp.feature_names = list(features)

    shap.plots.beeswarm(exp[..., 0])



if __name__ == "__main__":
       #[df_ML_X, df_ML_Y]= load_csv()
       data = pd.read_csv('h33_ptm_2kb.csv',na_values=["NA", "null", "?", " "],engine='python')
       print(data.columns)

       h33_ct0 = data["h3.3_ct0"].values
       h33_ct4 = data["h3.3_ct4"].values
       h33_ct8 = data["h3.3_ct8"].values
       h33_ct12 = data["h3.3_ct12"].values
       h33_ct16 = data["h3.3_ct16"].values
       h33_ct20 = data["h3.3_ct20"].values
       k27ac_gb_ct0=data["k27ac_gene_body_ct0"].values
       k27ac_gb_ct4=data["k27ac_gene_body_ct4"].values
       k27ac_gb_ct8=data["k27ac_gene_body_ct8"].values
       k27ac_gb_ct12=data["k27ac_gene_body_ct12"].values
       k27ac_gb_ct16=data["k27ac_gene_body_ct16"].values
       k27ac_gb_ct20=data["k27ac_gene_body_ct20"].values



       k27ac_pm_ct0=data["k27ac_promoters_ct0"].values
       k27ac_pm_ct4=data["k27ac_promoters_ct4"].values
       k27ac_pm_ct8=data["k27ac_promoters_ct8"].values
       k27ac_pm_ct12=data["k27ac_promoters_ct12"].values
       k27ac_pm_ct16=data["k27ac_promoters_ct16"].values
       k27ac_pm_ct20=data["k27ac_promoters_ct20"].values

       k4me3_gb_ct0=data["k4me3_gene_body_ct0"].values
       k4me3_gb_ct4 = data["k4me3_gene_body_ct4"].values
       k4me3_gb_ct8 = data["k4me3_gene_body_ct8"].values
       k4me3_gb_ct12 = data["k4me3_gene_body_ct12"].values
       k4me3_gb_ct16 = data["k4me3_gene_body_ct16"].values
       k4me3_gb_ct20 = data["k4me3_gene_body_ct20"].values

       k4me3_pm_ct0 = data["k4me3_promoters_ct0"].values
       k4me3_pm_ct4 = data["k4me3_promoters_ct4"].values
       k4me3_pm_ct8 = data["k4me3_promoters_ct8"].values
       k4me3_pm_ct12 = data["k4me3_promoters_ct12"].values
       k4me3_pm_ct16 = data["k4me3_promoters_ct16"].values
       k4me3_pm_ct20 = data["k4me3_promoters_ct20"].values

       k36me3_gb_ct0=data["k36me3_gene_body_ct0"].values
       k36me3_gb_ct4 = data["k36me3_gene_body_ct4"].values
       k36me3_gb_ct8 = data["k36me3_gene_body_ct8"].values
       k36me3_gb_ct12 = data["k36me3_gene_body_ct12"].values
       k36me3_gb_ct16 = data["k36me3_gene_body_ct16"].values
       k36me3_gb_ct20 = data["k36me3_gene_body_ct20"].values

       k36me3_pm_ct0 = data["k36me3_promoters_ct0"].values
       k36me3_pm_ct4 = data["k36me3_promoters_ct4"].values
       k36me3_pm_ct8 = data["k36me3_promoters_ct8"].values
       k36me3_pm_ct12 = data["k36me3_promoters_ct12"].values
       k36me3_pm_ct16 = data["k36me3_promoters_ct16"].values
       k36me3_pm_ct20 = data["k36me3_promoters_ct20"].values

       k4me1_pm_ct0 = data["k4me1_promoters_ct0"].values
       k4me1_pm_ct4 = data["k4me1_promoters_ct4"].values
       k4me1_pm_ct8 = data["k4me1_promoters_ct8"].values
       k4me1_pm_ct12 = data["k4me1_promoters_ct12"].values
       k4me1_pm_ct16 = data["k4me1_promoters_ct16"].values
       k4me1_pm_ct20 = data["k4me1_promoters_ct20"].values

       k4me1_gb_ct0=data["k4me1_gene_body_ct0"].values
       k4me1_gb_ct4 = data["k4me1_gene_body_ct4"].values
       k4me1_gb_ct8 = data["k4me1_gene_body_ct8"].values
       k4me1_gb_ct12 = data["k4me1_gene_body_ct12"].values
       k4me1_gb_ct16 = data["k4me1_gene_body_ct16"].values
       k4me1_gb_ct20 = data["k4me1_gene_body_ct20"].values

       k79me2_gb_ct0=data["k79me2_gene_body_ct0"].values
       k79me2_gb_ct4 = data["k79me2_gene_body_ct4"].values
       k79me2_gb_ct8 = data["k79me2_gene_body_ct8"].values
       k79me2_gb_ct12 = data["k79me2_gene_body_ct12"].values
       k79me2_gb_ct16 = data["k79me2_gene_body_ct16"].values
       k79me2_gb_ct20 = data["k79me2_gene_body_ct20"].values

       k79me2_pm_ct0 = data["k79me2_promoters_ct0"].values
       k79me2_pm_ct4 = data["k79me2_promoters_ct4"].values
       k79me2_pm_ct8 = data["k79me2_promoters_ct8"].values
       k79me2_pm_ct12 = data["k79me2_promoters_ct12"].values
       k79me2_pm_ct16 = data["k79me2_promoters_ct16"].values
       k79me2_pm_ct20 = data["k79me2_promoters_ct20"].values

       y_ct0=data["ct0_rpkm_cm_avg"].values
       y_ct4=data["ct4_rpkm_cm_avg"].values
       y_ct8=data["ct8_rpkm_cm_avg"].values
       y_ct12=data["ct12_rpkm_cm_avg"].values
       y_ct16=data["ct16_rpkm_cm_avg"].values
       y_ct20=data["ct20_rpkm_cm_avg"].values


       #X = np.concatenate([h33_ct0.reshape(-1,1),k27ac_gb_ct0.reshape(-1,1),k27ac_gb_ct4.reshape(-1,1),k27ac_gb_ct8.reshape(-1,1),k27ac_gb_ct12.reshape(-1,1),k27ac_gb_ct16.reshape(-1,1),k27ac_gb_ct20.reshape(-1,1),k27ac_pm_ct0.reshape(-1,1),k27ac_pm_ct4.reshape(-1,1),k27ac_pm_ct8.reshape(-1,1),k27ac_pm_ct8.reshape(-1,1),k27ac_pm_ct12.reshape(-1,1),k27ac_pm_ct16.reshape(-1,1),k27ac_pm_ct20.reshape(-1,1),k4me3_gb_ct0.reshape(-1,1),k4me3_gb_ct4.reshape(-1,1),k4me3_gb_ct8.reshape(-1,1),k4me3_gb_ct12.reshape(-1,1),k4me3_gb_ct16.reshape(-1,1),k4me3_gb_ct20.reshape(-1,1),k4me3_pm_ct0.reshape(-1,1),k4me3_pm_ct4.reshape(-1,1),k4me3_pm_ct8.reshape(-1,1),k4me3_pm_ct8.reshape(-1,1),k4me3_pm_ct12.reshape(-1,1),k4me3_pm_ct16.reshape(-1,1),k4me3_pm_ct20.reshape(-1,1)],axis=1)
       X = np.concatenate([h33_ct0.reshape(-1, 1),h33_ct4.reshape(-1, 1),h33_ct8.reshape(-1, 1),h33_ct12.reshape(-1, 1),h33_ct16.reshape(-1, 1),h33_ct20.reshape(-1, 1),
                           k27ac_gb_ct0.reshape(-1, 1), k27ac_gb_ct4.reshape(-1, 1),k27ac_gb_ct8.reshape(-1, 1), k27ac_gb_ct12.reshape(-1, 1), k27ac_gb_ct16.reshape(-1, 1),k27ac_gb_ct20.reshape(-1, 1),
                           k27ac_pm_ct0.reshape(-1, 1), k27ac_pm_ct4.reshape(-1, 1),k27ac_pm_ct8.reshape(-1, 1),  k27ac_pm_ct12.reshape(-1, 1),k27ac_pm_ct16.reshape(-1, 1), k27ac_pm_ct20.reshape(-1, 1),
                           k4me3_gb_ct0.reshape(-1, 1),k4me3_gb_ct4.reshape(-1, 1), k4me3_gb_ct8.reshape(-1, 1), k4me3_gb_ct12.reshape(-1, 1),k4me3_gb_ct16.reshape(-1, 1), k4me3_gb_ct20.reshape(-1, 1),
                           k4me3_pm_ct0.reshape(-1, 1),k4me3_pm_ct4.reshape(-1, 1), k4me3_pm_ct8.reshape(-1, 1), k4me3_pm_ct12.reshape(-1, 1), k4me3_pm_ct16.reshape(-1, 1), k4me3_pm_ct20.reshape(-1, 1),
                           k36me3_gb_ct0.reshape(-1, 1), k36me3_gb_ct4.reshape(-1, 1), k36me3_gb_ct8.reshape(-1, 1),k36me3_gb_ct12.reshape(-1, 1), k36me3_gb_ct16.reshape(-1, 1), k36me3_gb_ct20.reshape(-1, 1),
                           k36me3_pm_ct0.reshape(-1, 1), k36me3_pm_ct4.reshape(-1, 1), k36me3_pm_ct8.reshape(-1, 1),k36me3_pm_ct12.reshape(-1, 1), k36me3_pm_ct16.reshape(-1, 1), k36me3_pm_ct20.reshape(-1, 1),
                           k4me1_gb_ct0.reshape(-1, 1), k4me1_gb_ct4.reshape(-1, 1), k4me1_gb_ct8.reshape(-1, 1),k4me1_gb_ct12.reshape(-1, 1), k4me1_gb_ct16.reshape(-1, 1), k4me1_gb_ct20.reshape(-1, 1),
                           k4me1_pm_ct0.reshape(-1, 1), k4me1_pm_ct4.reshape(-1, 1), k4me1_pm_ct8.reshape(-1, 1), k4me1_pm_ct12.reshape(-1, 1), k4me1_pm_ct16.reshape(-1, 1), k4me1_pm_ct20.reshape(-1, 1),
                           k79me2_gb_ct0.reshape(-1, 1), k79me2_gb_ct4.reshape(-1, 1), k79me2_gb_ct8.reshape(-1, 1),k79me2_gb_ct12.reshape(-1, 1), k79me2_gb_ct16.reshape(-1, 1), k79me2_gb_ct20.reshape(-1, 1),
                           k79me2_pm_ct0.reshape(-1, 1), k79me2_pm_ct4.reshape(-1, 1), k79me2_pm_ct8.reshape(-1, 1),k79me2_pm_ct12.reshape(-1, 1), k79me2_pm_ct16.reshape(-1, 1), k79me2_pm_ct20.reshape(-1, 1)
                           ],axis=1)
       #
       # X = np.concatenate(
       #     [
       #      k79me2_gb_ct0.reshape(-1, 1), k79me2_gb_ct4.reshape(-1, 1), k79me2_gb_ct8.reshape(-1, 1),
       #      k79me2_gb_ct12.reshape(-1, 1), k79me2_gb_ct16.reshape(-1, 1), k79me2_gb_ct20.reshape(-1, 1),
       #      k79me2_pm_ct0.reshape(-1, 1), k79me2_pm_ct4.reshape(-1, 1), k79me2_pm_ct8.reshape(-1, 1),
       #      k79me2_pm_ct12.reshape(-1, 1), k79me2_pm_ct16.reshape(-1, 1), k79me2_pm_ct20.reshape(-1, 1)
       #      ], axis=1)

       y=np.concatenate([y_ct0.reshape(-1,1),y_ct4.reshape(-1,1),y_ct8.reshape(-1,1),y_ct12.reshape(-1,1),y_ct16.reshape(-1,1),y_ct20.reshape(-1,1)],axis=1)

       print(X.shape)
       print(y.shape)

       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
       #
       scalar= StandardScaler()
       X_train_norm = scalar.fit_transform(X_train)
       X_test_norm = scalar.transform(X_test)
       y_train_norm = scalar.fit_transform(y_train)
       y_test_norm = scalar.transform(y_test)
       # y_train_log = np.log1p(y_train)
       # y_test_log=np.log1p(y_test)
       #
       #
       #
       #
       #model=LinearRegression()
       # #model = SVR(kernel='rbf',C=0.01)
       model=RandomForestRegressor(n_estimators=100)
       # #model=XGBRegressor()
       # #model = Ridge(alpha=1)
       # #model=Lasso(alpha=0.01)
       # #model=TweedieRegressor(power=1, link='log',max_iter=1000,alpha=0.5)

       model.fit(X_train_norm, y_train)
       y_pred = model.predict(X_test_norm)
       r2 = r2_score(y_test, y_pred)

       rna_seq_df = pd.DataFrame({"rna-seq-predicted":y_pred.ravel(),"rna-seq-true":y_test.ravel()})
       rna_seq_df.to_csv("RNA-seq-predicted.csv",index=False)

       #kf = KFold(n_splits=5, shuffle=True)
       #scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
       #print('Mean R2 across 5 folds: ',scores.mean())
       #print('Std R2 across 5 folds: ', scores.std())

       # rmse = np.sqrt(mean_squared_error(y_test_log, y_pred))
       # #scores = cross_val_score(model, X, y, scoring="r2", cv=5)
       #
       print(r2)
       #shap_explain(model, X_train_norm, X_test_norm)
