import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LassoCV

import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_california_housing
#from sklearn.ensemble import RandomForestRegressor
#from cuml.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, TweedieRegressor, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
#import xgboost as xg
from os import scandir
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
#from xgboost import XGBRegressor
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

    # background = shap.sample(X_train, 100)
    # explainer = shap.KernelExplainer(model.predict, background)
    # explainer.feature_names = list(features)
    # shap_values = explainer.shap_values(X_test)
    #
    # shap.summary_plot(shap_values[0], X_test, feature_names=features)
    #
    # # Correct beeswarm usage
    # shap_explanation = explainer(X_test)
    # shap.plots.beeswarm(shap_explanation[:, :, 0])  # index 0 = first output
    # plt.savefig("beeswarm-ct0.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # explainer = shap.TreeExplainer(model)
    # explainer.feature_names=features
    # shap_values = explainer.shap_values(X_test)
    # shap.plots.beeswarm(shap_values)

    X_test = pd.DataFrame(X_test, columns=features)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap.plots.beeswarm(shap_values[...,1])

if __name__ == "__main__":
       #[df_ML_X, df_ML_Y]= load_csv()
       data = pd.read_csv('h33_ptm_5kb_50kb_rnaseq.csv',na_values=["NA", "null", "?", " "],engine='python')

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


       #y=np.concatenate([y_ct0.reshape(-1,1),y_ct4.reshape(-1,1),y_ct8.reshape(-1,1),y_ct12.reshape(-1,1),y_ct16.reshape(-1,1),y_ct20.reshape(-1,1)],axis=1)
       y=y_ct0.reshape(-1,1)
       idx = np.random.permutation(X.shape[0])
       X_shuffled = X[idx]
       y_shuffled=y[idx]

       X_10k = X_shuffled[:40000]
       y_10k= y[:40000]
       print(X_10k.shape)
       print(y_10k.shape)

       X_train, X_test, y_train, y_test = train_test_split(X_10k, y_10k, test_size=0.30, random_state=999)
       #
       #scalar= StandardScaler()
       poly = PolynomialFeatures(degree=2)
       #X_poly = poly.fit_transform(X)
       X_train_norm = poly.fit_transform(X_train)
       X_test_norm = poly.transform(X_test)
       y_train_norm = poly.fit_transform(y_train)
       y_test_norm = poly.transform(y_test)
       # y_train_log = np.log1p(y_train)
       # y_test_log=np.log1p(y_test)
       print(X_train_norm.shape)
       print(X_test_norm.shape)

       model = LinearRegression()
       model.fit(X_train_norm, y_train_norm)
       y_pred = model.predict(X_test_norm)
       r2 = r2_score(y_test_norm, y_pred)
       print(r2)






