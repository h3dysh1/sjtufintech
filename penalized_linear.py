import datetime
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    print(datetime.datetime.now())
    pd.set_option("mode.chained_assignment", None)

    # ==========================================
    # LOAD DATA
    # ==========================================
    work_dir = "/Users/hedyshi/Downloads/批量下载-Char_Description等4个文件"

    file_path = os.path.join(work_dir, "global_sample_500k.csv")
    raw = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)

    raw["date"] = pd.to_datetime(raw["date"])
    raw["char_date"] = pd.to_datetime(raw["char_date"])

    # load predictors
    file_path = os.path.join(work_dir, "factor_char_list.csv")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    ret_var = "stock_ret"
    new_set = raw[raw[ret_var].notna()].copy()

    # ==========================================
    # REMOVE NON-EXISTING PREDICTORS
    # ==========================================
    existing_vars = [v for v in stock_vars if v in new_set.columns]
    missing_vars = [v for v in stock_vars if v not in new_set.columns]

    if len(missing_vars) > 0:
        print("Missing variables (ignored):", missing_vars)

    stock_vars = existing_vars

    # ==========================================
    # MONTHLY RANK NORMALIZATION
    # ==========================================
    data = pd.DataFrame()
    for dt, grp in new_set.groupby("date"):
        group = grp.copy()
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(var_median)
            group[var] = group[var].rank(method="dense") - 1

            gmax = group[var].max()
            if gmax > 0:
                group[var] = (group[var] / gmax) * 2 - 1
            else:
                group[var] = 0
                print("Warning:", dt, var, "set to zero")

        data = data._append(group, ignore_index=True)

    # ==========================================
    # DYNAMIC SAMPLE-BASED SPLIT
    # ==========================================
    data = data.sort_values("date")
    unique_dates = sorted(data["date"].unique())
    n = len(unique_dates)

    cut1 = unique_dates[int(n * 0.6)]  # 60% train
    cut2 = unique_dates[int(n * 0.8)]  # 20% val

    train = data[data["date"] < cut1]
    validate = data[(data["date"] >= cut1) & (data["date"] < cut2)]
    test = data[data["date"] >= cut2]

    print("Train rows:", len(train))
    print("Validate rows:", len(validate))
    print("Test rows:", len(test))
    print("Sample date range:", data["date"].min(), "to", data["date"].max())

    # ==========================================
    # SCALING
    # ==========================================
    scaler = StandardScaler()
    scaler.fit(train[stock_vars])

    X_train = scaler.transform(train[stock_vars])
    X_val = scaler.transform(validate[stock_vars])
    X_test = scaler.transform(test[stock_vars])

    Y_train = train[ret_var].values
    Y_val = validate[ret_var].values
    Y_test = test[ret_var].values

    # mean center Y
    Y_mean = np.mean(Y_train)
    Y_train_dm = Y_train - Y_mean

    

    # ==========================================
    # OUTPUT DF FOR TEST PREDICTIONS
    # ==========================================
    reg_pred = test[["year", "month", "ret_eom", "id", "stock_ret", "stock_exret"]].copy()
    pred_out = reg_pred.copy()


    # ==========================================
    # OLS
    # ==========================================
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, Y_train_dm)
    pred_out["ols"] = reg.predict(X_test) + Y_mean

    # ==========================================
    # LASSO
    # ==========================================
    lambdas = np.arange(-2, 3.1, 1)   # MUCH smaller grid
    mse_list = []
    for i in lambdas:
        lasso = Lasso(alpha=(10**i), max_iter=1000000, fit_intercept=False)
        lasso.fit(X_train, Y_train_dm)
        mse = mean_squared_error(Y_val, lasso.predict(X_val) + Y_mean)
        mse_list.append(mse)

    best_i = lambdas[np.argmin(mse_list)]
    lasso = Lasso(alpha=(10**best_i), max_iter=1000000, fit_intercept=False)
    lasso.fit(X_train, Y_train_dm)
    pred_out["lasso"] = lasso.predict(X_test) + Y_mean

    # ==========================================
    # RIDGE
    # ==========================================
    lambdas = np.arange(-1, 8.1, 0.1)
    mse_list = []
    for i in lambdas:
        ridge = Ridge(alpha=(10**i)*0.5, fit_intercept=False)
        ridge.fit(X_train, Y_train_dm)
        mse = mean_squared_error(Y_val, ridge.predict(X_val) + Y_mean)
        mse_list.append(mse)

    best_i = lambdas[np.argmin(mse_list)]
    ridge = Ridge(alpha=(10**best_i)*0.5, fit_intercept=False)
    ridge.fit(X_train, Y_train_dm)
    pred_out["ridge"] = ridge.predict(X_test) + Y_mean

    # ==========================================
    # ELASTIC NET
    # ==========================================
    lambdas = np.arange(-4, 4.1, 0.1)
    mse_list = []
    for i in lambdas:
        en = ElasticNet(alpha=(10**i), max_iter=1000000, fit_intercept=False)
        en.fit(X_train, Y_train_dm)
        mse = mean_squared_error(Y_val, en.predict(X_val) + Y_mean)
        mse_list.append(mse)

    best_i = lambdas[np.argmin(mse_list)]
    en = ElasticNet(alpha=(10**best_i), max_iter=1000000, fit_intercept=False)
    en.fit(X_train, Y_train_dm)
    pred_out["en"] = en.predict(X_test) + Y_mean

    # ==========================================
    # DECISION TREE (IC-BASED)
    # ==========================================
    # Random Forest tuned for cross-sectional equity returns
    rf_model = RandomForestRegressor(
        n_estimators=200,        # number of trees
        max_depth=8,             # shallow to avoid overfitting
        min_samples_leaf=20,     # stabilizes noise
        max_features=0.5,        # feature subsampling (like RF)
        n_jobs=-1,               # use all CPU cores
        random_state=42
    )

    # Fit on demeaned returns
    rf_model.fit(X_train, Y_train_dm)

    # Predict out-of-sample
    x_pred = rf_model.predict(X_test) + Y_mean

    # Save predictions
    reg_pred["tree"] = x_pred
    pred_out["tree"] = x_pred

                 

    # ==========================================
    # SAVE OUTPUT
    # ==========================================
    out_path = os.path.join(work_dir, "output.csv")
    pred_out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # ==========================================
    # OOS R² ACROSS MODELS
    # ==========================================
    yreal = pred_out[ret_var].values
    for model_name in ["ols", "lasso", "ridge", "en", "tree"]:
        ypred = pred_out[model_name].values
        r2 = 1 - np.sum((yreal - ypred)**2) / np.sum(yreal**2)
        print(model_name, r2)

    print(datetime.datetime.now())
