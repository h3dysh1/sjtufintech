# import datetime
# import pandas as pd
# import numpy as np
# import os
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# from sklearn.metrics import mean_squared_error
# from sklearn.tree import DecisionTreeRegressor

# if __name__ == "__main__":
#     # for timing purpose
#     print(datetime.datetime.now())

#     # turn off pandas Setting with Copy Warning
#     pd.set_option("mode.chained_assignment", None)

#     # set working directory
#     work_dir = "/Users/hedyshi/Downloads/批量下载-Char_Description等4个文件"

#     # read sample data
#     file_path = os.path.join(
#         work_dir, "global_sample_500k.csv"
#     )  # replace with the correct file name
#     raw = pd.read_csv(
#         file_path, parse_dates=["date"], low_memory=False #patch1
#     )  # the date is the first day of the return month (t+1)
#     raw["date"] = pd.to_datetime(raw["date"])
#     raw["char_date"] = pd.to_datetime(raw["char_date"])

#     # read list of predictors for stocks
#     file_path = os.path.join(
#         work_dir, "factor_char_list.csv"
#     )  # replace with the correct file name
#     stock_vars = list(pd.read_csv(file_path)["variable"].values)

#     # define the left hand side variable
#     ret_var = "stock_ret"
#     new_set = raw[
#         raw[ret_var].notna()
#     ].copy()  # create a copy of the data and make sure the left hand side is not missing

#     ### PATCH START — handle missing columns safely ###
#     existing_vars = [v for v in stock_vars if v in new_set.columns]
#     missing_vars = [v for v in stock_vars if v not in new_set.columns]

#     if len(missing_vars) > 0:
#         print("Missing variables (ignored):", missing_vars)

#     stock_vars = existing_vars

#     ### PATCH END ###

#     # transform each variable in each month to the same scale
#     monthly = new_set.groupby("date")
#     data = pd.DataFrame()
#     for date, monthly_raw in monthly:
#         group = monthly_raw.copy()
#         # rank transform each variable to [-1, 1]
#         for var in stock_vars:
#             var_median = group[var].median(skipna=True)
#             group[var] = group[var].fillna(
#                 var_median
#             )  # fill missing values with the cross-sectional median of each month

#             group[var] = group[var].rank(method="dense") - 1
#             group_max = group[var].max()
#             if group_max > 0:
#                 group[var] = (group[var] / group_max) * 2 - 1
#             else:
#                 group[var] = 0  # in case of all missing values
#                 print("Warning:", date, var, "set to zero.")

#         # add the adjusted values
#         data = data._append(
#             group, ignore_index=True
#         )  # append may not work with certain versions of pandas, use concat instead if needed

#     # initialize the starting date, counter, and output data
#     starting = pd.to_datetime("20050101", format="%Y%m%d")
#     counter = 0
#     pred_out = pd.DataFrame()

#     # estimation with expanding window
#     while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime(
#         "20260101", format="%Y%m%d"
#     ):
#         cutoff = [
#             starting,
#             starting
#             + pd.DateOffset(
#                 years=8 + counter
#             ),  # use 8 years and expanding as the training set
#             starting
#             + pd.DateOffset(
#                 years=10 + counter
#             ),  # use the next 2 years as the validation set
#             starting + pd.DateOffset(years=11 + counter),
#         ]  # use the next year as the out-of-sample testing set

#         # cut the sample into training, validation, and testing sets
#         train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
#         validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
#         test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

#         ### PATCH START — skip empty windows to avoid StandardScaler error ###
#         if (len(train) == 0) or (len(validate) == 0) or (len(test) == 0):
#             print(
#                 f"Skipping window {cutoff[0].date()}–{cutoff[3].date()} "
#                 f"(train={len(train)}, val={len(validate)}, test={len(test)})"
#             )
#             counter += 1
#             continue
#         ### PATCH END ###

#         # Optional: if your data has additional binary or categorical variables,
#         # you can further standardize them here
#         scaler = StandardScaler().fit(train[stock_vars])
#         train[stock_vars] = scaler.transform(train[stock_vars])
#         validate[stock_vars] = scaler.transform(validate[stock_vars])
#         test[stock_vars] = scaler.transform(test[stock_vars])

#         # get Xs and Ys
#         X_train = train[stock_vars].values
#         Y_train = train[ret_var].values
#         X_val = validate[stock_vars].values
#         Y_val = validate[ret_var].values
#         X_test = test[stock_vars].values
#         Y_test = test[ret_var].values

#         # de-mean Y (because the regressions are fitted without an intercept)
#         # if you want to include an intercept (or bias in neural networks, etc), you can skip this step
#         Y_mean = np.mean(Y_train)
#         Y_train_dm = Y_train - Y_mean

#         # prepare output data
#         reg_pred = test[
#             ["year", "month", "ret_eom", "id", ret_var]
#         ]  # minimum identifications for each stock

#         # Linear Regression
#         # no validation is needed for OLS
#         reg = LinearRegression(fit_intercept=False)
#         reg.fit(X_train, Y_train_dm)
#         x_pred = reg.predict(X_test) + Y_mean
#         reg_pred["ols"] = x_pred

#         # Lasso
#         lambdas = np.arange(
#             -4, 4.1, 0.1
#         )  # search for the best lambda in the range of 10^-4 to 10^4, range can be adjusted
#         val_mse = np.zeros(len(lambdas))
#         for ind, i in enumerate(lambdas):
#             reg = Lasso(alpha=(10**i), max_iter=1000000, fit_intercept=False)
#             reg.fit(X_train, Y_train_dm)
#             val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

#         # select the best lambda based on the validation set
#         best_lambda = lambdas[np.argmin(val_mse)]
#         reg = Lasso(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
#         reg.fit(X_train, Y_train_dm)
#         x_pred = reg.predict(X_test) + Y_mean  # predict the out-of-sample testing set
#         reg_pred["lasso"] = x_pred

#         # Ridge
#         # same format as above
#         lambdas = np.arange(-1, 8.1, 0.1)
#         val_mse = np.zeros(len(lambdas))
#         for ind, i in enumerate(lambdas):
#             reg = Ridge(alpha=((10**i) * 0.5), fit_intercept=False)
#             reg.fit(X_train, Y_train_dm)
#             val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

#         best_lambda = lambdas[np.argmin(val_mse)]
#         reg = Ridge(alpha=((10**best_lambda) * 0.5), fit_intercept=False)
#         reg.fit(X_train, Y_train_dm)
#         x_pred = reg.predict(X_test) + Y_mean
#         reg_pred["ridge"] = x_pred

#         # Elastic Net
#         # same format as above
#         lambdas = np.arange(-4, 4.1, 0.1)
#         val_mse = np.zeros(len(lambdas))
#         for ind, i in enumerate(lambdas):
#             reg = ElasticNet(alpha=(10**i), max_iter=1000000, fit_intercept=False)
#             reg.fit(X_train, Y_train_dm)
#             val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

#         best_lambda = lambdas[np.argmin(val_mse)]
#         reg = ElasticNet(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
#         reg.fit(X_train, Y_train_dm)
#         x_pred = reg.predict(X_test) + Y_mean
#         reg_pred["en"] = x_pred

#         # De1sion Tree Regressor
#         best_ic = -999
#         best_model = None

#         # minimal hyperparameter grid
#         depth_list = [3, 5, 7]
#         leaf_list = [10, 20]

#         for depth in depth_list:
#             for leaf in leaf_list:
#                 model = DecisionTreeRegressor(
#                     max_depth=depth,
#                     min_samples_leaf=leaf
#                 )
#                 model.fit(X_train, Y_train_dm)

#                 # validation prediction
#                 y_val_pred = model.predict(X_val) + Y_mean

#                 # Compute IC (Spearman correlation) for validation set
#                 ic = pd.Series(y_val_pred).corr(pd.Series(Y_val), method="spearman")

#                 # choose model that maximizes IC
#                 if ic > best_ic:
#                     best_ic = ic
#                     best_model = model

#         # Use best model for test prediction
#         x_pred = best_model.predict(X_test) + Y_mean
#         reg_pred["tree"] = x_pred

#         # add to the output data
#         pred_out = pred_out._append(reg_pred, ignore_index=True)

#         # go to the next year
#         counter += 1

#     # output the predicted value to csv
#     out_path = os.path.join(work_dir, "output.csv")
#     print(out_path)
#     pred_out.to_csv(out_path, index=False)

#     # print the OOS R2
#     yreal = pred_out[ret_var].values
#     for model_name in ["ols", "lasso", "ridge", "en", "tree"]:
#         ypred = pred_out[model_name].values
#         r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
#         print(model_name, r2)

#     # for timing purpose
#     print(datetime.datetime.now())
#     print(data["date"].min(), data["date"].max())
#     print(len(data["date"].unique()))

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
    reg_pred = test[["year", "month", "ret_eom", "id", ret_var]].copy()
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
