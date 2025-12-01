# import pandas as pd
# import numpy as np


# # read predicted values (output.csv from the other script)
# pred_path = "/Users/hedyshi/Downloads/批量下载-Char_Description等4个文件/output.csv"
# pred = pd.read_csv(pred_path, parse_dates=["date"])
# # pred.columns = map(str.lower, pred.columns)

# # select model (ridge as an example)
# model = "ridge"

# # sort stocks into deciles (10 portfolios) each month based on the predicted returns and calculate portfolio returns
# # portfolio 1 is the decile with the lowest predicted returns, portfolio 10 is the decile with the highest predicted returns
# # portfolio 11 is the long-short portfolio (portfolio 10 - portfolio 1)
# # or you can pick the top and bottom n number of stocks as the long and short portfolios
# predicted = pred.groupby(["year", "month"])[model]
# pred["rank"] = np.floor(
#     predicted.transform(lambda s: s.rank())
#     * 10  # 10 portfolios
#     / predicted.transform(lambda s: len(s) + 1)
# )  # rank stocks into deciles
# pred = pred.sort_values(
#     ["year", "month", "rank", "permno"]
# )  # sort stocks based on the rank
# monthly_port = pred.groupby(["year", "month", "rank"]).apply(
#     lambda df: pd.Series(np.average(df["stock_exret"], axis=0))
# )  # calculate the realized return for each portfolio using realized stock returns, assume equal-weighted portfolios
# monthly_port = monthly_port.unstack().dropna().reset_index()  # reshape the data
# monthly_port.columns = ["year", "month"] + [
#     "port_" + str(x) for x in range(1, 11)
# ]  # rename columns
# monthly_port["port_11"] = (
#     monthly_port["port_10"] - monthly_port["port_1"]
# )  # port 11 is the long-short portfolio

# # Calculate the Sharpe ratio for long-short Portfolio
# # you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately
# sharpe = (
#     monthly_port["port_11"].mean()  # average return
#     / monthly_port["port_11"].std()  # standard deviation of return, volatility
#     * np.sqrt(12)  # annualized
# )  # Sharpe ratio is annualized
# print("Sharpe Ratio:", sharpe)

# # Calculate the cumulative return of the long-short Portfolio
# returns = monthly_port["port_11"].copy()
# returns = returns + 1
# cumulative_returns = returns.cumprod() - 1

# # Max one-month loss of the long-short Portfolio
# max_1m_loss = monthly_port["port_11"].min()
# print("Max 1-Month Loss:", max_1m_loss)

# # Calculate Drawdown of the long-short Portfolio
# monthly_port["log_port_11"] = np.log(
#     monthly_port["port_11"] + 1
# )  # calculate log returns
# monthly_port["cumsum_log_port_11"] = monthly_port["log_port_11"].cumsum(
#     axis=0
# )  # calculate cumulative log returns
# rolling_peak = monthly_port["cumsum_log_port_11"].cummax()
# drawdowns = rolling_peak - monthly_port["cumsum_log_port_11"]
# max_drawdown = drawdowns.max()
# print("Maximum Drawdown:", max_drawdown)
import pandas as pd
import numpy as np

# ==========================================
# Load predicted values
# ==========================================
pred_path = "/Users/hedyshi/Downloads/批量下载-Char_Description等4个文件/output.csv"

# Load predictions, parse ret_eom as date
pred = pd.read_csv(pred_path, parse_dates=["ret_eom"])
pred.rename(columns={"ret_eom": "date"}, inplace=True)

# Choose which model to analyse
model = "ridge"   # or "ols", "lasso", "en", "tree"

# ==========================================
# Rank stocks into deciles based on predicted returns
# ==========================================
pred["rank"] = pred.groupby(["year", "month"])[model].transform(
    lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1
)

# Sort for readability
pred = pred.sort_values(["year", "month", "rank", "id"])

# ==========================================
# Compute equal-weight portfolio returns
# ==========================================
# Use stock_ret as realized return (since stock_exret does not exist)
monthly_port = pred.groupby(["year", "month", "rank"])["stock_ret"].mean()

monthly_port = monthly_port.unstack().dropna().reset_index()
monthly_port.columns = ["year", "month"] + [f"port_{i}" for i in range(1, 11)]

# Long–short portfolio: top decile minus bottom decile
monthly_port["port_11"] = monthly_port["port_10"] - monthly_port["port_1"]

# ==========================================
# Performance Metrics
# ==========================================

# Sharpe ratio (annualized, monthly data)
sharpe = (
    monthly_port["port_11"].mean()
    / monthly_port["port_11"].std()
) * np.sqrt(12)

print("Sharpe Ratio:", sharpe)

# Cumulative return
returns = 1 + monthly_port["port_11"]
cumulative_returns = returns.cumprod() - 1
print("Cumulative Return:", cumulative_returns.iloc[-1])

# Max one-month loss
max_1m_loss = monthly_port["port_11"].min()
print("Max 1-Month Loss:", max_1m_loss)

# Drawdown
monthly_port["log_ret"] = np.log(1 + monthly_port["port_11"])
monthly_port["cumsum_log"] = monthly_port["log_ret"].cumsum()

rolling_peak = monthly_port["cumsum_log"].cummax()
drawdowns = rolling_peak - monthly_port["cumsum_log"]
max_drawdown = drawdowns.max()

print("Maximum Drawdown:", max_drawdown)
