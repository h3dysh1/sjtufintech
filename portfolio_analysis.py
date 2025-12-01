import pandas as pd
import numpy as np


# read predicted values (output.csv from the other script)
pred_path = "/Users/hedyshi/Downloads/批量下载-Char_Description等4个文件/output.csv"
pred = pd.read_csv(pred_path)

# Create excess returns using cross-sectional median as risk-free proxy
pred["rf_proxy"] = pred.groupby(["year","month"])["stock_ret"].transform("median")
pred["stock_exret"] = pred["stock_ret"] - pred["rf_proxy"]

# select model (ridge as an example)
model = "tree"

# sort stocks into deciles (10 portfolios) each month based on the predicted returns and calculate portfolio returns
# portfolio 1 is the decile with the lowest predicted returns, portfolio 10 is the decile with the highest predicted returns
# portfolio 11 is the long-short portfolio (portfolio 10 - portfolio 1)
# or you can pick the top and bottom n number of stocks as the long and short portfolios
pred["pred_signal"] = pred.groupby("id")[model].shift(1)
pred = pred.dropna(subset=["pred_signal"])

# ================== OPTIMISATION SETTINGS ==================
long_sizes  = [50, 75, 100, 125, 150]   # number of long positions to test
short_sizes = [50, 75, 100, 125, 150]   # number of short positions to test

results = []  # store Sharpe, returns, drawdown for each config

# ================== MAIN OPTIMISATION LOOP ==================
for K in long_sizes:
    for J in short_sizes:

        # ================= enforce 100–200 total holdings ==================
        if not (100 <= K + J <= 200):
            continue  # skip invalid configurations
        # ==================================================================

                
        monthly_res = []

        for (y,m), group in pred.groupby(["year","month"]):
            group = group.sort_values("pred_signal")

            short  = group.head(J)       # bottom J stocks
            long   = group.tail(K)       # top K stocks

            long_r  = long["stock_ret"].mean()
            short_r = short["stock_ret"].mean()
            ls_r    = long_r - short_r

            monthly_res.append(ls_r)

        monthly = pd.Series(monthly_res)

        # performance metrics
        sharpe = monthly.mean()/monthly.std()*np.sqrt(12)
        max_dd = (1+monthly).cumprod().cummax() - (1+monthly).cumprod()

        results.append([K,J,sharpe,monthly.mean(),monthly.std(),max_dd.max()])

# ================== RESULTS OUTPUT ==================
opt = pd.DataFrame(results, columns=["Long_K","Short_J","Sharpe","MeanRet","Vol","MaxDrawdown"])
opt = opt.sort_values("Sharpe",ascending=False)

print("\n========== OPTIMISATION SUMMARY (BEST → WORST) ==========")
print(opt.head(10).to_string(index=False))

print("\nBest configuration:")
print(opt.iloc[0])

predicted = pred.groupby(["year", "month"])["pred_signal"]
pred["rank"] = np.floor(
    predicted.transform(lambda s: s.rank())
    * 10
    / predicted.transform(lambda s: len(s) + 1)
)


pred = pred.sort_values(
    ["year", "month", "rank", "id"]
)  # sort stocks based on the rank
monthly_port = pred.groupby(["year", "month", "rank"]).apply(
    lambda df: pd.Series(np.average(df["stock_ret"], axis=0))
)  # calculate the realized return for each portfolio using realized stock returns, assume equal-weighted portfolios
monthly_port = monthly_port.unstack().dropna().reset_index()  # reshape the data
monthly_port.columns = ["year", "month"] + [
    "port_" + str(x) for x in range(1, 11)
]  # rename columns
monthly_port["port_11"] = (
    monthly_port["port_10"] - monthly_port["port_1"]
)  # port 11 is the long-short portfolio

# Calculate the Sharpe ratio for long-short Portfolio
# you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately
sharpe = (
    monthly_port["port_11"].mean()  # average return
    / monthly_port["port_11"].std()  # standard deviation of return, volatility
    * np.sqrt(12)  # annualized
)  # Sharpe ratio is annualized
print("\nSharpe Ratio:", sharpe)

# Calculate the cumulative return of the long-short Portfolio
returns = monthly_port["port_11"].copy()
returns = returns + 1
cumulative_returns = returns.cumprod() - 1

# Max one-month loss of the long-short Portfolio
max_1m_loss = monthly_port["port_11"].min()
print("Max 1-Month Loss:", max_1m_loss)

# Calculate Drawdown of the long-short Portfolio
monthly_port["log_port_11"] = np.log(
    monthly_port["port_11"] + 1
)  # calculate log returns
monthly_port["cumsum_log_port_11"] = monthly_port["log_port_11"].cumsum(
    axis=0
)  # calculate cumulative log returns
rolling_peak = monthly_port["cumsum_log_port_11"].cummax()
drawdowns = rolling_peak - monthly_port["cumsum_log_port_11"]
max_drawdown = drawdowns.max()
print("Maximum Drawdown:", max_drawdown)

# =========================
# FULL PERFORMANCE ANALYSIS
# =========================

import numpy as np
import pandas as pd

monthly = monthly_port["port_11"]              # long-short returns only

# ===== Basic Monthly Stats =====
mean_m = monthly.mean()
std_m  = monthly.std()
winrate = (monthly > 0).mean()

# ===== Annualised Metrics =====
annual_ret = (1 + monthly).prod()**(12/len(monthly)) - 1
annual_vol = std_m * np.sqrt(12)
sharpe = (mean_m/std_m) * np.sqrt(12)
sortino = (mean_m / monthly[monthly < 0].std()) * np.sqrt(12)

# ===== Drawdown Statistics =====
cumulative = (1+monthly).cumprod()
peak = cumulative.cummax()
drawdown = (cumulative / peak) - 1
max_dd = drawdown.min()

# recovery length: number of months to regain prior peak
recovery_length = drawdown.groupby((drawdown==0).cumsum()).cumcount().max()

# ===== Tail Risk =====
skew = monthly.skew()
kurt = monthly.kurtosis()

# ===== Calmar ratio =====
calmar = annual_ret / abs(max_dd) if max_dd!=0 else np.nan

# ==== Output neatly ====
print("\n=========== PERFORMANCE REPORT ===========")
print(f"Total Months Backtested       : {len(monthly)}")
print(f"Annualised Return             : {annual_ret:.4f}")
print(f"Annualised Volatility         : {annual_vol:.4f}")
print(f"Sharpe Ratio                  : {sharpe:.4f}")
print(f"Sortino Ratio                 : {sortino:.4f}")
print(f"Calmar Ratio                  : {calmar:.4f}")
print(f"Max Drawdown                  : {max_dd:.4f}")
print(f"Longest Drawdown Recovery     : {recovery_length} months")
print(f"Win rate (% positive months)  : {winrate*100:.2f}%")
print(f"Skewness                      : {skew:.4f}")
print(f"Kurtosis                      : {kurt:.4f}")
print("==========================================\n")

# Optional – export full equity curve
monthly_port["cum_return"] = cumulative
monthly_port.to_csv("/Users/hedyshi/Downloads/批量下载-Char_Description等4个文件/performance_output.csv", index=False)
print("📄 Results saved → performance_output.csv")

# Optional PLOT (uncomment to view visually)
# import matplotlib.pyplot as plt
# plt.plot(cumulative, label="Long-Short Portfolio")
# plt.title("Cumulative Return Curve")
# plt.xlabel("Months")
# plt.ylabel("Portfolio Value")
# plt.legend()
# plt.show()
