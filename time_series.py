import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# ==========================================
# 1. 데이터 수집 및 정제
# ==========================================
file_path = "AirPassengers.csv"

if not os.path.exists(file_path):
    print(
        f"Error: '{file_path}' 파일이 없습니다. 파일 위치를 확인해 주세요."
    )
    exit()

df = pd.read_csv(file_path)

# 컬럼명 공백 제거 및 이름 통일
df.columns = df.columns.str.strip()
for col in df.columns:
    if "Passenger" in col or "passenger" in col:
        df = df.rename(columns={col: "Passengers"})
    if "Month" in col or "date" in col or "Date" in col:
        df = df.rename(columns={col: "Month"})

df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

# 결측치 보간 및 이상치 IQR 조정
if df["Passengers"].isnull().sum() > 0:
    df["Passengers"] = df["Passengers"].interpolate(method="linear")

Q1 = df["Passengers"].quantile(0.25)
Q3 = df["Passengers"].quantile(0.75)
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

df["Passengers"] = np.where(
    df["Passengers"] > upper_bound, upper_bound, df["Passengers"]
)
df["Passengers"] = np.where(
    df["Passengers"] < lower_bound, lower_bound, df["Passengers"]
)

# 파생 변수 생성
df["MA_12"] = df["Passengers"].rolling(window=12).mean()
df["Month_Num"] = df["Month"].dt.month

# ==========================================
# 2. 시각화 그래프 3종 생성 및 저장
# ==========================================
sns.set_theme(style="whitegrid")

# [그래프 1] 기본 추세선 및 12개월 이동평균
plt.figure(figsize=(12, 5))
plt.plot(
    df["Month"],
    df["Passengers"],
    label="Original Passengers",
    color="#2b5c8f",
    linewidth=1.5,
)
plt.plot(
    df["Month"],
    df["MA_12"],
    label="12-Month Moving Average (Trend)",
    color="#e74c3c",
    linewidth=2.5,
)
plt.title(
    "1. Air Passengers Trend Analysis (1949 - 1960)", fontsize=13, pad=10
)
plt.xlabel("Date", fontsize=10)
plt.ylabel("Passengers Count", fontsize=10)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("passenger_trend.png")
plt.close()

# [그래프 2] 월별 분포 박스플롯 (계절성 확인)
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=df,
    x="Month_Num",
    y="Passengers",
    palette="coolwarm",
    hue="Month_Num",
    legend=False,
)
plt.title(
    "2. Air Passengers Seasonality: Monthly Distribution", fontsize=13, pad=10
)
plt.xlabel("Month", fontsize=10)
plt.ylabel("Passengers Count", fontsize=10)
plt.xticks(
    range(0, 12),
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
)
plt.tight_layout()
plt.savefig("monthly_boxplot.png")
plt.close()

# [그래프 3] 시계열 분해 (Trend, Seasonality, Residual)
ts_data = df.set_index("Month")["Passengers"]
result = seasonal_decompose(ts_data, model="multiplicative", period=12)

fig = result.plot()
fig.set_size_inches(12, 8)
fig.suptitle(
    "3. Time Series Decomposition (Trend / Seasonality / Residual)",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.savefig("ts_decomposition.png")
plt.close()

# 결과 저장 완료 메시지
print("=" * 50)
print("[분석 완료] 아래 3개 그래프 파일이 생성되었습니다:")
print("1. passenger_trend.png (기본 추세 그래프)")
print("2. monthly_boxplot.png (월별 성수기/비수기 박스플롯)")
print("3. ts_decomposition.png (시계열 요인 분해 그래프)")
print("=" * 50)

# 정제 데이터 CSV 저장
df.to_csv("AirPassengers_cleaned.csv", index=False)