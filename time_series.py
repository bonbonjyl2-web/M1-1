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

# 정제 전 통계치 기록
missing_before = df["Passengers"].isnull().sum()

# 결측치 보간
if missing_before > 0:
    df["Passengers"] = df["Passengers"].interpolate(method="linear")

# 이상치 IQR 계산 및 클리핑
Q1 = df["Passengers"].quantile(0.25)
Q3 = df["Passengers"].quantile(0.75)
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

outlier_count = ((df["Passengers"] < lower_bound) | (df["Passengers"] > upper_bound)).sum()

df["Passengers"] = np.where(
    df["Passengers"] > upper_bound, upper_bound, df["Passengers"]
)
df["Passengers"] = np.where(
    df["Passengers"] < lower_bound, lower_bound, df["Passengers"]
)

# ==========================================
# 2. 파생 변수 생성 (기법 적용)
# ==========================================
df["MA_12"] = df["Passengers"].rolling(window=12).mean()
df["Pct_Change(%)"] = df["Passengers"].pct_change() * 100
df["Month_Num"] = df["Month"].dt.month
df["Year"] = df["Month"].dt.year

# ==========================================
# 3. 시계열 분해 및 분산 기여율 계산 (평가항목 #12)
# ==========================================
ts_data = df.set_index("Month")["Passengers"]
result = seasonal_decompose(ts_data, model="multiplicative", period=12)

# 로그 변환 후 분산 분해 계산
log_ts = np.log(ts_data)
log_decomp = seasonal_decompose(log_ts, model="additive", period=12)

var_total = np.var(log_ts)
var_trend = np.var(log_decomp.trend.dropna())
var_seasonal = np.var(log_decomp.seasonal)
var_resid = np.var(log_decomp.resid.dropna())

print("=" * 60)
print("[시계열 요소별 분산 기여율]")
print(f"Total Variance: {var_total:.4f}")
print(f"Trend Variance Share: {(var_trend/var_total)*100:.2f}%")
print(f"Seasonal Variance Share: {(var_seasonal/var_total)*100:.2f}%")
print(f"Residual Variance Share: {(var_resid/var_total)*100:.2f}%")
print("=" * 60)

# ==========================================
# 4. 시각화 그래프 3종 생성
# ==========================================
sns.set_theme(style="whitegrid")

# [그래프 1]
plt.figure(figsize=(12, 5))
plt.plot(df["Month"], df["Passengers"], label="Original Passengers", color="#2b5c8f", linewidth=1.5)
plt.plot(df["Month"], df["MA_12"], label="12-Month Moving Average (Trend)", color="#e74c3c", linewidth=2.5)
plt.title("1. Air Passengers Trend Analysis (1949 - 1960)", fontsize=13, pad=10)
plt.xlabel("Date", fontsize=10)
plt.ylabel("Passengers Count", fontsize=10)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("passenger_trend.png")
plt.close()

# [그래프 2]
plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x="Month_Num", y="Passengers", palette="coolwarm", hue="Month_Num", legend=False)
plt.title("2. Air Passengers Seasonality: Monthly Distribution", fontsize=13, pad=10)
plt.xlabel("Month", fontsize=10)
plt.ylabel("Passengers Count", fontsize=10)
plt.xticks(range(0, 12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.tight_layout()
plt.savefig("monthly_boxplot.png")
plt.close()

# [그래프 3] 시계열 분해 시각화 수정 버전
fig = result.plot()
fig.set_size_inches(12, 10)

# 1. 제목의 높이(y)를 살짝 낮추고 (기존 1.01 -> 0.98)
fig.suptitle("3. Time Series Decomposition (Trend / Seasonality / Residual)", 
             fontsize=13, y=0.98)

# 2. 잔차 축 범위 조정
fig.axes[3].set_ylim(0.95, 1.05)
fig.axes[3].axhline(1, color='red', linestyle='--', alpha=0.5)

# 3. tight_layout 실행 시 상단에 여백(rect)을 강제로 부여 (핵심!)
# rect=[왼쪽, 아래, 오른쪽, 위] 순서입니다. 상단(0.95)에 여백을 줍니다.
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig("ts_decomposition.png")
plt.close()
df.to_csv("AirPassengers_cleaned.csv", index=False)
print("\n[완료] 데이터 정제 및 시각화 저장 성공 (AirPassengers_cleaned.csv 생성됨)")

# ==========================================
# 5. 분석 리포트 및 인사이트 (평가항목 대응)
# ==========================================

print("\n" + "=" * 60)
print("[분석 결과 해석 - Observation & Interpretation]")
print("=" * 60)

# 관찰(Observation)
print("1. 관찰 (Observation):")
print(f"   - 전체적인 추세는 1949년부터 1960년까지 지속적으로 우상향함.")
print(f"   - 계절성 분산 기여율({(var_seasonal/var_total)*100:.2f}%)이 잔차({(var_resid/var_total)*100:.2f}%)보다 압도적으로 높음.")
print(f"   - 박스플롯 상 7, 8월의 승객 수가 연중 가장 높게 나타남.")

# 해석(Interpretation)
print("\n2. 해석 및 가설 (Interpretation):")
print("   - (가설 1) 항공 산업의 성장기: 전후 경제 성장과 항공 기술 발달로 인해 장기적 수요가 증가함.")
print("   - (가설 2) 하계 휴가 영향: 7-8월의 피크는 전형적인 여름 휴가 시즌의 영향으로 판단됨.")
print("   - (가설 3) 모델 적합성: 잔차 비중이 매우 낮으므로, 이 데이터는 추세와 계절성만으로도 높은 예측력을 가질 것임.")

# ==========================================
# 6. AI 사용 투명성 로그 (필수 항목)
# ==========================================
print("\n" + "=" * 60)
print("[AI 사용 투명성 로그]")
print("=" * 60)
ai_log = """
1. 사용 작업: 시계열 분산 기여율 계산 로직 구현 및 시각화 코드 최적화.
2. 사용 이유: 복잡한 통계적 수치 계산의 정확도를 높이고 시각화 라이브러리(Seaborn)의 최신 문법 적용을 위함.
3. 검증 방법: statsmodels의 분해 결과값과 수동 계산값을 비교 검증하고, 생성된 PNG 파일의 라벨 일치 여부 확인.
"""
print(ai_log.strip())
print("=" * 60)

# 리포트 텍스트 파일 저장
with open("analysis_report.txt", "w", encoding="utf-8") as f:
    f.write("[AirPassengers Analysis Report]\n")
    f.write(f"Trend Share: {(var_trend/var_total)*100:.2f}%\n")
    f.write(f"Seasonal Share: {(var_seasonal/var_total)*100:.2f}%\n")
    f.write(ai_log)
