import requests
from bs4 import BeautifulSoup

url = "https://example.com"

headers = {
	"User-Agent": "Mozilla/5.0"
}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")

titles = soup.select("h2")

for title in titles:
	print(title.get_text(strip=True))

import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("AirPassengers.csv")

# 상위 5개 행 출력하기
print(df.head())