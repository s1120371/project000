FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 複製其他所有檔案到 /app 目錄下
COPY . .

# Hugging Face Spaces 預設對外開放 7860 port
EXPOSE 7860

# 使用 Flask 啟動 (注意 port 要設 7860)
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
