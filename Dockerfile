# Python 3.8 tabanlı resmi image'ı kullan
FROM python:3.10-slim

# Çalışma dizinini belirle
WORKDIR /app

# Gerekli sistem paketlerini yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python paketlerini yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Sadece model için volume tanımla
VOLUME ["/app/models"]

# Uygulamayı çalıştır
CMD ["python", "app.py"]