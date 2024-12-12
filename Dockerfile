# Используем базовый образ с Python
FROM python:3.9-slim

# Устанавливаем необходимые системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем файлы проекта
COPY . .

# Указываем порт, на котором работает приложение
EXPOSE 8501

# Запуск Streamlit приложения
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
