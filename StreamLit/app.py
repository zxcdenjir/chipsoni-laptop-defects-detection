import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# Загрузка модели YOLO
model = YOLO('best.pt')

# Словарь для сопоставления классов
class_names = {
    0: 'broken_button',
    1: 'broken_pixel',
    2: 'chip',
    3: 'disassembled_laptop',
    4: 'lock',
    5: 'missing_button',
    6: 'missing_screw',
    7: 'scratch'
}

# Функция для обработки изображения
def process_image(image):
    # Преобразуем изображение в формат, который понимает модель
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(image)  # Получаем результаты детекции

    detections = []  # Список для хранения результатов детекции

    # Рисуем bounding box на изображении
    for result in results:
        for box in result.boxes:  # box - это объект, содержащий координаты
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты
            conf = box.conf[0]  # Уверенность
            cls = int(box.cls[0])  # Класс

            # Получаем название класса из словаря
            class_name = class_names.get(cls, 'Unknown')  # Если класс не найден, показываем 'Unknown'

            # Добавляем детекцию в список
            detections.append({
                'Class': class_name,
                'Confidence': conf,
                'Coordinates': (x1, y1, x2, y2)
            })

            # Рисуем прямоугольник и текст на изображении
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f'{class_name} ({conf:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections  # Возвращаем изображение в RGB и детекции

# Основная функция приложения
def main():
    st.title("Обнаружение дефектов на ноутбуках")
    
    # Ввод серийного номера
    serial_number = st.text_input("Введите серийный номер и нажмите enter:", "")
    
    # Загрузка изображений
    uploaded_files = st.file_uploader("Загрузить фото...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files and serial_number:  # Проверяем, что загружены файлы и введен серийный номер
        # Кнопка для начала детекции
        if st.button("Начать детекцию"):
            # Создаем DataFrame для отчета
            report_data = []
            processed_images = []

            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                
                # Обработка изображения
                processed_image, detections = process_image(image)

                # Добавить обработанное изображение в список
                processed_images.append(processed_image)

                # Добавляем данные в отчет
                for detection in detections:
                    report_data.append({
                        'Serial Number': serial_number,  # Добавляем серийный номер
                        'File Name': uploaded_file.name,
                        'Class': detection['Class'],
                        'Confidence': detection['Confidence'],
                        'Coordinates': detection['Coordinates']
                    })

            # Показать обработанные изображения
            for i, processed_image in enumerate(processed_images):
                st.image(processed_image, caption=f'Processed Image {i + 1}', use_column_width=True)

            # Создание DataFrame отчета
            report_df = pd.DataFrame(report_data)

            # Показать отчет
            st.subheader("Отчет по детекциям")
            st.dataframe(report_df)

            # Кнопка для скачивания отчета
            csv = report_df.to_csv(index=False)
            st.download_button("Скачать отчет", csv, "detections_report.csv", "text/csv")

if __name__ == "__main__":
    main()