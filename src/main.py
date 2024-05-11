import streamlit as st
import cv2
import math
from ultralytics import YOLO

import tkinter as tk
from tkinter import filedialog


st.sidebar.title('Настройки модели')

st.title('Детекция дронов')
sample_img = cv2.imread('logo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')

cap = None

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)

if st.sidebar.button('Загрузка модели'):
    st.session_state['path_model'] = filedialog.askopenfilename(master=root)
    st.session_state['model'] = YOLO(st.session_state['path_model'])

path_model_file = st.sidebar.text_input('Путь до модели:', st.session_state.get('path_model', None))

model = st.session_state.get('model', None)
if model:
    # Load Class names
    class_labels = model.names

    # Confidence
    confidence = st.sidebar.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

    # Draw thickness
    draw_thick = st.sidebar.slider(
        'Draw Thickness:', min_value=1,
        max_value=20, value=3
    )

    # Web-cam
    cam_options = st.sidebar.selectbox('Webcam Channel',
                                    ('Select Channel', '0', '1', '2', '3'))

    if cam_options != 'Select Channel':
        cap = cv2.VideoCapture(int(cam_options))


if cap:
    btn_stop = st.button('Остановить детекцию')
    while True:

        if btn_stop:
            break

        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", class_labels[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, class_labels[cls], org, font, fontScale, color, thickness)

        FRAME_WINDOW.image(img, channels='BGR')

    cap.release()
    cv2.destroyAllWindows()
    cap = None
