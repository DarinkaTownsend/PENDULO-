import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Parámetros del péndulo
g = 9.81  # Aceleración debida a la gravedad (m/s^2)
length = 200  # Longitud del péndulo en píxeles
theta = np.pi / 4  # Ángulo inicial del péndulo (en radianes)
omega = 0  # Velocidad angular inicial

# Cargar imagen de la pelota
ball_img = cv2.imread('ball.png', cv2.IMREAD_UNCHANGED)

# Obtener dimensiones de la imagen de la pelota
ball_height, ball_width, _ = ball_img.shape

# Función para calcular la nueva posición de la pelota
def calculate_ball_position(center_x, center_y, theta):
    x = int(center_x + length * np.sin(theta))
    y = int(center_y + length * np.cos(theta))
    return x, y

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener las posiciones de los dedos
            x_index = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
            y_index = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
            x_middle = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
            y_middle = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)

            # Dibujar la posición de los dedos en la imagen
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x_middle, y_middle), 10, (0, 255, 0), -1)

            # Calcular el centro de la mano
            center_x = (x_index + x_middle) // 2
            center_y = (y_index + y_middle) // 2

            # Si los dos dedos están levantados
            if abs(x_index - x_middle) < 50 and abs(y_index - y_middle) < 50:
                # Simular el péndulo
                alpha = -(g / length) * np.sin(theta)
                omega += alpha * 0.01  # dt = 0.01
                theta += omega * 0.01

                # Calcular la posición de la pelota
                ball_x, ball_y = calculate_ball_position(center_x, center_y, theta)

                # Dibujar la pelota
                overlay = frame.copy()
                overlay[ball_y - ball_height // 2:ball_y + ball_height // 2,
                        ball_x - ball_width // 2:ball_x + ball_width // 2] = ball_img
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

                # Dibujar la cuerda del péndulo
                cv2.line(frame, (center_x, center_y), (ball_x, ball_y), (255, 0, 0), 2)

    cv2.imshow('Pendulum Simulation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
