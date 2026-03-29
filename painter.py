import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), np.uint8)

xp, yp = 0, 0
drawColor = (255, 0, 255)

# Color buttons
colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)]

def fingers_up(lmList):
    fingers = []

    if lmList[4][0] > lmList[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    tips = [8, 12, 16, 20]
    for tip in tips:
        if lmList[tip][1] < lmList[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Draw color palette UI
    for i, color in enumerate(colors):
        cv2.rectangle(img, (i*160, 0), ((i+1)*160, 80), color, cv2.FILLED)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            if lmList:
                fingers = fingers_up(lmList)

                x1, y1 = lmList[8]
                x2, y2 = lmList[12]

                # Selection mode
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0

                    # Check if selecting color
                    if y1 < 80:
                        drawColor = colors[x1 // 160]

                    cv2.rectangle(img, (x1, y1-20), (x2, y2+20), drawColor, cv2.FILLED)

                # Drawing mode
                elif fingers[1] and not fingers[2]:
                    cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    thickness = 20 if drawColor == (0, 0, 0) else 5
                    cv2.line(canvas, (xp, yp), (x1, y1), drawColor, thickness)
                    xp, yp = x1, y1

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Painter", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()