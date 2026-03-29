import cv2
import mediapipe as mp
import numpy as np
import time

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Canvas
canvas = np.zeros((480, 640, 3), np.uint8)

# Variables
xp, yp = 0, 0
drawColor = (255, 0, 255)

brush_thickness = 5
eraser_thickness = 20
smoothening = 7

# Colors
colors = [
    (255, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 0, 0)
]
labels = ["Pink", "Blue", "Green", "Red", "Eraser"]

points = []

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

    # Draw palette
    for i, color in enumerate(colors):
        x1 = i * 128
        x2 = (i + 1) * 128
        cv2.rectangle(img, (x1, 0), (x2, 80), color, cv2.FILLED)
        cv2.putText(img, labels[i], (x1 + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Clear button
    cv2.rectangle(img, (0, 400), (120, 480), (50,50,50), cv2.FILLED)
    cv2.putText(img, "CLEAR", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Show sizes
    cv2.putText(img, f"Brush: {brush_thickness}", (350, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(img, f"Eraser: {eraser_thickness}", (350, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append((cx, cy))

            if lmList:
                fingers = fingers_up(lmList)

                x1, y1 = lmList[8]
                x2, y2 = lmList[12]

                # Selection mode
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0

                    if y1 < 80:
                        drawColor = colors[x1 // 128]

                    if 0 < x1 < 120 and 400 < y1 < 480:
                        canvas = np.zeros((480,640,3), np.uint8)

                    # Shape detection
                    if len(points) > 10:
                        pts = np.array(points, np.int32)
                        pts = pts.reshape((-1,1,2))

                        approx = cv2.approxPolyDP(
                            pts, 0.02*cv2.arcLength(pts, True), True)

                        if len(approx) == 3:
                            print("Triangle")
                        elif len(approx) == 4:
                            print("Rectangle")
                        elif len(approx) > 6:
                            print("Circle")

                    points = []

                # Drawing mode
                elif fingers[1] and not fingers[2]:

                    # Smooth movement
                    x1 = xp + (x1 - xp) // smoothening
                    y1 = yp + (y1 - yp) // smoothening

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    thickness = eraser_thickness if drawColor == (0,0,0) else brush_thickness

                    # 🔥 Smooth continuous stroke
                    cv2.line(canvas, (xp, yp), (x1, y1), drawColor, thickness * 2)
                    cv2.circle(canvas, (x1, y1), thickness, drawColor, cv2.FILLED)

                    points.append((x1, y1))

                    xp, yp = x1, y1

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Merge canvas
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("AI Virtual Painter", img)

    key = cv2.waitKey(1)

    # Save
    if key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print("Saved:", filename)

    # Brush size
    if key == ord(']'):
        brush_thickness += 2
    if key == ord('['):
        brush_thickness = max(1, brush_thickness - 2)

    # Eraser size
    if key == ord('+'):
        eraser_thickness += 5
    if key == ord('-'):
        eraser_thickness = max(5, eraser_thickness - 5)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()