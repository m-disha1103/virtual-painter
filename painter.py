import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Camera
WIDTH, HEIGHT = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Canvas Setup
canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
canvas_history = []

def save_state(c):
    if len(canvas_history) > 20:
        canvas_history.pop(0)
    canvas_history.append(c.copy())


# Variables
xp, yp = 0, 0
drawColor = (255, 0, 255)
brush_thickness = 10     # scaled up default thickness for bigger screen
eraser_thickness = 60    # scaled up default thickness
smoothening = 7          # slightly smoother for larger resolution
is_shape_mode = False
is_filled_mode = False
two_hand_drawing = False
shape_drawing = False
points = []
drawing = False
show_saved_msg_until = 0

# Colors BGR
colors = [
    (255, 0, 255),  # Pink
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (0, 165, 255),  # Orange
    (255, 255, 255),# White
    (255, 255, 0),  # Cyan
    (128, 0, 128),  # Purple
    (15, 15, 15),   # Black (Use 15,15,15 to avoid transparency threshold conflict)
    (0, 0, 0)       # Eraser
]

def fingers_up(lmList):
    fingers = []
    # Thumb robust check using relative distance to pinky base
    dist_tip = abs(lmList[4][0] - lmList[17][0])
    dist_ip = abs(lmList[3][0] - lmList[17][0])
    if dist_tip > dist_ip:
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

def draw_ui(img, drawColor, brush_thickness, eraser_thickness, is_shape_mode, is_filled_mode):
    overlay = img.copy()
    
    # Top Panel
    cv2.rectangle(overlay, (5, 5), (WIDTH - 5, 95), (25, 25, 25), cv2.FILLED)
    
    # Left Side: Colors
    for i, color in enumerate(colors):
        x1 = i * 85 + 20
        y1 = 15
        x2 = x1 + 65
        y2 = 80
        
        # Selection highlight
        if drawColor == color:
            cv2.rectangle(overlay, (x1-5, y1-5), (x2+5, y2+5), (255, 255, 255), cv2.FILLED)
            
        if color == (0, 0, 0): # Eraser
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 50), cv2.FILLED)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)

    # Bottom Panel for tools
    cv2.rectangle(overlay, (5, HEIGHT - 75), (WIDTH - 5, HEIGHT - 5), (25, 25, 25), cv2.FILLED)
    
    # Clear All Button Header
    cv2.rectangle(overlay, (20, HEIGHT - 65), (140, HEIGHT - 15), (40, 40, 200), cv2.FILLED)
    
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw solid UI shapes and text after blending
    for i, color in enumerate(colors):
        x1 = i * 85 + 20
        y1 = 15
        if color == (0,0,0):
            cv2.putText(img, "ERASE", (x1+5, y1+42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    cv2.putText(img, "CLEAR", (45, HEIGHT - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Right Side: Info Panel (Top)
    info_x = WIDTH - 220
    cv2.putText(img, f"Brush: {brush_thickness} px", (info_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(img, f"Eraser: {eraser_thickness} px", (info_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    shape_color = (100, 255, 100) if is_shape_mode else (150, 150, 150)
    cv2.putText(img, f"Snap: {'ON (M)' if is_shape_mode else 'OFF (M)'}", (info_x, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, shape_color, 2)
    
    fill_color = (100, 255, 100) if is_filled_mode else (150, 150, 150)
    cv2.putText(img, f"Fill: {'ON (F)' if is_filled_mode else 'OFF (F)'}", (info_x, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fill_color, 2)

    # Instructions Bottom
    instructions = "[U] Undo | [S] Save | [F] Fill Shape | [M] Shape Snap | Thumb+Finger=More Shapes"
    cv2.putText(img, instructions, (160, HEIGHT - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)

    return img

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Resize image just in case camera does not support exactly 1280x720 smoothly
    img = cv2.resize(img, (WIDTH, HEIGHT))
        
    img = cv2.flip(img, 1)
    img = draw_ui(img, drawColor, brush_thickness, eraser_thickness, is_shape_mode, is_filled_mode)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        lmLists = []
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append((cx, cy))
            lmLists.append(lmList)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # -------------------------------------------------------------
        # Two Hand Shape Mode
        # -------------------------------------------------------------
        if num_hands == 2:
            fingers1 = fingers_up(lmLists[0])
            fingers2 = fingers_up(lmLists[1])
            x1_1, y1_1 = lmLists[0][8]
            x1_2, y1_2 = lmLists[1][8]

            if fingers1[1] and not fingers1[2] and fingers2[1] and not fingers2[2] and drawColor != (0,0,0):
                if not two_hand_drawing:
                    two_hand_drawing = True
                    save_state(canvas)
                
                active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                thickness = -1 if is_filled_mode else brush_thickness * 2
                
                cv2.circle(img, (x1_1, y1_1), brush_thickness, drawColor, 2)
                cv2.circle(img, (x1_2, y1_2), brush_thickness, drawColor, 2)
                cv2.rectangle(active_canvas, (x1_1, y1_1), (x1_2, y1_2), drawColor, thickness)
            elif two_hand_drawing:
                # Commit shape when hands are dropped / fingers are changed
                two_hand_drawing = False
                canvas = cv2.bitwise_or(canvas, active_canvas)
                active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                xp, yp = 0, 0
                drawing = False
                points = []

        # -------------------------------------------------------------
        # Single Hand / Normal Tool Mode
        # -------------------------------------------------------------
        if not two_hand_drawing:
            for lmList in lmLists:
                fingers = fingers_up(lmList)
                xt, yt = lmList[4]
                xi, yi = lmList[8]
                x2, y2 = lmList[12]
                
                active_fingers = fingers[1:]
                active_fingers_count = sum(active_fingers)

                # Draw tool indicator
                if active_fingers_count >= 1:
                    if drawColor != (0,0,0):
                        cv2.circle(img, (xi, yi), brush_thickness, drawColor, 2)
                    else:
                        cv2.circle(img, (xi, yi), eraser_thickness, (100,100,100), 2)

                # -------------------------------------------------------------
                # Thumb + Finger Shape Mode
                # -------------------------------------------------------------
                if fingers[0] == 1 and active_fingers_count == 1 and drawColor != (0,0,0):
                    if drawing: drawing = False
                    if not shape_drawing:
                        shape_drawing = True
                        save_state(canvas)
                        xp, yp = 0, 0
                    
                    active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    thickness = -1 if is_filled_mode else brush_thickness * 2
                    
                    if fingers[1] == 1: # Thumb + Index = Circle
                        r = int(math.hypot(xi - xt, yi - yt))
                        cv2.circle(active_canvas, (xt, yt), r, drawColor, thickness)
                        cv2.line(img, (xt, yt), (xi, yi), drawColor, 2)
                        
                    elif fingers[2] == 1: # Thumb + Middle = Rectangle
                        cv2.rectangle(active_canvas, (xt, yt), (x2, y2), drawColor, thickness)
                        cv2.rectangle(img, (xt, yt), (x2, y2), drawColor, 2)

                    elif fingers[3] == 1: # Thumb + Ring = Line
                        xr, yr = lmList[16]
                        cv2.line(active_canvas, (xt, yt), (xr, yr), drawColor, brush_thickness * 2)
                        cv2.line(img, (xt, yt), (xr, yr), drawColor, 2)
                        
                    elif fingers[4] == 1: # Thumb + Pinky = Ellipse
                        xpky, ypky = lmList[20]
                        ax = max(5, abs(xpky - xt))
                        ay = max(5, abs(ypky - yt))
                        cv2.ellipse(active_canvas, (xt, yt), (ax, ay), 0, 0, 360, drawColor, thickness)
                        cv2.line(img, (xt, yt), (xpky, ypky), drawColor, 2)
                        
                else:
                    if shape_drawing:
                        shape_drawing = False
                        canvas = cv2.bitwise_or(canvas, active_canvas)
                        active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

                    # -------------------------------------------------------------
                    # Selection mode (Two fingers up, no thumb)
                    # -------------------------------------------------------------
                    if fingers[1] == 1 and fingers[2] == 1 and active_fingers_count == 2:
                        xp, yp = 0, 0
                        
                        if drawing:
                            drawing = False
                            if is_shape_mode and len(points) > 15 and drawColor != (0,0,0):
                                pts = np.array(points, np.int32).reshape((-1,1,2))
                                epsilon = 0.04 * cv2.arcLength(pts, True)
                                approx = cv2.approxPolyDP(pts, epsilon, True)
                                
                                x_min, y_min = np.min(pts[:,0,0]), np.min(pts[:,0,1])
                                x_max, y_max = np.max(pts[:,0,0]), np.max(pts[:,0,1])
                                width_rect, height_rect = x_max - x_min, y_max - y_min
                                
                                thickness = -1 if is_filled_mode else brush_thickness * 2
                                
                                # Draw perfect shape directly on canvas
                                if len(approx) == 3:
                                    if is_filled_mode:
                                        cv2.fillPoly(canvas, [approx], drawColor)
                                    else:
                                        cv2.polylines(canvas, [approx], True, drawColor, brush_thickness * 2)
                                elif len(approx) == 4:
                                    cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), drawColor, thickness)
                                else:
                                    center = (x_min + width_rect//2, y_min + height_rect//2)
                                    radius = int((width_rect + height_rect) / 4)
                                    cv2.circle(canvas, center, radius, drawColor, thickness)
                                
                                active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                            else:
                                # Transfer active to permanent
                                canvas = cv2.bitwise_or(canvas, active_canvas)
                                active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                                
                            points = []

                        # Check for clicks on UI
                        if yi < 95:
                            for i, color in enumerate(colors):
                                x_start = i * 85 + 20
                                x_end = x_start + 65
                                if x_start < xi < x_end:
                                    drawColor = color
                                    
                        if 20 < xi < 140 and HEIGHT - 65 < yi < HEIGHT - 15:
                            if np.any(canvas):
                                save_state(canvas)
                                canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

                    # -------------------------------------------------------------
                    # Drawing mode (Index finger up only)
                    # -------------------------------------------------------------
                    elif fingers[1] == 1 and active_fingers_count == 1:
                        if not drawing:
                            drawing = True
                            save_state(canvas)
                            points = []
                            xp, yp = xi, yi

                        # Smooth movement
                        xi = xp + (xi - xp) // smoothening
                        yi = yp + (yi - yp) // smoothening

                        if xp == 0 and yp == 0:
                            xp, yp = xi, yi

                        thickness = eraser_thickness if drawColor == (0,0,0) else brush_thickness

                        if drawColor == (0,0,0):
                            # Eraser applies directly to canvas immediately
                            cv2.line(canvas, (xp, yp), (xi, yi), (0,0,0), thickness * 2)
                            cv2.circle(canvas, (xi, yi), thickness, (0,0,0), cv2.FILLED)
                        else:
                            cv2.line(active_canvas, (xp, yp), (xi, yi), drawColor, thickness * 2)
                            cv2.circle(active_canvas, (xi, yi), thickness, drawColor, cv2.FILLED)
                            points.append((xi, yi))

                        xp, yp = xi, yi
    else:
        if two_hand_drawing:
            two_hand_drawing = False
            canvas = cv2.bitwise_or(canvas, active_canvas)
            active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            xp, yp = 0, 0
            drawing = False
            points = []
        if shape_drawing:
            shape_drawing = False
            canvas = cv2.bitwise_or(canvas, active_canvas)
            active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            
    # Combine Active Canvas into Permanent Canvas for display
    combined_canvas = cv2.bitwise_or(canvas, active_canvas)

    # Merge canvas with live image
    imgGray = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img_bg = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img_bg, combined_canvas)
    
    # Save Notification overlay
    if time.time() < show_saved_msg_until:
        cv2.putText(img, "SAVED SUCCESSFULLY!", (WIDTH//2 - 200, HEIGHT//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)

    cv2.namedWindow("AI Virtual Painter Pro", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Virtual Painter Pro", WIDTH, HEIGHT)
    cv2.imshow("AI Virtual Painter Pro", img)

    key = cv2.waitKey(1) & 0xFF

    # Shortcuts
    if key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, combined_canvas)
        print("Saved:", filename)
        show_saved_msg_until = time.time() + 2
        
    if key == ord('u'):
        if len(canvas_history) > 0:
            canvas = canvas_history.pop()
            active_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            drawing = False
            points = []

    if key == ord('m'):
        is_shape_mode = not is_shape_mode

    if key == ord('f'):
        is_filled_mode = not is_filled_mode

    if key == ord(']'):
        brush_thickness += 5
    if key == ord('['):
        brush_thickness = max(5, brush_thickness - 5)

    if key == ord('+') or key == ord('='):
        eraser_thickness += 10
    if key == ord('-'):
        eraser_thickness = max(10, eraser_thickness - 10)

    if key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()