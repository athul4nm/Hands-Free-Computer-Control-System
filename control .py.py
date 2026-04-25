import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import cv2
import mediapipe as mp
import pyautogui
import time
import math

# ================= INITIALIZATION =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ================= SETTINGS (ADJUSTED) =================
SMOOTH = 0.10           # Speed kurachu, control kootan
MARGIN = 0.22           # Full screen bottom reach aakan
MOVE_THRESHOLD = 8      # Stability kootan
PINCH_THRESHOLD = 0.04  # Thumb & Index muttiyenkil drag thudangaan

# Dwell Timings
LEFT_DWELL = 0.6        
DOUBLE_DWELL = 1.3
KEYBOARD_COOLDOWN = 4

# ================= VARIABLES =================
smooth_x, smooth_y = None, None
last_move_time = time.time()
last_pos = (0, 0)
has_single_clicked = False
last_kb_time = 0
last_screenshot = 0
scroll_neutral_y = None
last_scroll = 0
is_dragging = False

def get_finger_states(hand):
    # Tip (8, 12, 16, 20) and PIP joints (6, 10, 14, 18)
    idx_up = hand.landmark[8].y < hand.landmark[6].y
    mid_up = hand.landmark[12].y < hand.landmark[10].y
    rng_up = hand.landmark[16].y < hand.landmark[14].y
    pky_up = hand.landmark[20].y < hand.landmark[18].y
    
    idx_down = hand.landmark[8].y > hand.landmark[6].y
    thumb_up = hand.landmark[4].x < hand.landmark[3].x 
    
    # Pinch distance (Thumb Tip 4 to Index Tip 8)
    t_tip = hand.landmark[4]
    i_tip = hand.landmark[8]
    dist = math.hypot(t_tip.x - i_tip.x, t_tip.y - i_tip.y)
    
    return thumb_up, idx_up, mid_up, rng_up, pky_up, idx_down, dist

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Gesture Mouse with DRAG Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    now = time.time()
    status = "IDLE"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        t_up, i_up, m_up, r_up, p_up, i_down, pinch_dist = get_finger_states(hand)

        # 1. SCREENSHOT (All fingers up)
        if i_up and m_up and r_up and p_up and now - last_screenshot > 3:
            pyautogui.screenshot(f"shot_{int(now)}.png")
            last_screenshot = now
            status = "SCREENSHOT SAVED"

        # 2. KEYBOARD (Thumb + Pinky only - Safe logic)
        elif t_up and p_up and not i_up and not m_up and not r_up:
            if now - last_kb_time > KEYBOARD_COOLDOWN:
                os.system("start osk")
                last_kb_time = now
                status = "KEYBOARD OPENED"

        # 3. SCROLL (Index + Middle up)
        elif i_up and m_up and not r_up:
            curr_y = hand.landmark[8].y
            if scroll_neutral_y is None: scroll_neutral_y = curr_y
            diff = curr_y - scroll_neutral_y
            if abs(diff) > 0.05 and now - last_scroll > 0.1:
                pyautogui.scroll(-80 if diff > 0 else 80)
                last_scroll = now
            status = "SCROLLING"

        # 4. CURSOR MOVE, DWELL CLICK & DRAG
        elif i_up and not m_up and not r_up:
            raw_x, raw_y = hand.landmark[8].x, hand.landmark[8].y
            
            # Mapping to screen
            nx = (raw_x - MARGIN) / (1 - 2 * MARGIN)
            ny = (raw_y - MARGIN) / (1 - 2 * MARGIN)
            nx, ny = max(0, min(1, nx)), max(0, min(1, ny))

            target_x, target_y = int(nx * SCREEN_W), int(ny * SCREEN_H)
            if smooth_x is None: smooth_x, smooth_y = target_x, target_y
            
            smooth_x += (target_x - smooth_x) * SMOOTH
            smooth_y += (target_y - smooth_y) * SMOOTH
            
            # Movement Check
            dist_moved = math.hypot(smooth_x - last_pos[0], smooth_y - last_pos[1])

            # --- DRAG LOGIC (PINCH) ---
            if pinch_dist < PINCH_THRESHOLD:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                status = "DRAGGING..."
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                    status = "DROPPED"
                
                # Normal Movement
                if dist_moved > MOVE_THRESHOLD:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    last_move_time = now
                    last_pos = (smooth_x, smooth_y)
                    has_single_clicked = False
                    status = "MOVING"
                else:
                    # Dwell Click Logic
                    dwell = now - last_move_time
                    if dwell > DOUBLE_DWELL:
                        pyautogui.doubleClick()
                        last_move_time = now
                        status = "DOUBLE CLICK"
                    elif dwell > LEFT_DWELL and not has_single_clicked:
                        pyautogui.click()
                        has_single_clicked = True
                        status = "LEFT CLICK"

        # 5. RIGHT CLICK (Middle up, Index down)
        elif m_up and i_down and not r_up:
            if now - last_scroll > 1.2:
                pyautogui.rightClick()
                last_scroll = now
                status = "RIGHT CLICK"
        
        else:
            # Cleanup when hand is removed or gesture changed
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
            scroll_neutral_y = None
            last_move_time = now

    cv2.putText(frame, f"STATUS: {status}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Perfect Assistive Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()