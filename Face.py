import cv2
import numpy as np
from mtcnn import MTCNN
import mediapipe as mp
from skimage.feature import local_binary_pattern
import collections
import math
from time import time
from collections import deque

ear_history = deque(maxlen=3)
outputFolderPath = 'CroppedFace/Faces'

#--- Initialize models
detector = MTCNN()
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#--- Eyelid landmarks for blink detection
LEFT_EYE = [33, 160, 158, 133, 153, 144]         #--- P1, P2, P3, P4, P5, P6   
RIGHT_EYE = [362, 385, 387, 263, 373, 380]       #--- [left corner, left-top, right-top, right corner, right-bottom, left-bottom]
EAR_THRESHOLD = 0.22  #--- slightly relaxed      #--- change consec_frame value [1,3] & ear_threshold [0.20, 0.28] if auto blink count 
CONSEC_FRAMES = 2
BLINK_WINDOW = 20  #--- look for blink in last 20 frames

#--- Texture analysis parameters
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LAP_VAR_THRESHOLD = 100.0  #--- lowered threshold

#--- Rolling blink storage
blink_history = collections.deque(maxlen=BLINK_WINDOW)

blink_count = 0
frames_below = 0

def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) or 1
    return (A + B) / (2.0 * C)

def compute_texture(gray_roi):
    #--- gray_roi shouldn't be empty else frame get close if no face is detected
    if gray_roi is None or gray_roi.size == 0:
        print("Warning: Empty or invalid ROI passed to compute_texture.")
        return 0.0, 0.0  #--- Or other default safe values
    
    lbp = local_binary_pattern(gray_roi, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_POINTS+3),
                           range=(0, LBP_N_POINTS+2))
    lbp_mean = hist.astype("float") / (hist.sum() + 1e-7)
    lap_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    return lbp_mean.mean(), lap_var

cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Failed to grab frame.')
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    real_face = False
    avg_ear = lbp_score = lap_var = 0

    faces = detector.detect_faces(rgb)
    if faces:
        x, y, width, height = faces[0]['box']
        #--- Expand/shrink face box by padding
        padding = 50  #--- Increase or decrease this value as needed
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        x2 = min(x + width + 2 * padding, frame.shape[1])
        y2 = min(y + height + 2 * padding, frame.shape[0])

        #x, y = max(x,0), max(y,0)
        #x2, y2 = x+width, y+height
        cv2.rectangle(frame, (x,y), (x2,y2), (255,0,0), 2)

        mesh_res = face_mesh.process(rgb)
        if mesh_res.multi_face_landmarks:
            lm = mesh_res.multi_face_landmarks[0].landmark
            
            #--- Calculate eye distance (outer corners)
            left_corner = lm[33]
            right_corner = lm[362]
            
            #----  Convert to pixel coordinates by multiply normalized coordinates by the image width(w) and height(h) to get pixel positions:--
            x1_eye = int(left_corner.x * w)    #---  'left_eye': [279, 243], 'right_eye': [371, 245],: left_corner.x = left pixel(x)/width
            y1_eye = int(left_corner.y * h)    #---     width = 640 height = 480                       left_corner.y = left pixel(y)/height
            x2_eye = int(right_corner.x * w)   #---         (May vary)                                 right_corner.x = right pixel(x)/w
            y2_eye = int(right_corner.y * h)   #---     These are normalized value                     right_corner.y = right pixel(y)/h
            
            #---- Applying Euclidean distance which gives the straight-line distance between the two points:---

            eye_distance = int(np.linalg.norm([x2_eye - x1_eye, y2_eye - y1_eye]))


            left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2
            ear_history.append(avg_ear)
            #--- Use smoothed EAR
            smoothed_ear = sum(ear_history) / len(ear_history)

            if smoothed_ear < EAR_THRESHOLD:
                frames_below += 1
            else:
                if frames_below >= CONSEC_FRAMES:
                    blink_count += 1
                    blink_history.append(True)
                frames_below = 0

            xs = [p.x for p in lm]; ys = [p.y for p in lm]      #--- This extracts all x coordinates (normalized between 0.0 and 1.0) from the facial landmarks lm
            x1, x2 = int(min(xs)*w), int(max(xs)*w)             #--- (x1, y1) is the top-left corner of the face bounding box--- min(xs)*w → leftmost point of the face (in pixels)
            y1, y2 = int(min(ys)*h), int(max(ys)*h)             #--- (x2, y2) is the bottom-right corner--- max(xs)*w → rightmost point of the face

            if x2-x1>30 and y2-y1>30:                           #--- Ensures the box is not too small (avoids noise or false detection).
                roi_gray = gray[y1:y2, x1:x2]                   #--- Extracts the Region of Interest (ROI) from the grayscale image — just the face area.
                lbp_score, lap_var = compute_texture(roi_gray)  #--- lbp_score: Local Binary Pattern average (texture uniformity)&lap_var: Laplacian variance (focus/blurriness indicator)
            
            #--- Decide real vs fake
            blink_recent = any(blink_history)
            texture_ok = lap_var > LAP_VAR_THRESHOLD
            real_face = blink_recent and texture_ok
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        

    status = "REAL" if real_face else "FAKE"
    color = (0,255,0) if real_face else (0,0,255)
    
    if real_face:       #--- if face found real then it show these text else show no face is detected ---

        cv2.putText(frame, f"EAR:{avg_ear:.2f} Blinks:{blink_count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
    #cv2.putText(frame, f"LBP:{lbp_score:.3f} LapVar:{lap_var:.1f}", (10,60),
                #cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,255),2)
        cv2.putText(frame, f"Face:{status}", (10,60),                 #--- if LBP & LapVar visible then Face text coord: (10,90)
                cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    
        confidence = faces[0]['confidence']  #--- Value between 0 and 1
    
        if faces:
            face_info = faces[0]
            confidence = face_info.get('confidence', 0.0)
        else:
            confidence = 0.0

        confidence_pct = confidence * 100
    
        cv2.putText(frame, f"Confidence: {confidence_pct:.1f}%", (10, 90),       #--- (10,120)
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"EyeDist: {eye_distance}px", (10, 120),              #--- (10,150)
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            
    #--- Resize frame(window) by a scaling factor (1.2x)
    scale = 1
    resized_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Live Real vs Fake Detection", resized_frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('c'):
        if faces:
            x, y, width, height = faces[0]['box']
            #--- Expand/shrink face box by padding
            padding = 50  #--- Increase or decrease this value as needed

            x, y, width, height = faces[0]['box']
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            x2 = min(x + width + 2 * padding, frame.shape[1])
            y2 = min(y + height + 2 * padding, frame.shape[0])

            #x, y = max(x, 0), max(y, 0)
            #x2, y2 = x + width, y + height
            face_crop = frame[y:y2, x:x2]

        if face_crop.size > 0:
            timeNow = time()
            timeNow = str(timeNow).split('.')
            timeNow = timeNow[0] + timeNow[1]
            #cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg', img) #--- Its save the whole face ---
            cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg', face_crop) #--- Cropped face
            print('Cropped face has been saved as "Face_Cropped.jpg"')
        else:
            print('Failed to crop face: empty region.')

    #--- Key to reset blink count ---
    elif key == ord('r'):
        blink_count = 0
        blink_history.clear()
        print("Blink Count has been reset.")

    #--- ASCII 27: Esc button(Exit from interface) ---        
    elif key == 27:
        break
    

cap.release()
cv2.destroyAllWindows()