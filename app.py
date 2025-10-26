# import cv2
# import imutils
# from imutils import face_utils
# import dlib
# from scipy.spatial import distance
# from pygame import mixer

# mixer.init()
# mixer.music.load("music.wav")

# def eye_aspect_ratio(eye):
#     # ye points(0,1,2,3,4,5) ek aankh ke around 6 facial landmark points hote hain (from dlib’s 68 landmarks model)
#     A=distance.euclidean(eye[1],eye[5])
#     B=distance.euclidean(eye[2],eye[4])
#     C=distance.euclidean(eye[0],eye[3])
#     ear=(A+B)/(2.0*C)
#     return ear
#     # Jab aankh khuli hoti hai, vertical distances (A, B) bade hote hain → EAR high.
#     # Jab aankh band hoti hai, vertical distances ghate jaate hain → EAR low.




# (lStart,lEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
# (rStart,rEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
# # FACIAL_LANDMARKS_68_IDXS ek dictionary hai jo har face part ke naam ke saath uske start aur end indexes store karti hai.
# # Ye points fixed numbering pe hote hain — jaise
# # Left eye: points [42–47]  (lStart, lEnd) = (42, 48)
# # Right eye: points [36–41]  (rStart, rEnd) = (36, 42)




# detect=dlib.get_frontal_face_detector()
# # Dlib ke andar already ek pre-trained face detector model hota hai
# # (jo HOG + SVM algorithm pe based hai).
# # Ye line us model ko initialize karti hai — matlab ek “face detector object” banati hai.


# predict=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# # Is line me hum ek pretrained model load kar rahe hain jiska naam hai
# # shape_predictor_68_face_landmarks.dat.
# # Ye .dat file ek trained deep learning model hai.
# # Ye model dlib me diya gaya hai aur 68 specific facial points detect karta hai.

# thresh=0.21
# flag=0
# frame_check=20

# cap=cv2.VideoCapture(0)
# # //konse camera se video capture kr re hain


# while True:
    
#     ret,frame=cap.read()
#     # read bhi ek builtin function hai jo return krta hai 2 values
#     # 1. bool return krega if frame is available or not
#     # 2. image array vector

#     frame=imutils.resize(frame,width=450)

#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     # Frame ko grayscale me convert kar diya (kyunki face detection grayscale pe faster aur better hoti hai).
    
#     subjects=detect(gray,0)
#     # Ye face detector grayscale image me faces detect karta hai aur har face ke liye ek rectangle region return karta hai.
#     # (multiple faces bhi detect ho sakte hain → isliye for subject in subjects )
#     for subject in subjects:


#         shape= predict(gray,subject)
#         # Ye line ek face ke andar ke 68 landmark points detect karti hai (eyes, nose, lips, etc.).
#         # subject ek detected face rectangle hai jisme ye points nikalta hai.
        
#         shape=face_utils.shape_to_np(shape)
#         # Dlib ka output ek special format me hota hai (not directly NumPy array).
#         # Is line se usse NumPy array (x, y coordinates) me convert kar lete hain — taaki indexing aur slicing easy ho jaaye.
        
        
#         leftEye=shape[lStart:lEnd]
#         rightEye=shape[rStart:rEnd]
#         # Ye dono lines nikalti hain left aur right eye ke 6–6 landmark points,
#         # using the start–end indexes we got earlier.


#         leftEar=eye_aspect_ratio(leftEye)
#         rightEar=eye_aspect_ratio(rightEye)
#         ear=(leftEar+rightEar)/2.0

#         leftEyeHull=cv2.convexHull(leftEye)
#         rightEyeHull=cv2.convexHull(rightEye)

#         cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
#         cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
#         if ear<thresh:
#             flag+=1
#             print(flag)
#             if flag>=frame_check:
#                 cv2.putText(frame,"****ALERT****",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
#                 cv2.putText(frame,"****ALERT****",(10,325),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
#                 mixer.music.play()

#         else:
#             flag=0


#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(1) & 0xFF==ord("q"):
#         break

# cv2.destroyAllWindows()
# cap.release()
















































# from scipy.spatial import distance
# from imutils import face_utils
# from pygame import mixer
# import imutils
# import dlib
# import cv2
# import time
# import math
# import numpy as np
# import os
# import sys

# # Performance optimization
# # Set OpenCV parameters for better performance
# cv2.setUseOptimized(True)
# cv2.setNumThreads(4)  # Adjust based on your CPU

# # Initialize mixer with smaller buffer size for better responsiveness
# mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

# # Alert debounce timers to prevent too frequent alerts
# last_drowsy_alert = 0
# last_yawn_alert = 0
# last_tilt_alert = 0
# alert_cooldown = 5.0  # seconds between alerts of the same type

# # Try to load the sound with a more robust approach
# sound_file = "music.wav"
# if not os.path.exists(sound_file):
#     print(f"Warning: Sound file '{sound_file}' not found in current directory")
#     # Search in the directory of the script
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     alternative_path = os.path.join(script_dir, sound_file)
#     if os.path.exists(alternative_path):
#         sound_file = alternative_path
#         print(f"Found sound file at {sound_file}")
#     else:
#         print(f"Warning: Sound file not found at {alternative_path} either")

# try:
#     # Try to load the sound
#     mixer.music.load(sound_file)
#     # Lower volume to prevent audio driver issues
#     mixer.music.set_volume(0.7)
#     print(f"Successfully loaded sound from {sound_file}")
# except Exception as e:
#     print(f"Error loading sound: {e}")

# # Sound alert function
# def play_alert_sound():
#     try:
#         # Only play if not already playing
#         if not mixer.music.get_busy():
#             mixer.music.play(-1)  # Play in loop (-1) until stopped
#             print("Playing alert sound")
#     except Exception as e:
#         print(f"Error playing sound: {e}")

# def stop_alert_sound():
#     try:
#         if mixer.music.get_busy():
#             mixer.music.stop()
#             print("Stopped alert sound")
#     except Exception as e:
#         print(f"Error stopping sound: {e}")

# def eye_aspect_ratio(eye):
# 	A = distance.euclidean(eye[1], eye[5])
# 	B = distance.euclidean(eye[2], eye[4])
# 	C = distance.euclidean(eye[0], eye[3])
# 	ear = (A + B) / (2.0 * C)
# 	return ear

# # Function to calculate mouth aspect ratio (for yawn detection)
# def mouth_aspect_ratio(mouth):
# 	# Vertical landmarks (top, bottom)
# 	A = distance.euclidean(mouth[3], mouth[9])  # Vertical distance
# 	B = distance.euclidean(mouth[2], mouth[10]) # Additional vertical distance
# 	C = distance.euclidean(mouth[4], mouth[8])  # Additional vertical distance
# 	# Horizontal landmarks (left, right)
# 	D = distance.euclidean(mouth[0], mouth[6])  # Horizontal distance
# 	# Compute ratio - larger value when mouth is open
# 	mar = (A + B + C) / (3.0 * D)
# 	return mar
	
# # Function to calculate head tilt angle
# def calculate_tilt(shape):
# 	# Get coordinates for key facial landmarks
# 	left_eye = np.mean(shape[36:42], axis=0)
# 	right_eye = np.mean(shape[42:48], axis=0)
# 	nose_tip = shape[33]
# 	left_mouth = shape[48]
# 	right_mouth = shape[54]
	
# 	# Calculate horizontal tilt
# 	eye_angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], 
# 										 right_eye[0] - left_eye[0]))
	
# 	# Calculate vertical tilt using nose and midpoint between eyes
# 	eyes_mid = (left_eye + right_eye) / 2
# 	mouth_mid = (left_mouth + right_mouth) / 2
# 	vertical_angle = math.degrees(math.atan2(nose_tip[0] - eyes_mid[0], 
# 											 nose_tip[1] - eyes_mid[1]))
	
# 	# Draw reference lines for visualization
# 	return abs(eye_angle), abs(vertical_angle), left_eye, right_eye, eyes_mid, nose_tip, mouth_mid
	
# # Thresholds and initializations
# thresh = 0.25
# frame_check = 30
# horizontal_tilt_thresh = 10.0  # Horizontal tilt threshold in degrees
# vertical_tilt_thresh = 25.0    # Vertical tilt threshold in degrees
# tilt_time_threshold = 3        # Time in seconds for head tilt alert
# tilt_start_time = None
# tilt_flag = False

# # Yawn detection thresholds and initialization
# yawn_thresh = 0.52              # Lower threshold for mouth aspect ratio when yawning
# yawn_frames = 10                # Fewer consecutive frames needed to confirm yawning
# yawn_counter = 0
# yawn_flag = False

# # Alert state trackers
# drowsy_alert_active = False
# yawn_alert_active = False
# tilt_alert_active = False

# # Overall alert status
# alert_active = False

# # Alert counters
# alert_counts = {
#     'drowsy': 0,
#     'yawn': 0,
#     'tilt': 0,
#     'total': 0
# }

# # Load face detector and landmark predictor
# detect = dlib.get_frontal_face_detector()

# # More robust path resolution for the model file
# model_file = "shape_predictor_68_face_landmarks.dat"
# if not os.path.exists(model_file):
#     print(f"Warning: Model file '{model_file}' not found in current directory")
#     # Search in the directory of the script
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     alternative_path = os.path.join(script_dir, model_file)
#     if os.path.exists(alternative_path):
#         model_file = alternative_path
#         print(f"Found model file at {model_file}")
#     else:
#         alternative_path = os.path.join(script_dir, "models", "shape_predictor_68_face_landmarks.dat")
#         if os.path.exists(alternative_path):
#             model_file = alternative_path
#             print(f"Found model file at {model_file}")
#         else:
#             print(f"ERROR: Model file not found!")
#             print(f"Please ensure '{model_file}' exists in the current directory or in a 'models' subdirectory.")
#             print(f"Current working directory: {os.getcwd()}")
#             print(f"Script directory: {script_dir}")
#             sys.exit(1)

# try:
#     predict = dlib.shape_predictor(model_file)
#     print(f"Successfully loaded face landmark predictor from {model_file}")
# except Exception as e:
#     print(f"Error loading face landmark predictor: {e}")
#     sys.exit(1)

# # Get facial landmarks indexes
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# # Initialize camera with better settings
# cap = cv2.VideoCapture(0)
# # Set lower resolution for better performance
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# # Set buffer size
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# drowsy_flag = 0

# print("DriveGuard started: Press 'q' to quit")

# # Frame rate control for consistency
# target_fps = 20
# frame_time = 1.0 / target_fps
# last_frame_time = time.time()

# while True:
# 	# Frame rate control
# 	current_time = time.time()
# 	elapsed = current_time - last_frame_time
	
# 	if elapsed < frame_time:
# 		# Sleep to maintain consistent frame rate
# 		time.sleep(frame_time - elapsed)
	
# 	last_frame_time = time.time()
	
# 	ret, frame = cap.read()
# 	if not ret:
# 		print("Failed to grab frame!")
# 		# Attempt to recover connection
# 		cap.release()
# 		time.sleep(0.5)
# 		cap = cv2.VideoCapture(0)
# 		continue
		
# 	frame = imutils.resize(frame, width=600)
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	subjects = detect(gray, 0)
	
# 	# Reset alert status for this frame
# 	current_alert_active = False
	
# 	# Display status on frame
# 	cv2.putText(frame, "DriveGuard Active", (10, 30),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	
# 	for subject in subjects:
# 		shape = predict(gray, subject)
# 		shape = face_utils.shape_to_np(shape)
		
# 		# ---------- Drowsiness Detection ----------
# 		leftEye = shape[lStart:lEnd]
# 		rightEye = shape[rStart:rEnd]
# 		leftEAR = eye_aspect_ratio(leftEye)
# 		rightEAR = eye_aspect_ratio(rightEye)
# 		ear = (leftEAR + rightEAR) / 2.0
		
# 		# Visualize eyes
# 		leftEyeHull = cv2.convexHull(leftEye)
# 		rightEyeHull = cv2.convexHull(rightEye)
# 		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
# 		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
# 		# ---------- Yawn Detection ----------
# 		mouth = shape[mStart:mEnd]
# 		mouthHull = cv2.convexHull(mouth)
# 		cv2.drawContours(frame, [mouthHull], -1, (0, 165, 255), 1)
		
# 		# Calculate mouth aspect ratio
# 		mar = mouth_aspect_ratio(mouth)
		
# 		# Display MAR value
# 		cv2.putText(frame, f"MAR: {mar:.2f}", (420, 80),
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
		
# 		# Check for yawning with debounce
# 		if mar > yawn_thresh:
# 			yawn_counter += 1
# 			# Show counter progress
# 			cv2.putText(frame, f"Yawn counter: {yawn_counter}/{yawn_frames}", (10, 180),
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
			
# 			current_time = time.time()
# 			if yawn_counter >= yawn_frames:
# 				if not yawn_alert_active or current_time - last_yawn_alert > alert_cooldown:
# 					last_yawn_alert = current_time
# 					yawn_alert_active = True
# 					# Increment counter only when new alert is triggered
# 					alert_counts['yawn'] += 1
# 					alert_counts['total'] += 1
				
# 				if yawn_alert_active:
# 					yawn_flag = True
# 					current_alert_active = True
# 					cv2.putText(frame, "YAWNING ALERT!", (10, 150),
# 							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# 		else:
# 			yawn_counter = max(0, yawn_counter - 1)  # Gradual decrease in counter
# 			yawn_flag = False
# 			yawn_alert_active = False
		
# 		# Check for drowsiness with debounce
# 		if ear < thresh:
# 			drowsy_flag += 1
# 			current_time = time.time()
# 			if drowsy_flag >= frame_check:
# 				if not drowsy_alert_active or current_time - last_drowsy_alert > alert_cooldown:
# 					last_drowsy_alert = current_time
# 					drowsy_alert_active = True
# 					# Increment counter only when new alert is triggered
# 					alert_counts['drowsy'] += 1
# 					alert_counts['total'] += 1
				
# 				if drowsy_alert_active:
# 					current_alert_active = True
# 					cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
# 						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# 		else:
# 			drowsy_flag = 0
# 			drowsy_alert_active = False
				
# 		# ---------- Improved Head Tilt Detection ----------
# 		# Calculate horizontal and vertical tilt angles
# 		h_angle, v_angle, left_eye, right_eye, eyes_mid, nose_tip, mouth_mid = calculate_tilt(shape)
		
# 		# Draw visual guides for tilt detection
# 		cv2.line(frame, tuple(left_eye.astype(int)), tuple(right_eye.astype(int)), (255, 255, 0), 1)
# 		cv2.line(frame, tuple(eyes_mid.astype(int)), tuple(nose_tip.astype(int)), (255, 0, 255), 1)
# 		cv2.line(frame, tuple(nose_tip.astype(int)), tuple(mouth_mid.astype(int)), (0, 255, 255), 1)
		
# 		# Display tilt values
# 		cv2.putText(frame, f"H-Tilt: {h_angle:.1f}°", (420, 30),
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
# 		cv2.putText(frame, f"V-Tilt: {v_angle:.1f}°", (420, 55),
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
					
# 		# Check for head tilt with debounce
# 		if h_angle > horizontal_tilt_thresh or v_angle > vertical_tilt_thresh:
# 			# Start timer if not already started
# 			if tilt_start_time is None:
# 				tilt_start_time = time.time()
				
# 			# Calculate how long the head has been tilted
# 			tilt_duration = time.time() - tilt_start_time
			
# 			# Show tilt duration
# 			cv2.putText(frame, f"Tilt duration: {tilt_duration:.1f}s", (350, 105),
# 						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
			
# 			# Alert if tilt exceeds threshold with debounce
# 			current_time = time.time()
# 			if tilt_duration > tilt_time_threshold:
# 				if not tilt_alert_active or current_time - last_tilt_alert > alert_cooldown:
# 					last_tilt_alert = current_time
# 					tilt_alert_active = True
# 					# Increment counter only when new alert is triggered
# 					alert_counts['tilt'] += 1
# 					alert_counts['total'] += 1
				
# 				if tilt_alert_active:
# 					tilt_flag = True
# 					current_alert_active = True
# 					if h_angle > horizontal_tilt_thresh:
# 						cv2.putText(frame, "HEAD SIDEWAYS ALERT!", (10, 90),
# 								cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# 					# if v_angle > vertical_tilt_thresh: 
# 					# 	cv2.putText(frame, "HEAD DOWN/UP ALERT!", (10, 120),
# 					# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# 		else:
# 			# Reset tilt timer if head is straight
# 			tilt_start_time = None
# 			tilt_flag = False
# 			tilt_alert_active = False
	
# 	# Handle sound alerts continuously until corrected
# 	if current_alert_active:
# 		# Show sound alert indicator
# 		cv2.putText(frame, "SOUND ALERT ACTIVE", (350, 130),
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
		
# 		# Play sound continuously until the alert condition is resolved
# 		if not mixer.music.get_busy():
# 			play_alert_sound()
		
# 		alert_active = True
# 	else:
# 		if alert_active:
# 			stop_alert_sound()
# 			alert_active = False
	
# 	# Display alert counts on frame
# 	cv2.putText(frame, f"Drowsy Alerts: {alert_counts['drowsy']}", (10, 210),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
# 	cv2.putText(frame, f"Yawn Alerts: {alert_counts['yawn']}", (10, 230),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
# 	cv2.putText(frame, f"Tilt Alerts: {alert_counts['tilt']}", (10, 250),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
# 	cv2.putText(frame, f"Total Alerts: {alert_counts['total']}", (10, 270),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	
# 	# Display the frame
# 	cv2.imshow("DriveGuard", frame)
	
# 	# Break loop on 'q' press
# 	key = cv2.waitKey(1) & 0xFF
# 	if key == ord("q"):
# 		break

# # Clean up
# stop_alert_sound()
# cv2.destroyAllWindows()
# cap.release()
# print("DriveGuard stopped")































import streamlit as st
import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
import os
import time

# ------------------- Streamlit Setup -------------------
st.set_page_config(page_title="DriveGuard - Drowsiness Detection", page_icon="🚗", layout="wide")
st.title("🚗 DriveGuard: Real-Time Driver Drowsiness Detection System")
st.markdown("""
**DriveGuard** detects driver drowsiness, yawning, and head tilt using facial landmarks via OpenCV & Dlib.  
Grant webcam access and click **Start Detection** to test it live!
""")

# ------------------- Sidebar Settings -------------------
st.sidebar.header("⚙️ Settings")
thresh = st.sidebar.slider("Eye Aspect Ratio Threshold", 0.1, 0.4, 0.25)
yawn_thresh = st.sidebar.slider("Yawn Threshold", 0.3, 0.8, 0.52)
frame_check = st.sidebar.slider("Frame Check (EAR)", 10, 60, 30)
run = st.sidebar.checkbox("▶️ Start Detection", False)

# ------------------- Load Dlib Model -------------------
model_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(model_path):
    st.error("❌ Model file 'shape_predictor_68_face_landmarks.dat' not found in the app directory.")
    st.stop()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

FRAME_WINDOW = st.image([])

# ------------------- Helper Functions -------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    D = distance.euclidean(mouth[0], mouth[6])
    return (A + B + C) / (3.0 * D)

# ------------------- Main Detection Logic -------------------
cap = None
if run:
    cap = cv2.VideoCapture(0)
    drowsy_counter = 0
    yawn_counter = 0

    st.info("🟢 Detection running... Press 'Stop' in sidebar to end.")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to access webcam.")
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)

        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Draw contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 165, 255), 1)

            # ------------------- Drowsiness Detection -------------------
            if ear < thresh:
                drowsy_counter += 1
                if drowsy_counter >= frame_check:
                    cv2.putText(frame, "DROWSINESS ALERT!", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # 🔊 Play sound alert in browser
                    st.markdown(
                        '<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg"></audio>',
                        unsafe_allow_html=True
                    )
            else:
                drowsy_counter = 0

            # ------------------- Yawn Detection -------------------
            if mar > yawn_thresh:
                yawn_counter += 1
                if yawn_counter > 15:
                    cv2.putText(frame, "YAWN ALERT!", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # 🔊 Play sound alert in browser
                    st.markdown(
                        '<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg"></audio>',
                        unsafe_allow_html=True
                    )
            else:
                yawn_counter = 0

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.warning("⏹️ Click 'Start Detection' in sidebar to begin.")
