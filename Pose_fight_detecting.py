import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec


hand_model_path="model/hand_landmarker.task"
pose_model_path="model/pose_landmarker_lite.task"


connection_style = DrawingSpec(color=(0, 0, 255), thickness=2)

def convert_to_mp_image(image):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)


## for hand landmarker

hand_options=vision.HandLandmarkerOptions(                             
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=4,
)


## for pose landmarker
pose_options=vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=pose_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=2,
)


#create object for hand
hand_detector=vision.HandLandmarker.create_from_options(hand_options)

##create object for pose
pose_detector=vision.PoseLandmarker.create_from_options(pose_options)


##start video
cap=cv2.VideoCapture(0)
cap.set(3,780)
cap.set(4,780)

start_time = cv2.getTickCount()

while True:
    success,frames = cap.read()
    if not success:
        break
    

    ## bgr to rgb
    RGB_frames=cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
    rgb_image=convert_to_mp_image(RGB_frames)

    current_time=cv2.getTickCount()

    #calculating time stamp between the frames
    timestamp = int(((current_time-start_time)/cv2.getTickFrequency())*1e6)
    
    #for detecting hands

    hand_results = hand_detector.detect_for_video(rgb_image,timestamp)

    if hand_results.hand_landmarks:
        for hand_landmarks in hand_results.hand_landmarks:
             # Convert the simple list to MediaPipe's special object
            landmark_list = landmark_pb2.NormalizedLandmarkList(
            landmark=[landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks]
        )
            
            solutions.drawing_utils.draw_landmarks(                                 ##this will draw landmarks on hand 
            image=frames,
            landmark_list=landmark_list,                                            ##landmarks is a list of 21 handlandmarks point
            connections=solutions.hands.HAND_CONNECTIONS,                           ##this will draw connection between 21 landmarks
            landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(),                          ##tell how an  individual dot should look like
            connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style()                       ##tells how connection should look like
        )
            

    #for detecting pose 
    pose_results = pose_detector.detect_for_video(rgb_image,timestamp)

    if pose_results.pose_landmarks:
        for pose_landmarks in pose_results.pose_landmarks:
            # Convert the simple list to MediaPipe's special object
            landmark_list = landmark_pb2.NormalizedLandmarkList(
            landmark=[landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks]
        )
            solutions.drawing_utils.draw_landmarks(
            image=frames,
            landmark_list=landmark_list,
            connections=solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=connection_style
        )
    
    cv2.imshow("camera",frames)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
            





