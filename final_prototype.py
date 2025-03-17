import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import tensorflow as tf

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define the custom loss function
def stable_categorical_crossentropy(y_true, y_pred):
    # Assuming this is a more numerically stable version of categorical crossentropy
    # You might need to adjust this implementation to match the original
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)


# Load model
model_path = r"G:\GitHub\Thesis20-Models\Model-2-94_\new_model.keras"
model = keras.models.load_model(model_path, custom_objects={'stable_categorical_crossentropy': stable_categorical_crossentropy})


# Recognized gestures
actions = np.array(['good_morning', 'good_afternoon', 'good_evening', 'hello',
                    'how_are_you', 'Im_fine', 'nice_to_meet_you', 'thank_you',
                    'youre_welcome', 'see_you_tomorrow', 'understand',
                    "dont_understand", 'know', "dont_know", 'no', 'yes', 'wrong',
                    'correct', 'slow', 'fast', 'one', 'two', 'three', 'four', 'five',
                    'six', 'seven', 'eight', 'nine', 'ten', 'january', 'february',
                    'march', 'april', 'may', 'june', 'july', 'august', 'september',
                    'october', 'november', 'december', 'monday', 'tuesday',
                    'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'today',
                    'tomorrow', 'yesterday', 'father', 'mother', 'son', 'daughter',
                    'grandfather', 'grandmother', 'uncle', 'auntie', 'cousin',
                    'parents', 'boy', 'girl', 'man', 'woman', 'deaf',
                    'hard_of_hearing', 'wheelchair_person', 'blind', 'deaf_blind',
                    'married', 'blue', 'green', 'red', 'brown', 'black', 'white',
                    'yellow', 'orange', 'gray', 'pink', 'violet', 'light', 'dark',
                    'bread', 'egg', 'fish', 'meat', 'chicken', 'spaghetti', 'rice',
                    'longanisa', 'shrimp', 'crab', 'hot', 'cold', 'juice', 'milk',
                    'coffee', 'tea', 'beer', 'wine', 'sugar', 'no_sugar', 'my_name_is',
                    'are_you_deaf', 'what_is_your_name', 'I_know_a_little_sign',
                    'you_sign_fast', 'again', 'please_sign_slowly', 'sorry',
                    'handsome', 'see_you_later', 'hearing', 'I_forget', 'family',
                    'baby', 'wait', 'brother', 'sister', 'step_sister', 'step_brother',
                    'happy_birthday', 'I_love_you', 'goodbye', 'please', 'hungry',
                    'dog', 'child', 'teenager', 'cat', 'is_she_your_mother',
                    "whats_your_favorite_subject", 'they_are_pretty',
                    'they_are_kind', 'school', 'principal', 'friend', 'paper',
                    'student', 'teacher', 'library', 'childhood', 'close_friend',
                    'best_friend', 'boyfriend', "whats_wrong", 'keep_working',
                    'please_come_here', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                    'w', 'x', 'y', 'z', 'who', 'Im_not_fine', 'what', 'where', 'why',
                    'which', 'how', 'how_much', 'maybe', 'sign_language',
                    'could_you_please_teach_me', 'excuse_me', 'I_miss_you',
                    'I_live_in', 'nice_to_meet_you_too', 'take_care', 'do_you_like',
                    'science', 'mathematics', 'exam', 'computer', 'Filipino',
                    'English', 'assignment'])
# actions = np.array([
#     'GOOD MORNING', 'GOOD AFTERNOON', 'GOOD EVENING', 'HELLO', 
#     'HOW ARE YOU', 'IM FINE', 'NICE TO MEET YOU', 'THANK YOU', 
#     'YOURE WELCOME', 'SEE YOU TOMORROW', 'MONDAY', 'TUESDAY', 
#     'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY', 
#     'TODAY', 'TOMORROW', 'YESTERDAY', 'BLUE', 'GREEN', 'RED', 
#     'BROWN', 'BLACK', 'WHITE', 'YELLOW', 'ORANGE', 'GRAY', 'PINK', 
#     'VIOLET', 'LIGHT', 'DARK'
# ])


def mediapipe_detection(image, model):
    """Processes the image and returns the detected results."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), results

def draw_styled_landmarks(image, results):
    """Draws the detected landmarks on the image."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), 
                                   mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                                   mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# Setup video capture
cap = cv2.VideoCapture(0)
sequence = []
recognized_action = None
predictions = []
threshold = 0.10

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-120:]  # Keep the last 120 keypoints

        if len(sequence) == 120:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            recognized_action = actions[np.argmax(res)]
        #     predictions.append(np.argmax(res))


        # #3. Viz logic
        #     if np.unique(predictions[-10:])[0]==np.argmax(res):
        #         if res[np.argmax(res)] > threshold:

        #             if len(sentence) > 0:
        #                 if actions[np.argmax(res)] != sentence[-1]:
        #                     sentence.append(actions[np.argmax(res)])
        #             else:
        #                 sentence.append(actions[np.argmax(res)])

        #     if len(sentence) > 5:
        #         sentence = sentence[-5:]

        #     # Viz probabilities
        #     image = prob_viz(res, actions, image, colors)        
        # Display recognized action at the top of the image
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, recognized_action if recognized_action else '...', (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()