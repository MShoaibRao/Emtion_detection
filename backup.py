import cv2
import numpy as np
from keras.models import load_model
import os

face_cascade_file = 'haarcascade_frontalface_alt.xml'
model_file = 'video.h5'


face_cascade = cv2.CascadeClassifier(face_cascade_file)
model = load_model(model_file)

def detect_emotions(image, emotion_labels):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    total_faces = 0
    happy_count = 0
    sad_count = 0
    nut_count = 0
    fear_count = 0
    disg_count = 0
    surp_count = 0
    angry_count = 0

    if len(faces) == 0:
        print("No faces detected")
    else:
        
        print("Faces detected")

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
            predictions = model.predict(roi_gray)
            predicted_emotion_index = np.argmax(predictions)
            predicted_emotion_label = emotion_labels[predicted_emotion_index]

            
            if predicted_emotion_label == 'Happy':
                happy_count += 1
            elif predicted_emotion_label == 'Sad':
                sad_count += 1
            elif predicted_emotion_label == 'Angry':
                angry_count += 1
            elif predicted_emotion_label == 'Disgust':
                disg_count += 1
            elif predicted_emotion_label == 'Surprise':
                surp_count += 1
            elif predicted_emotion_label == 'Fear':
                fear_count += 1
            elif predicted_emotion_label == 'Neutral':
                nut_count += 1
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, predicted_emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image, happy_count, sad_count, angry_count, surp_count, fear_count, disg_count, nut_count, total_faces

# img folder
image_folder = 'pics'

# listi
image_files = os.listdir(image_folder)

# initialize veriable
total_faces = 0
total_count = 0
happy_count = 0
sad_count = 0
angry_count = 0
surp_count = 0
fear_count = 0
disg_count = 0
nut_count = 0

# labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


# Processing each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Reading the image
    frame = cv2.imread(image_path)

    # Detect and classify emotions in the image
    processed_image, img_happy_count, img_sad_count, img_angry_count, img_surp_count, img_fear_count, img_disg_count, img_nut_count, total_faces = detect_emotions(frame, emotion_labels)

    # new nmbrs
    total_faces += 1
    total_count += 1
    happy_count += img_happy_count
    sad_count += img_sad_count
    angry_count += img_angry_count
    surp_count += img_surp_count
    fear_count += img_fear_count
    disg_count += img_disg_count
    nut_count += img_nut_count

    # Picture show krwao
    cv2.imshow('Emotion Detection', processed_image)
    cv2.waitKey(100)

print(happy_count)
print('face')
print(total_faces)

Asli_total_faces = happy_count+sad_count+angry_count+surp_count+fear_count+disg_count+nut_count
print(Asli_total_faces)
happy_per = (happy_count / Asli_total_faces) * 100
sad_per = (sad_count / Asli_total_faces) * 100
angry_per = (angry_count / Asli_total_faces) * 100
surp_per = (surp_count / Asli_total_faces) * 100
fear_per = (fear_count / Asli_total_faces) * 100
disg_per = (disg_count / Asli_total_faces) * 100
nut_per = (nut_count / Asli_total_faces) * 100





cv2.destroyAllWindows()
