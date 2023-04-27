from flask import Flask, render_template, request
import cv2
import os
import numpy as np

import dlib
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize

import matplotlib.pyplot as plt

app = Flask(__name__)

def detect_faces(img):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    return faces

def extract_faces(img, faces):
    cut_faces = []
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        roi = img[y:y+h, x:x+w]
        cut_faces.append(roi)
    return cut_faces

def analyze_emotions(imgs):
    model = load_model('Resource/expression.model')
    preds_emotional = []
    for i in imgs:
        image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48)) / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image)
        
        preds_emotional.append(tf.argmax(predictions, axis=1)[0].numpy())
    return preds_emotional

def draw_boxes(img, faces, emotions):
    for face, emotion in zip(faces, emotions):
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, emotion, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def detect_slides(video, slide_similarity_threshold=0.85, skip_frames=3):
    skip_frames = 5
    previous_slide = None
    slide_count = 0
    slides = {}
    Emotions = {}
    largest_area = 0

    frame_count = 0
    slide_duration = 0

    slide_similarity_threshold = 0.999 #best

    # Get frame rate of video
    fps = video.get(cv2.CAP_PROP_FPS)
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Angry"]
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % skip_frames != 0:
            frame_count += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        slide_contour = None
        largest_area = 0
        
        # Loop through the contours
        for contour in contours:
            # Get the area of the contour
            area = cv2.contourArea(contour)
            
            # If the area is too small, ignore the contour
            if area < 100000:
                continue
            
            # Find the largest contour that represents the slide
            if area > largest_area:
                slide_contour = contour
                largest_area = area

        # Extract the slide from the frame and perform similarity check
        if slide_contour is not None:
            
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(slide_contour)
            
            # Extract the slide image from the frame
            slide = frame[y:y+h, x+20:x+w]

            # Check if the current slide is similar to the previous slide
            if previous_slide is not None:
                # Check size of slide and previous_slide images
                if slide.shape[0] > previous_slide.shape[0] or slide.shape[1] > previous_slide.shape[1]:
                    # Resize the previous_slide image to match the size of the slide image
                    previous_slide = cv2.resize(previous_slide, (slide.shape[1], slide.shape[0]))

                similarity = cv2.matchTemplate(slide, previous_slide, cv2.TM_CCORR_NORMED)
                if np.any(similarity >= slide_similarity_threshold):
                    frame_count += 1
                    continue

            # Calculate the duration of the previous slide
            slide_duration = (frame_count * skip_frames) / fps
            frame_count = 0

            # Add the slide image and timestamp to the slides dictionary
            slides[slide_count] = {'image': slide, 'timestamp': slide_duration}

            cv2.imwrite(f'static/slide/slide_{slide_count+1}.jpg', slides[slide_count]['image'])
            print(f'Slide {slide_count+1}: {slides[slide_count]["timestamp"]}')

            #predict emotional
            faces = detect_faces(frame)
            cut_faces = extract_faces(frame, faces)
            emotions = analyze_emotions(cut_faces)
            emotions_lb = []
            
            for i in emotions:
                emotions_lb.append(labels[int(i)])
            frame_draw = draw_boxes(frame, faces, emotions_lb)
            Emotions[slide_count] = {'image': frame_draw, 'emo': emotions_lb}
            cv2.imwrite(f'static/emotion/emotion_{slide_count+1}.jpg', Emotions[slide_count]['image'])
            # Increment the slide count
            slide_count += 1

            # Set the previous slide to the current slide
            previous_slide = slide

        # Increment frame count
        frame_count += 1
    return slides, Emotions


@app.route('/')
def index(): return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    video = request.files['video']
    video_path = os.path.join('Resource/', video.filename)
    video = cv2.VideoCapture(video_path)
    slides,Emotions = detect_slides(video)
    video.release()
    
    fig, ax = plt.subplots()
    slide_nums = list(Emotions.keys())
    emotions = [Emotions[i]['emo'] for i in slide_nums]
    flattened_emotions = [e for sublist in emotions for e in sublist]
    unique_emotions = list(set(flattened_emotions))  # convert set to list
    emotion_counts = [flattened_emotions.count(e) for e in unique_emotions]
    ax.bar(unique_emotions, emotion_counts)
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.set_title('Emotion Summary')
    plt.xticks(rotation=45)
    plt.tight_layout()
    emotion_summary_path = os.path.join('static/graph/', 'emotion_summary.png')
    plt.savefig(emotion_summary_path)
    plt.close()
    
    return render_template('showcase.html', slides=slides, emotionals=Emotions, emotion_summary_path=emotion_summary_path)

if __name__ == '__main__': app.run(debug=True, port=5001)
