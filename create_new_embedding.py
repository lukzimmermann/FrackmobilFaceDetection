import cv2 as cv
import numpy as np
import time
import os
import sys
import json
from deepface import DeepFace

TIME_INTERVAL = 0.2
NUMBER_OF_IMAGES = 30


def main(name: str):
    path = f'faces/{name}'
    if not os.path.exists(path):
        os.makedirs(path)

    cap = cv.VideoCapture(1)
    cv.namedWindow("frame", cv.WND_PROP_FULLSCREEN)

    captured_images = 0
    target_time = time.time() + TIME_INTERVAL

    while captured_images < NUMBER_OF_IMAGES:
        _, image = cap.read()

        remaining_time = np.abs(target_time-time.time())
        image = cv.putText(image.copy(), str(f'{remaining_time:.1f}'), (20, 80), cv.FONT_HERSHEY_SIMPLEX,  
           3, (0,0,255), 3, cv.LINE_AA)
        
        cv.imshow("frame", image)

        if time.time() > target_time:
            try:
                embedding_objs = DeepFace.represent(img_path = image)
                embedding = embedding_objs[0]['embedding']
                confidence = embedding_objs[0]['face_confidence']
                x = embedding_objs[0]['facial_area']['x']
                y = embedding_objs[0]['facial_area']['y']
                w = embedding_objs[0]['facial_area']['w']
                h = embedding_objs[0]['facial_area']['h']

                image = cv.rectangle(image.copy(), (x,y), (x+w, y+h), (0,0,255), 3)
                
                cv.imshow("frame", image)
                cv.waitKey(500)
                if confidence > 0.85:
                    with open(f'{path}/{name}_{captured_images+1}.json', 'w') as file:
                        json.dump(embedding, file)
                    print("Face recorded...")
                    captured_images += 1
                    
            except:
                print("No face detected...")
            finally:
                target_time = time.time() + TIME_INTERVAL

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No name in parameter...")
        sys.exit(1)

    name = sys.argv[1]
    main(name)