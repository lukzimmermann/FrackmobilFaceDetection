import numpy as np
import cv2 as cv
from deepface import DeepFace

class FaceDetector():
    def __init__(self, max_faces: int) -> None:
        self.max_faces = max_faces
        self.actions = ['age', 'gender', 'race', 'emotion']

    def detect(self, image: np.ndarray) -> np.ndarray:
        faces = DeepFace.analyze(img_path=image, 
            actions = self.actions,
            silent=True,
            enforce_detection=False
        )

        final_image = image.copy()

        for i, face in enumerate(faces):
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']
            age = face['age']
            gender = face['dominant_gender']
            race = face['dominant_race']
            emotion = face['dominant_emotion']

            if h == image.shape[0] and w == image.shape[1]:
                break

            final_image = self.__draw_box(final_image, x, y, w, h, age, gender, race, emotion)
            
            if i == self.max_faces-1:
                break
        
        return final_image
    
    def __draw_box(self,
                   image: np.ndarray,
                   x: int,
                   y: int,
                   w: int,
                   h: int,
                   age: int,
                   gender: str,
                   race: str,
                   emotion: str) -> np.ndarray:
        
        final_image = cv.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 3)

        font_style = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_initial_offset = 10
        text_offset = 30

        final_image = cv.putText(final_image, str(age), (x+w+20, y+text_initial_offset), font_style,  
           font_scale, (0,0,255), font_thickness, cv.LINE_AA)
        final_image = cv.putText(final_image, gender, (x+w+20, y+text_initial_offset+1*text_offset), font_style,  
           font_scale, (0,0,255), font_thickness, cv.LINE_AA)
        final_image = cv.putText(final_image, race, (x+w+20, y+text_initial_offset+2*text_offset), font_style,  
           font_scale, (0,0,255), font_thickness, cv.LINE_AA) 
        final_image = cv.putText(final_image, emotion, (x+w+20, y+text_initial_offset+3*text_offset), font_style,  
           font_scale, (0,0,255), font_thickness, cv.LINE_AA)
        
        return final_image
