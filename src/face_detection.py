import numpy as np
import os
import cv2 as cv
from deepface import DeepFace
from src.embedding import Embedding

class FaceDetector():
    def __init__(self, max_faces: int) -> None:
        self.max_faces = max_faces
        self.actions = ['age', 'gender', 'race', 'emotion']
        self.faces = self.__load_embeddings()

        for face in self.faces:
            print(face)



    def detect(self, image: np.ndarray) -> np.ndarray:
        faces = DeepFace.analyze(img_path=image, 
            actions = self.actions,
            silent=True,
            enforce_detection=False
        )

        embedding_objs = DeepFace.represent(img_path = image)
    

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
            target_embedding = embedding_objs[i]['embedding']

            name = self.__find_face(target_embedding)

            if h == image.shape[0] and w == image.shape[1]:
                break
                
            final_image = self.__draw_box(final_image, x, y, w, h, age, gender, race, emotion, name)
            
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
                   emotion: str,
                   name: str) -> np.ndarray:
        
        color = (0,0,255)
        font_style = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_initial_offset = 10
        text_offset = 30
        
        if name != "":
            color = (0,255,255)

        final_image = cv.rectangle(image, (x,y), (x+w, y+h), color, 3)

        final_image = cv.putText(final_image, name, (x+w+20, y+text_initial_offset+0*text_offset), font_style,  
           font_scale, color, font_thickness, cv.LINE_AA)
        final_image = cv.putText(final_image, str(age), (x+w+20, y+text_initial_offset+1*text_offset), font_style,  
           font_scale, color, font_thickness, cv.LINE_AA)
        final_image = cv.putText(final_image, gender, (x+w+20, y+text_initial_offset+2*text_offset), font_style,  
           font_scale, color, font_thickness, cv.LINE_AA)
        final_image = cv.putText(final_image, race, (x+w+20, y+text_initial_offset+3*text_offset), font_style,  
           font_scale, color, font_thickness, cv.LINE_AA) 
        final_image = cv.putText(final_image, emotion, (x+w+20, y+text_initial_offset+4*text_offset), font_style,  
           font_scale, color, font_thickness, cv.LINE_AA)
        
        return final_image

    def __load_embeddings(self):
        embedding_list = []
        if os.path.exists('faces'):
            face_list = os.listdir('faces')

            for name in face_list:
                embedding_files = os.listdir(f'faces/{name}')
                embeddings = []
                embedding = []

                for file in embedding_files:
                    with open(f'faces/{name}/{file}', 'r') as file:
                        values = file.read().strip('[]').split(',')
                        embedding = [float(val.strip()) for val in values]

                    embeddings.append(embedding)

                embedding_list.append(Embedding(name, embeddings))
            return embedding_list
        else:
            return []
    
    def __find_face(self, target_embedding) -> str:
        best_result = ""
        min_distance = float('inf')

        for face in self.faces:
            distance = np.linalg.norm(np.array(target_embedding) - np.array(face.embedding))

            if distance < min_distance: 
                min_distance = distance
                best_result = face.name

        if min_distance < 0.75:
            return best_result
        else:
            return ""