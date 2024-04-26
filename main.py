import cv2 as cv
from src.face_detection import FaceDetector

def main():
    print("Initialize...")
    face_detector = FaceDetector(3)
    cap = cv.VideoCapture(1)
    cv.namedWindow("frame", cv.WND_PROP_FULLSCREEN)

    while True:
        try:
            _, frame = cap.read()
            frame = frame[:,0:1280,:]
            image = face_detector.detect(frame)
            cv.imshow("frame", image)
        except Exception as e:
            print(e)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()