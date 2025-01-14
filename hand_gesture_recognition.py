import cv2
import numpy as np

def hand_gesture_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible!")
        return
    while True:
        ret ,frame = cap.read()
        
        if not ret:
            print("Error: Unable to capture frame.")
            break
        frame = cv2.flip(frame,1)
        roi = frame[100:400 , 100:400]

        # gray = cv2.cvtColor(roi, cv2.COLOR.BGR2GRAY)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray,(35,35),0)

        _, thresholded = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _=cv2.findContours(thresholded.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours,key=cv2.contourArea)
            cv2.drawContours(roi,[contour],-1,(0,255,0),3)
            hull = cv2.convexHull(contour)
            cv2.drawContours(roi,[hull],-1,(0,0,255),2)
        cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2) 
        cv2.imshow("Hand Gesture Recongition", frame)
        cv2.imshow("Thresholded",thresholded)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_gesture_recognition()