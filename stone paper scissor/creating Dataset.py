import cv2
import os
cap = cv2.VideoCapture(0)
dataset = "datasets/val"
name = "stone"
path = os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)
start = False
count = 0
num_samples=20

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if count == num_samples:
        break

    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    if start:
        roi = frame[100:500, 100:500]
        cv2.imwrite("%s/%s.jpg"%(path,count),roi)
        count+=1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        start = not start

    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()