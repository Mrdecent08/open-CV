from keras.models import load_model
import cv2
import numpy as np
from random import choice
from keras.preprocessing import image
REV_CLASS_MAP = {
    0: "stone",
    1: "scissor",
    2: "stone"
}
def mapper(val):
    return REV_CLASS_MAP[val]
def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"
    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"
model = load_model("model.h5")
cap = cv2.VideoCapture(0)
prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)
    roi = frame[100:500, 100:500]
    '''
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    '''
    img = image.load_img(roi, target_size = (256, 256),grayscale=True)
    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
    if prev_move != user_move_name:
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
    prev_move = user_move_name
    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    icon = cv2.imread(
        "images/{}.PNG".format(computer_move_name))
    icon = cv2.resize(icon, (400, 400))
    frame[100:500, 800:1200] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
