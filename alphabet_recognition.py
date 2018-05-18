from keras.models import load_model
from collections import deque
import numpy as np
import cv2

# Load the models built in the previous steps
mlp_model = load_model('emnist_mlp_model.h5')
cnn_model = load_model('emnist_cnn_model.h5')

# Letters lookup
letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Define Black Board
blackboard = np.zeros((480,640,3), dtype=np.uint8)
alphabet = np.zeros((200, 200, 3), dtype=np.uint8)

# Setup deques to store alphabet drawn on screen
points = deque(maxlen=512)

# Define prediction variables
prediction1 = 26
prediction2 = 26

index = 0
# Load the video
camera = cv2.VideoCapture(0)

# Keep looping
while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Find contours (bottle cap in my case) in the image
    (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Check to see if any contours were found
    if len(cnts) > 0:
    	# Sort the contours and find the largest one -- we
    	# will assume this contour correspondes to the area of the bottle cap
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        points.appendleft(center)

    elif len(cnts) == 0:
        if len(points) != 0:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            if len(blackboard_cnts) >= 1:
                cnt = sorted(blackboard_cnts, key = cv2.contourArea, reverse = True)[0]

                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    alphabet = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
                    newImage = cv2.resize(alphabet, (28, 28))
                    newImage = np.array(newImage)
                    newImage = newImage.astype('float32')/255

                    prediction1 = mlp_model.predict(newImage.reshape(1,28,28))[0]
                    prediction1 = np.argmax(prediction1)

                    prediction2 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
                    prediction2 = np.argmax(prediction2)

            # Empty the points deque and the blackboard
            points = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

    # Connect the points with a line
    for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                    continue
            cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
            cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

    # Put the result on the screen
    cv2.putText(frame, "Multilayer Perceptron : " + str(letters[int(prediction1)+1]), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
    cv2.putText(frame, "Convolution Neural Network:  " + str(letters[int(prediction2)+1]), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("alphabets Recognition Real Time", frame)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
