import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize webcam capture
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Define offset for cropping the hand image and target size for output image
offset = 20
imgSize = 300

folder = "Data/I AM"
counter = 0

while True:
    # Capture frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    # Create a white background for the UI
    ui = np.ones((720, 1280, 3), np.uint8) * 255

    # Display title
    cv2.putText(ui, "Sign Language Data Collection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Display instructions
    cv2.putText(ui, "Press 's' to save the current hand image", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(ui, "Press 'Esc' to quit", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(ui, f"Images collected: {counter}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display the webcam feed
    imgResized = cv2.resize(img, (640, 480))
    ui[200:200 + 480, 50:50 + 640] = imgResized

    if hands:
        hand = hands[0]  # Get first and only hand detected
        x, y, w, h = hand['bbox']  # Get the bounding box of the hand

        # Ensure the bounding box is within the frame boundaries
        x1, y1, x2, y2 = max(0, x - offset), max(0, y - offset), min(img.shape[1], x + w + offset), min(img.shape[0],
                                                                                                        y + h + offset)

        # Crop the image around the hand with offset
        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape  # Get shape of cropped image

        if imgCropShape[0] > 0 and imgCropShape[1] > 0:  # Ensure the cropped image is not empty
            # Create a white image with the target size
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Calculate aspect ratio of cropped hand image
            aspectRatio = h / w

            if aspectRatio > 1:
                # If height is greater than width
                k = imgSize / h  # Calculate scaling factor
                wCal = math.ceil(k * w)  # Calculate the new width after resizing

                # Resize the cropped image to fit target height
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape  # Get the shape of the resized image

                # Calculate gap to center the image horizontally
                wGap = math.ceil((imgSize - wCal) / 2)

                # Place the resized image on the white background
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                # If width is greater than height
                k = imgSize / w
                hCal = math.ceil(k * h)

                # Resize cropped image to fit height
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape

                # Calculate gap to center the image
                hGap = math.ceil((imgSize - hCal) / 2)

                # Place the resized image on white background
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display the cropped and resized image on the UI
            imgWhiteResized = cv2.resize(imgWhite, (300, 300))
            ui[200:200 + 300, 700:700 + 300] = imgWhiteResized

    # Display the UI
    cv2.imshow("Sign Language Data Collection", ui)

    # Wait for key press
    key = cv2.waitKey(1)

    # If 's' is pressed, save the current processed hand image
    if key == ord("s"):
        if 'imgWhite' in locals():  # Check if imgWhite exists
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # Save image
            print(f"Image saved: {counter}")  # Print counter value on screen

    # If esc key is pressed, exit the loop and close window
    if key == 27:
        break

    # Check if the window close button (X) is clicked
    if cv2.getWindowProperty("Sign Language Data Collection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()