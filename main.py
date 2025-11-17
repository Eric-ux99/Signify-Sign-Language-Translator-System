import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyttsx3
from nltk.corpus import words
import nltk


class SignifyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language To Text Conversion")

        # Initialize Sentence and Prediction
        self.sentence = ""
        self.predicted_char = ""
        self.suggestions = []

        # Load dictionary words
        nltk.download('words')
        self.dictionary_words = set(word.upper() for word in words.words())

        # Text-to-Speech Engine
        self.speaker = pyttsx3.init()

        # UI Components
        self.create_ui()
        self.bind_keys()

        # Initialize Camera and Model
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                       "T", "U", "V", "W", "X", "Y", "Z", "I am"]

        self.offset = 20
        self.imgSize = 300

        # Start Video Feed
        self.update_frame()

    def create_ui(self):
        """Builds the main UI layout."""
        # Title
        title = ttk.Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 24, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=10)

        # Video Feed
        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=1, column=0, padx=10, pady=10)

        # Cropped Hand Image
        self.crop_label = ttk.Label(self.root)
        self.crop_label.grid(row=1, column=1, padx=10, pady=10)

        # Text Information
        ttk.Label(self.root, text="Character :", font=("Helvetica", 16)).grid(row=2, column=0, sticky="e")
        self.character_label = ttk.Label(self.root, text="-", font=("Helvetica", 16))
        self.character_label.grid(row=2, column=1, sticky="w")

        ttk.Label(self.root, text="Sentence :", font=("Helvetica", 16)).grid(row=3, column=0, sticky="e")
        self.sentence_label = ttk.Label(self.root, text=self.sentence, font=("Helvetica", 16))
        self.sentence_label.grid(row=3, column=1, sticky="w")

        # Suggestions
        ttk.Label(self.root, text="Suggestions :", font=("Helvetica", 16, "bold"), foreground="red").grid(row=4,
                                                                                                          column=0,
                                                                                                          sticky="e")
        self.suggestions_frame = ttk.Frame(self.root)
        self.suggestions_frame.grid(row=4, column=1, sticky="w")

        # Instructions under camera
        instruction_label = ttk.Label(self.root,
                                      text="Press Shift to record text\nPress Spacebar for space\nPress Backspace to delete character",
                                      font=("Helvetica", 14), foreground="blue")
        instruction_label.grid(row=1, column=2, padx=10, pady=10)

        # Buttons
        self.clear_button = ttk.Button(self.root, text="Clear", command=self.clear_sentence, style="Large.TButton")
        self.clear_button.grid(row=5, column=0, pady=10)

        self.speak_button = ttk.Button(self.root, text="Speak", command=self.speak_sentence, style="Large.TButton")
        self.speak_button.grid(row=5, column=1, pady=10)

        # Configure Button Style
        style = ttk.Style()
        style.configure("Large.TButton", font=("Helvetica", 14), padding=10)

    def bind_keys(self):
        """Binds keyboard inputs for recording, spaces, and backspace."""
        self.root.bind('<Shift_L>', self.add_prediction)
        self.root.bind('<space>', self.add_space)
        self.root.bind('<BackSpace>', self.delete_character)

    def update_frame(self):
        """Reads the video feed and processes gestures."""
        success, img = self.cap.read()
        if not success:
            self.root.after(30, self.update_frame)
            return

        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Ensure crop is valid
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = self.imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                    wGap = math.ceil((self.imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = self.imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                    hGap = math.ceil((self.imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                self.predicted_char = self.labels[index]

                # Display Predicted Character
                self.character_label.config(text=self.predicted_char)
                self.update_suggestions()

                # Draw Filled Rectangle and Text
                cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50),
                              (x - self.offset + 90, y - self.offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, self.predicted_char, (x, y - 26),
                            cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 1)

                # Draw Rectangle Around Hand
                cv2.rectangle(imgOutput, (x - self.offset, y - self.offset),
                              (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)

                # Display Cropped Hand Image
                imgCropResized = cv2.resize(imgWhite, (300, 300))
                imgCropRGB = cv2.cvtColor(imgCropResized, cv2.COLOR_BGR2RGB)
                imgCropRGB = Image.fromarray(imgCropRGB)
                imgCropTk = ImageTk.PhotoImage(image=imgCropRGB)
                self.crop_label.imgtk = imgCropTk
                self.crop_label.configure(image=imgCropTk)

        imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        imgRGB = Image.fromarray(imgRGB)
        imgtk = ImageTk.PhotoImage(image=imgRGB)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_frame)

    def add_prediction(self, event):
        """Appends the predicted character to the sentence."""
        self.sentence += self.predicted_char
        self.sentence_label.config(text=self.sentence)
        self.update_suggestions()

    def add_space(self, event):
        """Adds a space to the sentence."""
        self.sentence += " "
        self.sentence_label.config(text=self.sentence)
        self.update_suggestions()

    def delete_character(self, event):
        """Deletes the last character from the sentence."""
        self.sentence = self.sentence[:-1]
        self.sentence_label.config(text=self.sentence)
        self.update_suggestions()

    def update_suggestions(self):
        """Updates text recommendations dynamically using the dictionary."""
        partial_input = self.sentence.strip().split(" ")[-1].upper()  # Get the last word being typed

        # Find words from dictionary starting with the partial input
        if partial_input:  # Show suggestions only if there's input
            self.suggestions = [word for word in self.dictionary_words if word.startswith(partial_input)][:5]
        else:
            self.suggestions = []  # Clear suggestions if no input

        self.display_suggestions()

    def display_suggestions(self):
        """Displays the suggestions in the UI."""
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()
        for suggestion in self.suggestions:
            btn = ttk.Button(self.suggestions_frame, text=suggestion,
                             command=lambda s=suggestion: self.apply_suggestion(s))
            btn.pack(side=tk.LEFT, padx=5)

    def apply_suggestion(self, word):
        """Replaces the last word in the sentence with the selected suggestion."""
        words = self.sentence.strip().split(" ")
        words[-1] = word
        self.sentence = " ".join(words) + " "
        self.sentence_label.config(text=self.sentence)
        self.update_suggestions()

    def speak_sentence(self):
        """Speaks the current sentence using text-to-speech."""
        self.speaker.say(self.sentence)
        self.speaker.runAndWait()

    def clear_sentence(self):
        """Clears the current sentence."""
        self.sentence = ""
        self.sentence_label.config(text=self.sentence)
        self.suggestions = []
        self.display_suggestions()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignifyApp(root)
    root.mainloop()