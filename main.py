import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import util


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(
            self.main_window, "login", "green", self.login
        )
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(
            self.main_window, "logout", "red", self.logout
        )
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window,
            "register new user",
            "gray",
            self.register_new_user,
            fg="black",
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = "./db"
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = "./log.txt"

    def add_webcam(self, label):
        if "cap" not in self.__dict__:
            self.cap = cv2.VideoCapture(0)  # Use 0 or 1 based on your system setup

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            self._label.after(20, self.process_webcam)
            return

        # Draw a face outline on the frame
        self.draw_face_outline(frame)

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def draw_face_outline(self, frame):
        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Define the ellipse parameters
        center = (width // 2, height // 2)  # Center of the ellipse
        axes = (150, 200)  # Width and height of the ellipse
        angle = 0  # Angle of rotation of the ellipse in degrees
        startAngle = 0  # Starting angle of the elliptic arc in degrees
        endAngle = 360  # Ending angle of the elliptic arc in degrees

        # Draw the ellipse
        cv2.ellipse(
            frame, center, axes, angle, startAngle, endAngle, (0, 255, 0), 2
        )

        # Add a label for guidance
        cv2.putText(
            frame,
            "Align your face here",
            (center[0] - 120, center[1] - axes[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def calculate_recognition_accuracy(self, unknown_embedding, known_embedding):
        # Calculate the Euclidean distance
        distance = np.linalg.norm(unknown_embedding - known_embedding)

        # More sensitive confidence calculation
        if distance < 0.4:
            confidence = (1 - distance) * 100  # High confidence
        elif distance < 0.5:
            confidence = (1 - distance * 1.5) * 100  # Medium confidence
        else:
            confidence = (1 - distance * 2.0) * 100  # Low confidence

        return max(0, confidence)  # Ensure confidence is non-negative

    def display_accuracy(self, frame, accuracy, name):
        # Display the accuracy and name on the frame
        cv2.putText(
            frame,
            f"Recognized {name} - Accuracy: {accuracy:.2f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def login(self):
        name, accuracy = self.recognize(self.most_recent_capture_arr)

        if name in ["unknown_person", "no_persons_found"]:
            util.msg_box(
                "Ups...", "Unknown user. Please register new user or try again."
            )
        else:
            util.msg_box(
                "Welcome back !", f"Welcome, {name}. Accuracy: {accuracy:.2f}%"
            )
            with open(self.log_path, "a") as f:
                f.write("{},{},in\n".format(name, datetime.datetime.now()))

    def logout(self):
        name, accuracy = self.recognize(self.most_recent_capture_arr)

        if name in ["unknown_person", "no_persons_found"]:
            util.msg_box(
                "Ups...", "Unknown user. Please register new user or try again."
            )
        else:
            util.msg_box(
                "Hasta la vista !", f"Goodbye, {name}. Accuracy: {accuracy:.2f}%"
            )
            with open(self.log_path, "a") as f:
                f.write("{},{},out\n".format(name, datetime.datetime.now()))

    def recognize(self, img):
        # Get the face embeddings for the unknown image
        embeddings_unknown = face_recognition.face_encodings(img)
        if len(embeddings_unknown) == 0:
            return "no_persons_found", 0.0

        embeddings_unknown = embeddings_unknown[0]

        # Load known faces from the database
        db_dir = sorted(os.listdir(self.db_dir))
        best_match_name = "unknown_person"
        highest_confidence = 0.0

        for file in db_dir:
            path_ = os.path.join(self.db_dir, file)
            with open(path_, "rb") as f:
                known_embedding = pickle.load(f)

            # Calculate accuracy
            accuracy = self.calculate_recognition_accuracy(
                embeddings_unknown, known_embedding
            )

            if accuracy > highest_confidence:
                highest_confidence = accuracy
                best_match_name = file[:-7]  # Remove .pickle extension

        # Display accuracy on the frame
        if highest_confidence > 50:  # Set a threshold for recognition
            self.display_accuracy(
                self.most_recent_capture_arr, highest_confidence, best_match_name
            )

        return best_match_name, highest_confidence

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window,
            "Accept",
            "green",
            self.accept_register_new_user,
        )
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window,
            "Try again",
            "red",
            self.try_again_register_new_user,
        )
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(
            self.register_new_user_window
        )
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, "Please, \ninput username:"
        )
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()

        # Check if the name is not empty
        if not name:
            util.msg_box("Error!", "Username cannot be empty.")
            return

        # Encode face and save the embeddings
        embeddings = face_recognition.face_encodings(self.register_new_user_capture)
        if embeddings:
            with open(
                os.path.join(self.db_dir, "{}.pickle".format(name)), "wb"
            ) as file:
                pickle.dump(embeddings[0], file)
            util.msg_box("Success!", "User was registered successfully!")
        else:
            util.msg_box("Error!", "No face detected. Please try again.")

        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
