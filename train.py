""" We will be training the face_recognition modeule and saving the encodings
and the respective labels in this file """

#importing modules
import face_recognition
import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

KNOWN_FACES_DIR = 'known_faces'
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

print('\n[INFO] ENCODING IMAGES..\n.')
face_encodings = []
labels = []
files_skipped = []
skip = 0
trained = 0

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        try:
            # Load an image
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

            # Get 128-dimension face encoding
            # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
            encoding = face_recognition.face_encodings(image)[0]
            print(f"[#] {name}/{filename} -> done")
            trained += 1
        except:
            print(f"[X] cound not detect a face / cound not load file {filename}, please check file ")
            files_skipped.append(f"{name}/{filename}")
            skip += 1
            continue
        # Append encodings and name
        face_encodings.append(encoding)
        labels.append(name)


print('\n'*20)
face_encodings = np.array(face_encodings)
labels = np.array(labels)
np.save('assets/face_encoding.npy', face_encodings)
np.save('assets/labels.npy', labels)
print('x'*46, "RESULT",'x'*46 )
print("[INFO] FILES SKIPPED:")
for each in files_skipped:
    print(f"-> {each}")
print('-'*100)
print(f"[INFO] {skip} files skipped ")
print(f"[INFO] trained on {trained} files")
print('x'*100)
