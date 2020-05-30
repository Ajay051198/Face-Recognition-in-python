""" in this file we define the code for recogonizing faces in images and videos
using face_recognition"""

#importing modules
import face_recognition
import os
import numpy as np
import cv2
import warnings
import argparse

TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'



warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default=None,
	help="path to image")
parser.add_argument("-v", "--video", default=None,
	help="path to video")
args = vars(parser.parse_args())

# defining path based on choice
if args['image'] != None:
	path = f"to_detect/{args['image']}"
elif args['video'] != None:
	path = f"to_detect/{args['video']}"
else:
    print('[INFO] Enter Live feed mode! ')
    print('[INFO] Press Q to exit Live feed mode!')

# laoding encodings and labels
known_faces = np.load("assets/face_encoding.npy")
known_names = np.load("assets/labels.npy")

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

def check_path(p):
    # check if path exists
    if not os.path.exists(p):
        print('[INFO] Could not find file at addrees [ {} ],\
         please retry with correct file name'.format(path))
        input('\n'+'-> Press enter to exit')
        exit()

if args['image'] != None:
    check_path(path)
    count = 0
    # Load image
    image = face_recognition.load_image_file(path)

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f"[INFO] Found {len(encodings)} face(s)")
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            count+=1

            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = name_to_color(match)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    print(f"[INFO] recogonized {count} face(s)")
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Result')

if args['video'] != None:
    check_path(path)
    vc = cv2.VideoCapture(path)

    while True:
        # Load frames
        count = 0
        ret, image = vc.read()

        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(image, model=MODEL)

        # Now since we know loctions, we can pass them to face_encodings as second argument
        # Without that it will search for faces once again slowing down whole process
        encodings = face_recognition.face_encodings(image, locations)

        # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
        print(f"[INFO] Found {len(encodings)} face(s)")
        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                count+=1

                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                # Get color by name using our fancy function
                color = name_to_color(match)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

        # Show image
        print(f"[INFO] recogonized {count} face(s)")
        cv2.imshow('Result', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

else:
    vc = cv2.VideoCapture(0)
    while True:
        # Load frames
        count = 0
        ret, image = vc.read()

        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(image, model=MODEL)

        # Now since we know loctions, we can pass them to face_encodings as second argument
        # Without that it will search for faces once again slowing down whole process
        encodings = face_recognition.face_encodings(image, locations)

        # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
        print(f"[INFO] Found {len(encodings)} face(s)")
        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                count+=1

                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                # Get color by name using our fancy function
                color = name_to_color(match)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

        # Show image
        print(f"[INFO] recogonized {count} face(s)")
        cv2.imshow('Result', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
