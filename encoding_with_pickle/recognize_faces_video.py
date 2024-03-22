# import the necessary packages
import face_recognition
import pickle
import time
import cv2
import os


# Initialisaties
video_path = os.path.join("..", "data", "pictures", "Friends.mp4")
output_path = os.path.join("output_result_videos", "test3.avi")
show_display = 1

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open('encodings_cnn.pickle', "rb").read())
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(video_path)
vs.set(cv2.CAP_PROP_FPS, 30)

writer = None
time.sleep(2.0)


def resize_frame(shape, desired_width):
    # convert the input frame from BGR to RGB then resize it to have
    # a smaller width (to speedup processing)
    height, width, _ = shape  # Get original image dimensions
    new_height = int(height * (desired_width / width))  # Calculate proportional height
    return desired_width, new_height

# loop over frames from the video file stream
def name_box(cv2, frame, left, top, right, bottom, name):
    # draw the predicted face name on the image
    cv2.rectangle(frame, (left, top), (right, bottom),
                  (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)


while vs.isOpened():
    # grab the frame from the threaded video stream
    _, frame = vs.read()

    new_size = resize_frame(frame.shape, 450)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_CUBIC)

    r = frame.shape[1] / float(rgb.shape[1])
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
                                            model='hog', number_of_times_to_upsample=2)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding, tolerance=0.6)
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            name_box(cv2, frame, left, top, right, bottom, name)


        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
        if writer is None and output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        # if the writer is not None, write the frame with recognized
        # faces to disk
        if writer is not None:
            writer.write(frame)

    # check to see if we are supposed to display the output frame to
        # the screen
        if show_display > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
