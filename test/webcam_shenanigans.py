import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

if __name__ == '__main__':

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        #color = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #color = cv.flip(frame, 1)

        width = int(128)
        height = int(128)
        dim = (width, height)
        original_dim = (frame.shape[1], frame.shape[0])

        # resize image
        resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        # another one
        reresized = cv.resize(resized, original_dim, interpolation=cv.INTER_AREA)

        # Display the resulting frame
        #print('Original Dimensions : ', frame.shape)
        cv.imshow('frame', frame)

        #print('Resized Dimensions : ', resized.shape)
        #cv.imshow('reresized', reresized)
        #cv.resizeWindow('resized', 640, 480)

        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def capture_frame():
    # Capture frame-by-frame
    ret, webcam_frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    else:
        return webcam_frame


def video_to_arr(video_path):
    vidcap = cv.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = [image]
    while success:
        try:
            success, video_frame = vidcap.read()
            video_dim = (128, 128)
            video_frame = cv.resize(video_frame, video_dim, interpolation=cv.INTER_AREA)
            frames.append(video_frame)
        except:
            print("Corrupted thing???")

    print("Converted video to " + str(len(frames)) + " frames")
    return frames


def disconnect():
    cap.release()
    cv.destroyAllWindows()

    pass
