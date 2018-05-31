#!/usr/bin/env python
'''Tracks cars out my window using OpenCV.

This works by keeping track of a running average for the scene and
subtracting the average from the current frame to find the parts
that are different/moving (like cars). The difference is processed
to find the bounding box of these car-sized changes.

Once the blobs are found, they are compared with previously-found
blobs so that we can track the progress of blobs across the image.
From those tracks we can compute speed and also count the number
of cars crossing the field of view in each direction.
'''

import uuid
from itertools import tee, izip
import argparse
import time

import cv2

from box import BoundingBox
from config import Config

def nothing(*args, **kwargs):
    " A helper function to use for OpenCV slider windows. "
    print args, kwargs


def get_frame(cap, conf):
    " Grabs a frame from the video capture and resizes it. "
    rval, frame = cap.read()
    if rval and False:
        (height, width) = frame.shape[:2]
        frame = cv2.resize(
            frame,
            (int(width * conf.resize_ratio), int(height * conf.resize_ratio)),
            interpolation=cv2.INTER_CUBIC)
    return rval, frame


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    iter_a, iter_b = tee(iterable)
    next(iter_b, None)
    return izip(iter_a, iter_b)


def get_closest_blob(tracked_blobs, center, conf):
    """Find the closest previously detected blob."""
    # Sort the blobs we have seen in previous frames by pixel
    # distance from this one
    closest_blobs = sorted(
        tracked_blobs.values(), key=lambda b: cv2.norm(b['trail'][0], center))

    # Starting from the closest blob, make sure the blob in
    # question is in the expected direction
    for close_blob in closest_blobs:
        distance = cv2.norm(center, close_blob['trail'][0])

        # Check if the distance is close enough to "lock on"
        if distance > conf.blob_lockon_distance_px:
            continue

        # If it's close enough, make sure the blob was
        # moving in the expected direction
        expected_dir = close_blob['dir']
        if expected_dir == 'left' and close_blob['trail'][0][0] < center[0]:
            continue
        elif expected_dir == 'right' and close_blob['trail'][0][0] > center[0]:
            continue
        else:
            return close_blob
    return None


def add_blob_metadata(closest_blob, box, frame_num):
    """Add metadata to blob.
    Calculate speed, direction, etc.
    """
    prev_center = closest_blob['trail'][0]
    if box.center[0] < prev_center[0]:
        # It's moving left
        closest_blob['dir'] = 'left'
        closest_blob['bumper_x'] = box.bb_x
    else:
        # It's moving right
        closest_blob['dir'] = 'right'
        closest_blob['bumper_x'] = box.bb_x + box.bb_w

    # ...and we should add this centroid to the trail of
    # points that make up this blob's history.
    closest_blob['trail'].insert(0, box.center)
    closest_blob['last_seen'] = frame_num
    closest_blob['box'] = box
    return closest_blob


def get_moving_contours(frame, conf, avg=None):
    """Get a set of moving blobs from frame."""
    # Convert the frame to Hue Saturation Value (HSV) color space.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Only use the Value channel of the frame.
    (_, _, gray_frame) = cv2.split(hsv_frame)
    # Apply a blur to the frame to smooth out any instantaneous changes
    # like leaves glinting in sun or birds flying around.
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if avg is None:
        # Set up the average if this is the first time through.
        avg = gray_frame.copy().astype("float")
        return avg, []

    # Build the average scene image by accumulating this frame
    # with the existing average.
    cv2.accumulateWeighted(gray_frame, avg, conf.default_average_weight)
    # cv2.imshow("average", cv2.convertScaleAbs(avg))

    # Compute the grayscale difference between the current grayscale frame
    # and the average of the scene.
    diff_frame = cv2.absdiff(gray_frame, cv2.convertScaleAbs(avg))
    # cv2.imshow("difference", diff_frame)

    # Apply a threshold to the difference: any pixel value above the
    # sensitivity value will be set to 255 and any pixel value below
    # will be set to 0.
    _, threshold_img = cv2.threshold(diff_frame, conf.threshold_sensitivity,
                                     255, cv2.THRESH_BINARY)
    threshold_img = cv2.dilate(threshold_img, None, iterations=2)
    # cv2.imshow("threshold", threshold_img)

    # Find contours aka blobs in the threshold image.
    _, contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    # Filter out the blobs that are too small to be considered cars.
    return avg, [c for c in contours if cv2.contourArea(c) > conf.blob_size]


def find_or_make_blob(contour, tracked_blobs, frame, frame_num, conf):
    """Find a blob that matches a contour.
    Otherwise create a new one.
    """
    # Find the bounding rectangle and center for each blob
    box = BoundingBox(*cv2.boundingRect(contour))

    ## Optionally draw the rectangle around the blob on the frame
    # that we'll show in a UI later
    cv2.rectangle(frame, box.top_left, box.bottom_right, (0, 0, 255),
                  conf.line_thickness)

    # Look for existing blobs that match this one
    closest_blob = None
    if tracked_blobs:
        closest_blob = get_closest_blob(tracked_blobs, box.center, conf)

    # If we found a blob to attach this blob to, we should
    # do some math to help us with speed detection
    if closest_blob:
        return add_blob_metadata(closest_blob, box, frame_num)

    # If we didn't find a blob, let's make a new one and add it to the list
    return {
        'id': str(uuid.uuid4())[:8],
        'first_seen': frame_num,
        'last_seen': frame_num,
        'dir': None,
        'bumper_x': None,
        'trail': [box.center],
        'box': box,
    }


def draw_blob_info(tracked_blobs, frame, conf):
    """Draw info about blobs onto the frame."""
    for blob in tracked_blobs.itervalues():
        box = blob['box']
        for (prev_blob, new_blob) in pairwise(blob['trail']):
            cv2.circle(frame, prev_blob, 3, (255, 0, 0), conf.line_thickness)

            if blob['dir'] == 'left':
                cv2.line(frame, prev_blob, new_blob, (255, 255, 0),
                         conf.line_thickness)
            else:
                cv2.line(frame, prev_blob, new_blob, (0, 255, 255),
                         conf.line_thickness)

            # bumper_x = blob['bumper_x']
            # if bumper_x:
            #     cv2.line(frame, (bumper_x, 100), (bumper_x, 500),
            #              (255, 0, 255), 3)

            cv2.rectangle(frame, box.top_left, box.bottom_right, (0, 0, 255),
                          conf.line_thickness)
            cv2.circle(frame, box.center, 10, (0, 255, 0), conf.line_thickness)
    return frame


def prune_tracked_blobs(tracked_blobs, frame_num, conf):
    """Prune out the blobs that haven't been seen in some amount of time."""
    for id_ in tracked_blobs.keys():
        blob = tracked_blobs[id_]
        if frame_num - blob['last_seen'] > conf.blob_track_timeout:
            print "Removing expired track {}".format(blob['id'])
            del tracked_blobs[id_]
    return tracked_blobs


def pause(frame_time, conf, interesting):
    """Pause to maintain target frame rate."""
    speed = conf.focus_speed if interesting else conf.speed
    time.sleep(max(0, (1.0 / (speed * conf.frame_rate)) - (time.time() - frame_time)))
    return time.time()


def get_speed(blob, conf, mph=True):
    """Calculate average speed of a blob across the ROI."""
    if len(blob['trail']) < 2:
        return None
    start_x, end_x = blob['trail'][0][0], blob['trail'][-1][0]
    start_num, end_num = blob['first_seen'], blob['last_seen']
    px_per_ft = float(conf.roi[2]) / conf.real_distance
    feet_traveled = abs(start_x - end_x) / px_per_ft
    seconds_elapsed = float(end_num - start_num) / conf.frame_rate
    if mph:
        return (feet_traveled / seconds_elapsed) * (3600 / 5280.0)
    # return speed in ft/s
    return feet_traveled / seconds_elapsed


def track(path, roi, conf):
    """Run tracking on a given input."""

    ## Set up a video capture device
    # (number for webcam, filename for video file input)
    # vc = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(path)

    if conf.display:
        cv2.namedWindow("preview")
    # cv2.namedWindow("threshold")
    # cv2.namedWindow("difference")
    # cv2.namedWindow("average")
    # cv2.cv.SetMouseCallback("preview", nothing)

    # A variable to store the running average.
    avg = None
    # A list of "tracked blobs".
    tracked_blobs = {}

    frame_num = 0
    frame_time = time.time()
    interesting = None
    while True:
        frame_num += 1
        # Grab the next frame from the camera or video file
        grabbed, frame = get_frame(cap, conf)
        if not grabbed or (interesting and
                           interesting + conf.blob_track_timeout < frame_num):
            break
        avg, contours = get_moving_contours(frame, conf, avg)

        for contour in contours:
            blob = find_or_make_blob(contour, tracked_blobs, frame, frame_num,
                                     conf)
            if blob['box'].center in roi:
                if conf.verbose and blob['id'] not in tracked_blobs:
                    print 'New blob detected in ROI. Current # of blobs:', len(
                        tracked_blobs) + 1
                tracked_blobs[blob['id']] = blob
                interesting = frame_num

        tracked_blobs = prune_tracked_blobs(tracked_blobs, frame_num, conf)

        # Draw the ROI
        cv2.rectangle(frame, roi.top_left, roi.bottom_right, 0)

        # Draw information about the blobs on the screen
        frame = draw_blob_info(tracked_blobs, frame, conf)

        # Show the image from the camera (along with all the lines and annotations)
        # in a window on the user's screen.
        if conf.display:
            cv2.imshow("preview", frame)
            frame_time = pause(frame_time, conf, interesting)

        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            break

    for blob in tracked_blobs.itervalues():
        print path, get_speed(blob, conf)

    cv2.destroyAllWindows()


def main():
    """Run the tracker."""
    parser = argparse.ArgumentParser(
        description='Calculate speed of objects in a video.')
    parser.add_argument('input', nargs='*', help='Video file to analyze')
    parser.add_argument(
        '--config', '-c', default='settings.yaml', help='path to config file')
    parser.add_argument(
        '--display',
        '-d',
        action='store_true',
        default=False,
        help='Show video display')
    parser.add_argument(
        '--speed',
        '-s',
        type=float,
        default=1.0,
        help='Playback speed'
    )
    parser.add_argument(
        '--focus-speed',
        '-f',
        dest='focus_speed',
        type=float,
        default=0.25,
        help='Playback speed when objects are in ROI'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        default=False,
        help='Print extra output')

    args = parser.parse_args()
    conf = Config(args.config)
    conf.update('display', args.display)
    conf.update('speed', args.speed)
    conf.update('focus_speed', args.focus_speed)
    conf.update('verbose', args.verbose)

    # Get region of interest
    roi = BoundingBox(*conf.roi)
    print 'ROI:', roi.top_left, roi.bottom_right

    for input_path in args.input:
        track(input_path, roi, conf)


if __name__ == '__main__':
    main()
