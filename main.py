import cv2
import numpy as np
import argparse

from os import path

from Saliency import  Saliency
from MultiObjectTracking import MultiObjectTracker

parser = argparse.ArgumentParser(description='Object tracking with saliency map and mean-shift tracking')

parser.add_argument('--filename', type=str, help='name of the video file', default='./soccer.avi')
parser.add_argument('--roi', required=True, nargs='+',
                    help='upper left and lower right of coordinates for bounding box')
args = parser.parse_args()

def main():

    if path.isfile(args.filename):
        print args.filename
        video = cv2.VideoCapture(args.filename)
    else:
        print('File: "' + args.filename + '" does not exist.')
        raise SystemExit

    # initialize tracker
    mot = MultiObjectTracker()

    while True:
        success, img = video.read()

        if success:
            if args.roi:
                # grab some meaningful ROI
                roi = ((int(args.roi[0]), int(args.roi[1])), (int(args.roi[2]), int(args.roi[3])))
                print args.roi[0]
                print args.roi[2]
                print args.roi[1]
                print args.roi[3]
                img = img[roi[0][0]: roi[1][0], roi[0][1]: roi[1][1]]

            # generate saliency1 map
            sal = Saliency(img, use_numpy_fft=False, gauss_kernel=(3, 3))
            #mag = sal.plot_magnitude()
            #sal.plot_power_spectrum()
            #objects = sal.get_proto_objects()
            cv2.imshow('original', img)
            cv2.imshow('saliency', sal.get_saliency_map())
            cv2.imshow('objects' , sal.get_proto_objects(use_otsu=False))
            cv2.imshow('tracker', mot.advance_frame(img, sal.get_proto_objects(use_otsu=False)))

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break


if __name__ == "__main__":

    main()
