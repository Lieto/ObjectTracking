import cv2
import numpy as np
import copy


class MultiObjectTracker:

    def __init__(self, min_area = 400, min_shift2 = 5):
        self.object_roi = []
        self.object_box = []

        self.min_cnt_area = min_area
        self.min_shift2 = min_shift2

        # set up the termination criteria, either 100 iterations or move by at least 1 px
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)


    def advance_frame(self, frame, proto_objects_map):

        self.tracker = copy.deepcopy(frame)

        box_all = []

        box_all = self._append_boxes_from_saliency(proto_objects_map, box_all)

        box_all = self._append_boxes_from_meanshift(frame, box_all)

        if len(self.object_roi) == 0:
            group_thresh = 0
        else:
            group_thresh = 1

        box_grouped, _ = cv2.groupRectangles(box_all, group_thresh, 0.1)

        self._update_mean_shift_bookkeeping(frame, box_grouped)

        for (x, y, w, h) in box_grouped:
            cv2.rectangle(self.tracker, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        return self.tracker


    def _append_boxes_from_saliency(self, proto_objects_map, box_all):

        box_sal = []
        cnt_sal, _ = cv2.findContours(proto_objects_map, 1, 2)
        for cnt in cnt_sal:
            if cv2.contourArea(cnt) < self.min_cnt_area:
                continue

            box = cv2.boundingRect(cnt)
            box_all.append(box)

        return box_all

    def _append_boxes_from_meanshift(self, frame, box_all):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for i in xrange(len(self.object_roi)):
            roi_hist = copy.deepcopy(self.object_roi[i])
            box_old = copy.deepcopy(self.object_box[i])

            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, box_new = cv2.meanShift(dst, tuple(box_old), self.term_crit)
            self.object_box[i] = copy.deepcopy(box_new)

            (xo, yo, wo, ho) = box_old
            (xn, yn, wn, hn) = box_new

            co = [xo + wo/2, yo + ho/2]
            cn = [xn + wn/2, yn + hn/2]

            if (co[0]-cn[0])**2 + (co[1]-cn[1])**2 >= self.min_shift2:
                box_all.append(box_new)

        return box_all

    def _update_mean_shift_bookkeeping(self, frame, box_grouped):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        self.object_roi = []
        self.object_box = []

        for box in box_grouped:
            (x, y, w, h) = box
            hsv_roi = hsv[y:y + h, x:x + w]
            mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                               np.array((180., 255., 255.)))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            self.object_roi.append(roi_hist)
            self.object_box.append(box)








