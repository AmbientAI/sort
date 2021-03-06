"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

# TODO: Figure out how to install numba and uncomment this and @jit on iou method below.
# Not using jit may have perf impact. Don't know if its significant.
# from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import collections


TrackedObject = collections.namedtuple(
    'TrackedObject',
    ['bbox',                # numpy array with shape [4]
     'tracked_id',          # tracked ID (1-indexed)
     'object_class',        # object class name
     # index into the original list of dets or -1 if no matched detections
     'original_index'],
    verbose=False)


class CostFunction:
    IOU = 'iou'
    L2 = 'l2'

# @jit
def iou(bb_test,bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

def l2(bb_test,bb_gt):
    center_test = [(bb_test[0] + bb_test[2])/2.0, (bb_test[1] + bb_test[3])/2.0]
    center_gt = [(bb_gt[0] + bb_gt[2])/2.0, (bb_gt[1] + bb_gt[3])/2.0]

    avg_width = (bb_gt[2] - bb_gt[0] + bb_test[2] - bb_test[0])/2.0

    #negative sign because original cost function iou returns the negative cost
    return  -np.linalg.norm(np.array(center_test) - np.array(center_gt))/avg_width

def assignment_cost(bb_test,bb_gt,cost_function=CostFunction.IOU):

    return eval(cost_function)(bb_test,bb_gt)

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    #scale is just area
    r = w/float(h)
    return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox, obj_class):
        """
        Initialises a tracker using initial bounding box and object class.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hit_streak = 0
        self.obj_class = obj_class

    def update(self,bbox):
      """
      Updates the state vector with observed bbox.
      """
      self.time_since_update = 0
      self.history = []
      self.hit_streak += 1
      self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0

        self.kf.predict()

        # NB(@vikesh): Maintain hit streak even if time_since_update > 0
        # This is necessary to for good smoothing. Otherwise, when the ID is re-acquired,
        # the hit_streak is 0 and the caller does not include this box in the return value
        # if(self.time_since_update>0):
        #   self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,
                                     trackers,
                                     threshold=0.3,
                                     cost_function=CostFunction.IOU):
    """
    Assigns detections to tracked object (both represented as bounding boxes),
    aware of object class.

    Args:
      detections: list of 3-tuples of (bbox, confidence, object class)
      trackers: list of 3-tuples, same structure as detections
      threshold: threshold for match based on cost function
      cost_function: iou or l2

    Returns:
      3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    cost_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    # Assigns cost based on class match first, then based on the provided
    # cost_function. If classes don't match, the linear_assignment will likely
    # not match them (and if they do, it will be filtered out later). Note that
    # in this code, HIGHER cost is BETTER.
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if det[2] != trk[2]:
                if cost_function == CostFunction.IOU:
                    # For IOU, 0 is the "worst" IOU
                    cost_matrix[d, t] = 0.0
                if cost_function == CostFunction.L2:
                    # For L2, lower = worse L2
                    # TODO: replace this with -inf, but this breaks
                    # linear_assignment, temporarily using -99
                    cost_matrix[d, t] = -99
            else:
                # Note: currently confidence (det[1]) is not used.
                cost_matrix[d, t] = assignment_cost(
                        det[0], trk[0], cost_function=cost_function)

    matched_indices = linear_assignment(-cost_matrix)
    unmatched_detections = []

    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []

    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    #filter out matched with low IOU or class mismatch
    matches = []
    for m in matched_indices:
        c = cost_matrix[m[0], m[1]]
        if detections[m[0]][2] != trackers[m[1]][2]:
            # Class mismatch
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
            continue

        # TODO: I think if using L2, c is always negative so it will never be >
        # threshold (currently 20 in YML file), so detections and trackers will
        # never be unmatched by distance. Fix this and set a new threshold
        # value.
        if ((cost_function == CostFunction.IOU and c < threshold) or
            (cost_function == CostFunction.L2 and c > threshold)):
            # Low IOU / high distance
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
            continue

        # Valid match, add to the final matches we return.
        matches.append(m.reshape(1,2))

    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self,max_age=5,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self, dets, threshold=0.3, cost_function=CostFunction.IOU):
    """
    Params:
      dets: a list of 3-length tuples, each corresponding to a detected entity,
        elements in the tuple are: (bbox, confidence, object class), e.g.
        [([1.0, 1.0, 2.0, 2.0], 0.9, 'person'),
         ([3.0, 3.0, 4.0, 4.0], 0.8, 'car'), ...]

    Requires: this method must be called once for each frame even with empty detections.

    Returns:
        list of TrackedObjects

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = []  # same structure as dets
    to_del = []
    for t, tracker in enumerate(self.trackers):
        pos = tracker.predict()[0]
        trk_cls = tracker.obj_class
        if np.any(np.isnan(pos)):
            to_del.append(t)
        else:
            trks.append((pos[:4], 0, trk_cls))

    for t in reversed(to_del):
        self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, threshold=threshold,
            cost_function=cost_function)

    # Maintain assocations of tracker to det. If no association, this array
    # has -1
    associations = [-1 for _ in  self.trackers]

    # update matched trackers with assigned detections
    for t, trk in enumerate(self.trackers):
        if t not in unmatched_trks:
            # d_idx = np.array([i1, i2, i3])
            det_match_idx = np.nonzero(matched[:,1]==t)[0][0]
            det_idx = matched[det_match_idx, 0]
            # TODO(chris): replace this print with assert when we are confident
            # assert dets[det_idx][2] == trk.obj_class
            if dets[det_idx][2] != trk.obj_class:
                print('[ERROR] tracking class does not match detection class')
            trk.update(dets[det_idx][0])
            # We can do this because trackers are only added beyond
            # this point. So this index will be valid.
            associations[t] = det_idx

        # If the trk is unmatched, there is no update to be done and the
        # association should be -1

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i][0], dets[i][2])
        # Attach class to new trackers
        self.trackers.append(trk)
        associations.append(i)

    i = len(self.trackers) - 1
    ret = []
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        # If the tracker is valid, add trivially if unmatched (for smoothing)
        # or validate hits
        if ((trk.time_since_update < self.max_age) and
            (i in unmatched_trks or
             trk.hit_streak >= self.min_hits or
             self.frame_count <= self.min_hits)):
            ret.append(TrackedObject(
                bbox=d, tracked_id=trk.id + 1,
                object_class=trk.obj_class,
                original_index=associations[i]))

        # Remove dead tracklet
        if trk.time_since_update > self.max_age:
            self.trackers.pop(i)
            # Don't need to pop from associations because it is created from
            # the tracker list at each frame.
        i -= 1

    return ret
