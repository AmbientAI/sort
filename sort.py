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
  def __init__(self, bbox, obj_class=None):
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

def associate_detections_to_trackers(
        detections, trackers, threshold=0.3, cost_function=CostFunction.IOU):
    """
    Assigns detections to tracked object (both represented as bounding boxes), aware of object class

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    cost_matrix = np.zeros((len(detections[0]),len(trackers[0])),dtype=np.float32)

    for d,det in enumerate(detections[0]):
        for t,trk in enumerate(trackers[0]):
            if detections[1][d] != trackers[1][t]:
                if cost_function == CostFunction.IOU:
                    # For IOU, 0 is the "worst" IOU
                    cost_matrix[d, t] = 0.0
                if cost_function == CostFunction.L2:
                    # For L2, lower = worse L2
                    # TODO: replace this with -inf, but this breaks
                    # linear_assignment, temporarily using -999
                    cost_matrix[d, t] = -999
            else:
                cost_matrix[d, t] = assignment_cost(det, trk, cost_function=cost_function)

    matched_indices = linear_assignment(-cost_matrix)

    unmatched_detections = []

    for d,det in enumerate(detections[0]):
        if (d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    unmatched_trackers = []

    for t,trk in enumerate(trackers[0]):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        c = cost_matrix[m[0],m[1]]
        # TODO: I think if using L2, c is always negative so it will never be >
        # threshold (currently 20 in YML file), so detections and trackers will
        # never be unmatched.
        if ((cost_function == CostFunction.IOU and c < threshold) or
            (cost_function == CostFunction.L2 and c > threshold)):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
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
      dets: a tuple of detection boxes and object classes,
            dets[0] is a numpy matrix [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...],
            dets[1] is a list of object classes associated with each detection.
    Requires: this method must be called once for each frame even with empty detections.

    Returns the a similar array as dets[0], where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    trks_classes = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trks_classes.append(self.trackers[t].obj_class)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
      trks_classes.pop(t)
    # trackers_with_classes is same structure as dets
    trackers_with_classes = (trks, trks_classes)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trackers_with_classes, threshold=threshold,
            cost_function=cost_function)

    # Maintain assocations to det. If no association, this array has -1
    associations = [-1 for _ in  self.trackers]

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        # d_idx = np.array([i1, i2, i3])
        d_idx = np.where(matched[:,1]==t)[0]
        d = matched[d_idx, 0]
        trk.update(dets[0][d,:][0])
        # We can do this because trackers are only added beyond
        # this point. So this index will be valid.
        associations[t] = d_idx[0]

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[0][i,:], dets[1][i])
        # Attach class to new trackers
        self.trackers.append(trk)
        associations.append(i)

    i = len(self.trackers) - 1
    ret_to_dets = []

    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        # If the tracker is valid, add trivially if unmatched (for smoothing) or validate hits
        if ((trk.time_since_update < self.max_age) and
            (i in unmatched_trks or
             trk.hit_streak >= self.min_hits or
             self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
          # NB: This may be -1 if tracker was unmatched
          ret_to_dets.append(associations[i])

        # Remove dead tracklet
        if trk.time_since_update > self.max_age:
          self.trackers.pop(i)

        i -= 1

    if(len(ret)>0):
      return np.concatenate(ret), ret_to_dets
    return np.empty((0,5)), ret_to_dets

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  sequences = ['PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','ETH-Bahnhof','ETH-Sunnyday','ETH-Pedcross2','KITTI-13','KITTI-17','ADL-Rundle-6','ADL-Rundle-8','Venice-2']
  args = parse_args()
  display = args.display
  phase = 'train'
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32,3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()

  if not os.path.exists('output'):
    os.makedirs('output')

  for seq in sequences:
    mot_tracker = Sort() #create instance of the SORT tracker
    seq_dets = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
    with open('output/%s.txt'%(seq),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:,0]==frame,2:7]
        dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          ax1 = fig.add_subplot(111, aspect='equal')
          fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase,seq,frame)
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq+' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
            ax1.set_adjustable('box-forced')

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  if(display):
    print("Note: to get real runtime results run without the option: --display")
