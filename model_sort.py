'''
Wrapper class ModelSort around Sort class.

A SORT object maintains all tracking info (trajectories) for 
a given stream/video

A ModelSort object gathers the SORT objects associated to several 
streams/videos

Note on vocabulary: 'tracker' can denote
    - a single position in a trajectory
    - a single trajectory
        - we will denote those as traj_trackers
    - the instance that maintains all tracking info for a given stream
        - meaning of self.trackers of model_maskrcnn.py
        - we will denote those as stream_trackers
    - a tracking algorithm

Default parameters for SORT are stored in sort.py.
However best practice is to specify them each time in model config yml 
file.

Owner: @ferdinand
'''

from ambient.third_party.sort import sort
import yaml

class ModelSort():

    def __init__(self, cost_function, threshold, max_age, min_hits):
        """
        Parameters of tracking algorithm are shared across all streams

        Args: see sort.py
        """

        self.streams_trackers       = {}    # {stream_id:SORT object}
        self.max_age                = max_age
        self.min_hits               = min_hits
        self.cost_function          = cost_function
        self.threshold              = threshold

    @staticmethod
    def conf(config_path):
        """
        Static method used to instantiate a ModelSort object with parameters 
        specified in model config file at config_path

        Notes:
            - the config file is NOT the evaluation config file, it is the model
              config file, eg ~/ambient/src/dlcore/configs/latest.yml
            - the streams_trackers will be instantiated at inference, in a similar
              way as in model_maskrcnn.py

        Returns ModelSort object
        """

        # Load config file
        with open(config_path) as fin:
            cfg = yaml.load(fin)

        # Extract tracking parameters from config
        tracking_kwargs = cfg['engine']['model']['args']['tracking_kwargs']

        max_age         = tracking_kwargs.get('max_age', sort.DEFAULT_MAX_AGE)               
        min_hits        = tracking_kwargs.get('min_hits', sort.DEFAULT_MIN_HITS)             
        cost_function   = tracking_kwargs.get('cost_function', sort.DEFAULT_COST_FUNCTION)   
        threshold       = tracking_kwargs.get('threshold', sort.DEFAULT_THRESHOLD)

        return ModelSort(cost_function, threshold, max_age, min_hits)


    def add_stream_tracker(self,stream_identifier):
        """
        Adds a stream_tracker to the streams_trackers dictionary.
        All stream_trackers have same parameters.
        """

        self.streams_trackers[stream_identifier] = sort.Sort(
                    max_age   = self.max_age,
                    min_hits  = self.min_hits)