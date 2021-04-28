import cv2
import numpy as np
import torch
import os

from opts import opts
from tracker import FairTracker
from fairmot.utils.transformation import *
from fairmot.tracking_utils import visualization as vis
from test_utils import write_results

"""
This is the kernel of system and do the tracking task.
It can accept tow intput format. One is the image sequence direction, the other is a video file.
For the image sequence direction, it should be the format similar to the following:
    - DIRECTION "img1": the image set of all image files named like "000397"
    - INI FILE "seqinfo.ini": the file has some information about sequence, for instance:

        [Sequence]
        name=MOT17-01-FRCNN
        imDir=img1
        frameRate=30
        seqLength=450
        imWidth=1920
        imHeight=1080
        imExt=.jpg

And for the video file, it should be a file type which can be loaded by cv2.videocapture.
"""
class TrackingKernel:
    def init_kernel(self, frame_rate, image_size, target_size, opt):
        """ Create kernel only. """
        self.image_size = image_size
        self.target_size = target_size
        
        # tracker_init
        # setup the MOT_Tracker
        self.tracker = FairTracker(opt, frame_rate)

    def pre_processing(self, img0):
        img = np.array(img0)
        # Padded resize
        img, _, _, _ = letterbox(img, height=self.target_size[0], width=self.target_size[1])
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img
    
    def call_once(self, img0):
        """ Kernel Executor in a single step. Current only support for video file. """
        #### PRE PROCESSING ####
        img = self.pre_processing(img0)
        #### UPDATE TRACKER ####
        #### ********* ####
        ret_trks = []
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        #### TRACKING ####
        trks = self.tracker.update(blob, self.image_size)
        return trks

if __name__ == "__main__":
    opt = opts().init()
    input_file = '../data/demo.avi'
    show_image = False
    save_dir = 'outputs'
    result_filename = 'demo.txt'
    data_type = 'mot'
    
    vc = cv2.VideoCapture(input_file)
    frame_rate = vc.get(cv2.CAP_PROP_FPS)
    raw_shape_wh = (int(vc.get(3)), int(vc.get(4)))
    max_frame_idx = vc.get(cv2.CAP_PROP_FRAME_COUNT)

    raw_shape_hw = [raw_shape_wh[1], raw_shape_wh[0]]
    target_shape_hw = [608, 1088]

    tk = TrackingKernel()
    tk.init_kernel(frame_rate, raw_shape_hw, target_shape_hw, opt)
    results = []
    frame_idx = 1
    while frame_idx <= max_frame_idx:
        # Load image
        print("Processing frame %05d" % frame_idx)
        #### LOADING IMAGE ####
        _, img0 = vc.read()

        # Execute
        online_targets = tk.call_once(img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
            # save results
        results.append((frame_idx, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_idx)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_idx)), online_im)

        frame_idx += 1
    # save results
    write_results(result_filename, results, data_type)
