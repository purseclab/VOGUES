# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    #cfg.MODEL.FCOS.INFERENCE_TH = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x_apollo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        #default=0.2,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_corresponding_text_file(filename):
    parts = filename.split('\\')
    numPart = parts[len(parts) - 1].split('.')[0]
    parts[len(parts) - 1] = numPart + '.txt'
    return "\\".join(parts)

def get_filename(filename):
    parts = filename.split('\\')
    numPart = parts[len(parts) - 1].split('.')[0]
    return numPart

def read_gt_boxes(filename):
    #Pass in file
    f = open(filename, 'r')
    all_boxes_str = f.readlines()
    f.close()
    gt_boxes = []
    for box_str in all_boxes_str:
        box_str_arr = box_str.strip()[1:-1].split(' ')
        gt_box = []
        for item in box_str_arr:
            try:
                gt_box.append(float(item))
            except ValueError:
                pass
        #reproc bbox
        gt_box[2] = gt_box[2] + gt_box[0]
        gt_box[3] = gt_box[3] + gt_box[1]
        gt_boxes.append(gt_box)
        
    return gt_boxes

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    #Pass in your stuff
    vids_glob = '*.mov'
    
    numSuccesses = 0
    numFailures = 0
    jobNum = 0
    skipJob = 5
    for dvid in glob.iglob(vids_glob):
        print(dvid)
        jobNum += 1
    #        if(jobNum <= skipJob):
    #            continue
        print("----------------------")
        print("Job " + str(jobNum))
        hasValidResult = False
        isSuccess = False
        
        video = cv2.VideoCapture(dvid)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(dvid)
        
        groundtruth_list = read_gt_boxes(get_corresponding_text_file(dvid))
        
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".avi"
            else:
                output_fname = args.output
            if(os.path.isfile(output_fname)):
                os.remove(output_fname) #just yeet it lmao
            output_file = cv2.VideoWriter(
                filename=output_fname,
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame, result in tqdm.tqdm(demo.run_on_video(video, groundtruth_list), total=num_frames):
            if(result > 0):
                hasValidResult= True
            if(result == 1):
                isSuccess = True # We need only one success to make it
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
        
        #Final gut check
        if not hasValidResult or isSuccess:
            #delete the output video if not a failure case
#            os.remove(output_fname)
            pass
#        if not hasValidResult:
#            os.remove(output_fname)
            
        if not hasValidResult:
            print("-Successes-: " + str(numSuccesses))
            print("-Failures-: " + str(numFailures))
            continue
        
        if isSuccess:
            numSuccesses += 1
        else:
            numFailures += 1
        print("Successes: " + str(numSuccesses))
        print("Failures: " + str(numFailures))
    
    out_file = "defense_results_car.txt"
    percentage_success = 0.0
    if(numSuccesses + numFailures > 0):
        percentage_success = (numSuccesses/(numFailures + numSuccesses)) * 100.0
    print("\n\nDONE")
    print("Successes: " + str(numSuccesses))
    print("Failures: " + str(numFailures))
    print("Percent:" + str(percentage_success))
    print("Jobs: " + str(jobNum))
    #Write to file
    f = open(os.path.join(args.output, out_file), "w")
    f.write("Successes: " + str(numSuccesses) + "\n")
    f.write("Failures: " + str(numFailures) + "\n")
    f.write("Percent:" + str(percentage_success) + "\n")
    f.write("Jobs: " + str(jobNum) + "\n")
    f.close()