"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time
import cv2

import numpy as np
import torch
from tqdm import tqdm
import natsort
import glob

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def check_input():
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # for detection results
    if len(args.detfile):
        if os.path.isfile(args.detfile):
            detfile = args.detfile
            return 'detfile', detfile
        else:
            raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]

        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info():
#    print('===========================> Finish Model Running.')
#    if (args.save_img or args.save_video) and not args.vis_fast:
#        print('===========================> Rendering remaining images in the queue...')
#        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    pass

def loop():
    n = 0
    while True:
        yield n
        n += 1

#Customize our specific application
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
#    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
#    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
#    print("RETURNING IOU: " + str(iou))
    # return the intersection over union value
    return min(iou * 2, 1)

similarity_threshold = 0.5 # Threshold for defense to be considered successful (arbitrary, just depends on the accuracy of groundtruth)
suppress_writer = False
only_write_failures = True

def run_image_evaluation(image_file, gt_box):
    mode = 'image'
    input_source = [image_file]

    # Load detection loader
    det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize) #for now still using args
    det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if args.pose_track:
        tracker = Tracker(tcfg, args)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()
    # pose_model is the model
    
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = args.qsize
    if not suppress_writer:
        writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start() #Disable writer for now

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        success = -1
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                #save tensor
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    if not suppress_writer:
                        writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                if args.pose_track:
                    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                hm = hm.cpu()
                if not suppress_writer:
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                #Compare with original
                maxIoU = -1
                boxes_arr = cropped_boxes.cpu().detach().numpy()
                for inner_box in boxes_arr:
                    tempScore = max(bb_intersection_over_union(inner_box, gt_box), bb_intersection_over_union(gt_box, inner_box))
                    if tempScore > maxIoU:
                        maxIoU = tempScore
                success = maxIoU
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

                break

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info()
        if not suppress_writer:
            while(writer.running()):
                time.sleep(1)
#                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()
        det_loader.stop()
        return success
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            if not suppress_writer:
                while(writer.running()):
                    time.sleep(1)
#                    print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
                writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            if not suppress_writer:
                writer.terminate()
                writer.clear_queues()
            det_loader.clear_queues()

def get_corresponding_text_file(filename):
    parts = filename.split('\\')
    numPart = parts[len(parts) - 1].split('.')[0]
    parts[len(parts) - 1] = numPart + '.txt'
    return "\\".join(parts)

def get_filename(filename):
    parts = filename.split('\\')
    numPart = parts[len(parts) - 1].split('.')[0]
    return numPart


def read_gt_box(filename):
    f = open(filename, 'r')
    box_str = f.readline()
    f.close()
    box_str_arr = box_str[1:-1].split(' ')
    gt_box = []
    for item in box_str_arr:
        try:
            gt_box.append(float(item))
        except ValueError:
            pass
    #reproc bbox
    gt_box[2] = gt_box[2] + gt_box[0]
    gt_box[3] = gt_box[3] + gt_box[1]
    return gt_box

if __name__ == "__main__":
    out_path = 'C:\\Users\\raymo\\Downloads\\AlphaPose\\out'

    image_files = glob.glob('C:\\Users\\raymo\\Downloads\\Processed Attacks\\Tracker Hijacking\\Human\\added\\*.jpg')
    numSuccess = 0
    numFailure = 0
    iterator = 0
    
    for img in image_files:
        result = run_image_evaluation(img, read_gt_box(get_corresponding_text_file(img)))
        if not suppress_writer:
            if not only_write_failures or (only_write_failures and result < similarity_threshold and result > 0):
                temp_im = cv2.imread(os.path.join('C:\\Users\\raymo\\Downloads\\AlphaPose\\out\\vis', get_filename(img) + '.jpg'))
                box = read_gt_box(get_corresponding_text_file(img))
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                cv2.rectangle(temp_im, (round(x),round(y)), (round(w),round(h)), (0, 255, 255), 2)
                cv2.putText(temp_im, str(result), (round(x)-10,round(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imwrite(os.path.join('C:\\Users\\raymo\\Downloads\\AlphaPose\\out\\vis', get_filename(img) + '.jpg'), temp_im)
                del temp_im
            elif only_write_failures and result >= similarity_threshold:
                os.remove(os.path.join('C:\\Users\\raymo\\Downloads\\AlphaPose\\out\\vis', get_filename(img) + '.jpg'))
            
        if(result >= similarity_threshold):
            numSuccess += 1
        elif result >= 0:
            numFailure += 1
        iterator += 1
        print("---------------------")
        print("Job " + str(iterator))
        print("Successes: " + str(numSuccess))
        print("Failures: " + str(numFailure))
    

    out_file = "defense_results.txt"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("DONE")
    print("Successes: " + str(numSuccess))
    print("Failures: " + str(numFailure))
    #Write to file
    f = open(os.path.join(out_path, out_file), "w")
    f.write("Successes: " + str(numSuccess) + "\n")
    f.write("Failures: " + str(numFailure) + "\n")
    f.close()
