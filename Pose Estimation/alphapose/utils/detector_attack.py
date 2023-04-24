import os
import sys
from threading import Thread
from queue import Queue

import cv2
import numpy as np

import torch
import torch.multiprocessing as mp

from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder

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

class DetectionLoader():
    def __init__(self, input_source, detector, cfg, opt, mode='image', batchSize=1, queueSize=128):
        self.cfg = cfg
        self.opt = opt
        self.mode = mode
        self.device = opt.device

        if mode == 'image':
            self.img_dir = opt.inputpath
            self.imglist = [os.path.join(self.img_dir, im_name.rstrip('\n').rstrip('\r')) for im_name in input_source]
            self.datalen = len(input_source)
        elif mode == 'video':
            stream = cv2.VideoCapture(input_source)
            assert stream.isOpened(), 'Cannot capture source'
            self.path = input_source
            self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
            self.fps = stream.get(cv2.CAP_PROP_FPS)
            self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': self.frameSize}
            stream.release()

        self.detector = detector
        self.batchSize = batchSize
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)

        # initialize the queue used to store data
        """
        image_queue: the buffer storing pre-processed images for object detection
        det_queue: the buffer storing human detection results
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        if opt.sp:
            self._stopped = False
            self.image_queue = Queue(maxsize=queueSize)
            self.det_queue = Queue(maxsize=10 * queueSize)
            self.pose_queue = Queue(maxsize=10 * queueSize)
        else:
            self._stopped = mp.Value('b', False)
            self.image_queue = mp.Queue(maxsize=queueSize)
            self.det_queue = mp.Queue(maxsize=10 * queueSize)
            self.pose_queue = mp.Queue(maxsize=10 * queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        if self.mode == 'image':
            image_preprocess_worker = self.start_worker(self.image_preprocess)
        elif self.mode == 'video':
            image_preprocess_worker = self.start_worker(self.frame_preprocess)
        # start a thread to detect human in images
        image_detection_worker = self.start_worker(self.image_detection)
        # start a thread to post process cropped human image for pose estimation
        image_postprocess_worker = self.start_worker(self.image_postprocess)

        return [image_preprocess_worker, image_detection_worker, image_postprocess_worker]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        if self.opt.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.det_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def image_preprocess(self):
        for i in range(self.num_batches):
            imgs = []
            orig_imgs = []
            im_names = []
            im_dim_list = []
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                if self.stopped:
                    self.wait_and_put(self.image_queue, (None, None, None, None))
                    return
                im_name_k = self.imglist[k]

                # expected image shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(im_name_k)
                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)
                orig_img_k = cv2.cvtColor(cv2.imread(im_name_k), cv2.COLOR_BGR2RGB) # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
                im_dim_list_k = orig_img_k.shape[1], orig_img_k.shape[0]

                imgs.append(img_k)
                orig_imgs.append(orig_img_k)
                im_names.append(os.path.basename(im_name_k))
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                imgs = torch.cat(imgs)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                # im_dim_list_ = im_dim_list

            self.wait_and_put(self.image_queue, (imgs, orig_imgs, im_names, im_dim_list))

    def frame_preprocess(self):
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), 'Cannot capture source'

        for i in range(self.num_batches):
            imgs = []
            orig_imgs = []
            im_names = []
            im_dim_list = []
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                (grabbed, frame) = stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed or self.stopped:
                    # put the rest pre-processed data to the queue
                    if len(imgs) > 0:
                        with torch.no_grad():
                            # Record original image resolution
                            imgs = torch.cat(imgs)
                            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                        self.wait_and_put(self.image_queue, (imgs, orig_imgs, im_names, im_dim_list))
                    self.wait_and_put(self.image_queue, (None, None, None, None))
                    print('===========================> This video get ' + str(k) + ' frames in total.')
                    sys.stdout.flush()
                    stream.release()
                    return

                # expected frame shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(frame)

                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)

                im_dim_list_k = frame.shape[1], frame.shape[0]

                imgs.append(img_k)
                orig_imgs.append(frame[:, :, ::-1])
                im_names.append(str(k) + '.jpg')
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Record original image resolution
                imgs = torch.cat(imgs)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                # im_dim_list_ = im_dim_list

            self.wait_and_put(self.image_queue, (imgs, orig_imgs, im_names, im_dim_list))
        stream.release()

    def image_detection(self):
        outfile_max = "max_perturb.txt"
        outfile_avg = "avg_perturb.txt"
        for i in range(self.num_batches):
            imgs, orig_imgs, im_names, im_dim_list = self.wait_and_get(self.image_queue)
            strength = 80
            max_strength = 90
            god_box = None
            perturb = None
            threshold_to_break = 0.5
            attack_done = False
            if imgs is None or self.stopped:
                self.wait_and_put(self.det_queue, (None, None, None, None, None, None, None))
                return

            with torch.no_grad():
                get_next_batch = False
                while not attack_done:
                    print("str: " + str(strength))
                    # pad useless images to fill a batch, else there will be a bug
                    if perturb is not None:
                        for i in range(0, len(imgs)):
                            imgs[i] = np.clip(imgs[i] + perturb, 0, 255)
                    for pad_i in range(self.batchSize - len(imgs)):
                        imgs = torch.cat((imgs, torch.unsqueeze(imgs[0], dim=0)), 0)
                        im_dim_list = torch.cat((im_dim_list, torch.unsqueeze(im_dim_list[0], dim=0)), 0)
                    
                    #Prep perturbs
                    strength += 10
                    if strength > max_strength:
                        break
                    perturb = np.absolute(np.random.normal(0, 1, imgs[0].shape[0] * imgs[0].shape[1] * imgs[0].shape[2]).reshape(imgs[0].shape)) * strength
                    perturb = np.clip(perturb, 0, 255)
                    dets = self.detector.images_detection(imgs, im_dim_list)
                    if isinstance(dets, int) or dets.shape[0] == 0:
                        for k in range(len(orig_imgs)):
                            self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], None, None, None, None, None))
#                        get_next_batch = True
                        continue
                    if isinstance(dets, np.ndarray):
                        dets = torch.from_numpy(dets)
                    dets = dets.cpu()
                    boxes = dets[:, 1:5]
                    #Get the first box
                    special_box = boxes.cpu().detach().numpy()[0]
                    if god_box is None:
                        god_box = special_box
                    else:
                        if bb_intersection_over_union(god_box, special_box) < threshold_to_break:
                            #success
                            attack_done = True
                    
                    if(strength >= max_strength):
                        attack_done = True
                    scores = dets[:, 5:6]
                    if self.opt.tracking:
                        ids = dets[:, 6:7]
                    else:
                        ids = torch.zeros(scores.shape)
#                if get_next_batch:
#                    continue

            
            #First check if we did a thing
            if attack_done and perturb is not None:
                print("ATTACK DONE")
                #get statistics
                max_perturb = np.amax(perturb)
                avg_perturb = np.average(perturb)
                #Write to file
                if(os.path.exists(outfile_max)):
                    f = open(outfile_max, "r+")
                    f.write(str(max_perturb) + "\n")
                    f.close()
                else:
                    f = open(outfile_max, "w")
                    f.write(str(max_perturb) + "\n")
                    f.close()
                    
                if(os.path.exists(outfile_avg)):
                    f = open(outfile_avg, "r+")
                    f.write(str(avg_perturb) + "\n")
                    f.close()
                else:
                    f = open(outfile_avg, "w")
                    f.write(str(avg_perturb) + "\n")
                    f.close()
            else:
                if not attack_done:
                    print("attack not done")
                if perturb is None:
                    print("perturb none")
                

            for k in range(len(orig_imgs)):
                boxes_k = boxes[dets[:, 0] == k]
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], None, None, None, None, None))
                    continue
                inps = torch.zeros(boxes_k.size(0), 3, *self._input_size)
                cropped_boxes = torch.zeros(boxes_k.size(0), 4)
                
                if(perturb is None):
                    self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes))
                else:
                    #time for some transformation jutsu
                    rdim = (orig_imgs[k].shape[1], orig_imgs[k].shape[0])
                    if(orig_imgs[k].shape != perturb.shape):
                        print("orig")
                        print(orig_imgs[k].shape)
                        print("perturb")
                        print(perturb.shape)
                        perturb = np.reshape(perturb, (perturb.shape[1], perturb.shape[2], perturb.shape[0]))
                        perturb = cv2.resize(perturb, rdim, interpolation=cv2.INTER_CUBIC)
                    
                    self.wait_and_put(self.det_queue, (np.clip(orig_imgs[k] + perturb, 0, 255), im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes))

    def image_postprocess(self):
        for i in range(self.datalen):
            with torch.no_grad():
                (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.wait_and_get(self.det_queue)
                if orig_img is None or self.stopped:
                    self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
                    return
                if boxes is None or boxes.nelement() == 0:
                    self.wait_and_put(self.pose_queue, (None, orig_img, im_name, boxes, scores, ids, None))
                    continue
                # imght = orig_img.shape[0]
                # imgwidth = orig_img.shape[1]
                for i, box in enumerate(boxes):
                    inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                    cropped_boxes[i] = torch.FloatTensor(cropped_box)

                # inps, cropped_boxes = self.transformation.align_transform(orig_img, boxes)

                self.wait_and_put(self.pose_queue, (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes))

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def length(self):
        return self.datalen
