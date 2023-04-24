import logging
import numpy as np
from typing import Dict
import torch
from torch import nn
import math
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
import pickle

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, make_stage
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from .keypoint_head import build_keypoint_head, keypoint_rcnn_inference, keypoint_rcnn_loss
from .mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss

import soft_renderer as sr

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-regioni computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)

def smooth_l1_loss(pred, targets, beta=2.8):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    """
    diff = pred - targets

    dis_trans = torch.norm(diff, dim=1)

    inbox_idx = torch.tensor(dis_trans <= 2.8, dtype=torch.float32).cuda()
    outbox_idx = torch.tensor(dis_trans > 2.8, dtype=torch.float32).cuda()

    in_pow_diff = 0.5 * torch.pow(diff, 2) / beta
    in_loss = in_pow_diff.sum(dim=1) * inbox_idx

    out_abs_diff = torch.abs(diff)
    out_loss = (out_abs_diff.sum(dim=1) - beta / 2) * outbox_idx

    loss = in_loss + out_loss
    N = loss.size(0)
    loss = loss.view(-1).sum(0) / N
    return loss

def euler_angles_to_rotation_matrix(car_rotation, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = car_rotation[:,0], car_rotation[:,1], car_rotation[:,2]

    rollMatrix = torch.tensor([[
        [1, 0, 0],
        [0, math.cos(roll[i]), -math.sin(roll[i])],
        [0, math.sin(roll[i]), math.cos(roll[i])]] for i in range(car_rotation.shape[0])])
    

    pitchMatrix = torch.tensor([[
        [math.cos(pitch[i]), 0, math.sin(pitch[i])],
        [0, 1, 0],
        [-math.sin(pitch[i]), 0, math.cos(pitch[i])]] for i in range(car_rotation.shape[0])])

    yawMatrix = torch.tensor([[
        [math.cos(yaw[i]), -math.sin(yaw[i]), 0],
        [math.sin(yaw[i]), math.cos(yaw[i]), 0],
        [0, 0, 1]] for i in range(car_rotation.shape[0])])

    R = torch.matmul(torch.matmul(yawMatrix, pitchMatrix), rollMatrix)
    return R

def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals):
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection).squeeze(1)
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            #print('======num class:', self.num_classes)
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()

@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        self._init_mask_head(cfg)
        self._init_keypoint_head(cfg)
        self._init_3d_head(cfg)
        self._init_3d_mesh(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        # fmt: off
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution                        = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales                            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio                           = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type                              = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_3d_head(self, cfg):
        """
        init the weights for 3d pose and car cls prediction head
        """
        self.relu = nn.ReLU(inplace=True)
        self.camera_intrisic = [2304.54786556982, 2305.875668062, 1686.23787612802, 1354.98486439791]
        self.kpts_dim = 66

        self.fuse_roi_feature_conv = nn.Conv2d(256, 256, 3, 2, 1)
        self.fuse_roi_feature_conv_1 = nn.Conv2d(256, 256, 3, 2, 1)
        self.fuse_roi_feature_conv_2 = nn.Conv2d(256, 128, 3, 2, 1)

        self.fuse_heatmap_conv = nn.Conv2d(self.kpts_dim, self.kpts_dim, 5, 3, 1)
        self.fuse_heatmap_conv_1 = nn.Conv2d(self.kpts_dim, self.kpts_dim, 5, 3, 1)
        self.fuse_heatmap_conv_2 = nn.Conv2d(self.kpts_dim, self.kpts_dim, 5, 3, 1)
        self.fuse_heatmap_conv_3 = nn.Conv2d(self.kpts_dim, self.kpts_dim, 3, 2, 1)

        self.fuse_box_pos = nn.Linear(4, 100)
        self.fuse_box_pos_1 = nn.Linear(100, 100)
        
        self.fuse_kpts_pos = nn.Linear(198, 300)
        self.fuse_kpts_pos_1 = nn.Linear(300, 300)
 
        self.regress_translation = nn.Linear(128 + 100 + 300, 3)
        self.regress_rotation = nn.Linear(128 + self.kpts_dim + 300, 3)

        self.regress_car_params_0 = nn.Linear(128 + self.kpts_dim, 10)
        self.regress_car_params_1 = nn.Linear(128 + self.kpts_dim, 5)
        self.regress_car_params_2 = nn.Linear(128 + self.kpts_dim, 5)
        self.regress_car_params_3 = nn.Linear(128 + self.kpts_dim, 8)
        self.regress_car_cluster_type = nn.Linear(128 + self.kpts_dim, 4)

        self.regress_car_cls = nn.Linear(128 + self.kpts_dim, self.num_classes)

        # weight init
        weight_init.c2_msra_fill(self.fuse_roi_feature_conv)
        weight_init.c2_msra_fill(self.fuse_roi_feature_conv_1)
        weight_init.c2_msra_fill(self.fuse_roi_feature_conv_2)

        weight_init.c2_msra_fill(self.fuse_heatmap_conv)
        weight_init.c2_msra_fill(self.fuse_heatmap_conv_1)
        weight_init.c2_msra_fill(self.fuse_heatmap_conv_2)
        weight_init.c2_msra_fill(self.fuse_heatmap_conv_3)

        weight_init.c2_xavier_fill(self.fuse_box_pos)
        weight_init.c2_xavier_fill(self.fuse_box_pos_1)
        weight_init.c2_xavier_fill(self.fuse_kpts_pos)
        weight_init.c2_xavier_fill(self.fuse_kpts_pos_1)
        weight_init.c2_xavier_fill(self.regress_translation)
        weight_init.c2_xavier_fill(self.regress_rotation)

        weight_init.c2_xavier_fill(self.regress_car_params_0)
        weight_init.c2_xavier_fill(self.regress_car_params_1)
        weight_init.c2_xavier_fill(self.regress_car_params_2)
        weight_init.c2_xavier_fill(self.regress_car_params_3)
        weight_init.c2_xavier_fill(self.regress_car_cluster_type)

        weight_init.c2_xavier_fill(self.regress_car_cls)

    def _init_3d_mesh(self, cfg):
        """
        init the car mesh basis and load the pca components
        """
        template_path = './merge_mean_car_shape/'
        car_mesh_path = './car_deform_result/'
        car_kpt_mapping_path = './kpts_mapping/'

        self.mesh_0_vertices = sr.Mesh.from_obj(template_path + 'merge_mean_car_model_0.obj').vertices
        self.mesh_1_vertices = sr.Mesh.from_obj(template_path + 'merge_mean_car_model_1.obj').vertices
        self.mesh_2_vertices = sr.Mesh.from_obj(template_path + 'merge_mean_car_model_2.obj').vertices
        self.mesh_3_vertices = sr.Mesh.from_obj(template_path + 'merge_mean_car_model_3.obj').vertices

        self.eigen_basis_0 = torch.from_numpy(np.load('./pca_components/new_merge_0_components.npy').astype('float32')).cuda() 
        self.eigen_basis_1 = torch.from_numpy(np.load('./pca_components/new_merge_1_components.npy').astype('float32')).cuda() 
        self.eigen_basis_2 = torch.from_numpy(np.load('./pca_components/new_merge_2_components.npy').astype('float32')).cuda() 
        self.eigen_basis_3 = torch.from_numpy(np.load('./pca_components/new_merge_3_components.npy').astype('float32')).cuda() 
        
        self.car_gt_meshes = [sr.Mesh.from_obj(car_mesh_path + str(i) + '.obj').vertices for i in range(79)]
        self.kpts_mapping = torch.tensor(np.load(car_kpt_mapping_path + 'kpts_mapping.npy').astype(np.int)).cuda()
 
        self.intrinsic = torch.tensor(np.load('./camera_intrinsic/camera_intrinsic.npy'), requires_grad=False).cuda()
 
        #print('mesh 0 vertices:', self.mesh_0_vertices)
        #print('mesh 0 vertices:', self.mesh_0_vertices.shape)
        #print('eigen base 0:', self.eigen_basis_0)
        #print('eigen base 0:', self.eigen_basis_0.shape)

    def forward(self, images, features, proposals, curr_iter, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]
        # start inference 
        if not self.training:
            pred_instances, selected_boxes, selected_roi_features, selected_heatmap, selected_kpt_pos = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self._forward_3d_pose_inference(selected_roi_features, selected_boxes, selected_kpt_pos, selected_heatmap, pred_instances)

            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features_init = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features_init)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        # start inference 
        if not self.training:
            self.test_score_thresh = 0.3
            #self.test_nms_thresh = 0.5

            pred_instances, _indexes = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )

            selected_boxes = outputs.predict_boxes()[0][_indexes[0],:]
            selected_roi_features = box_features_init[_indexes[0],:,:,:].clone()
            del box_features_init

            pred_boxes = [x.pred_boxes for x in pred_instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            selected_keypoint_logits_heatmap = self.keypoint_head(keypoint_features)
            #print('select keypoint logits shape:', selected_keypoint_logits_heatmap.shape)
            selected_keypoint_res = keypoint_rcnn_inference(selected_keypoint_logits_heatmap, pred_instances)

            return pred_instances, selected_boxes, selected_roi_features, selected_keypoint_logits_heatmap, selected_keypoint_res

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        # start inference 
        if not self.training:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        # start inference 
        if not self.training:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_res = keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def _forward_3d_pose_inference(self, roi_feature, box_pos, keypoint_pos, heatmap, instances_per_image):
        """
        inference code for the 3d pose and shape prediction branch
        """
        roi_feature = F.relu(self.fuse_roi_feature_conv(roi_feature))
        roi_feature = F.relu(self.fuse_roi_feature_conv_1(roi_feature))
        roi_feature = F.relu(self.fuse_roi_feature_conv_2(roi_feature))
        roi_feature = roi_feature.view(roi_feature.shape[0], -1)

        heatmap = F.relu(self.fuse_heatmap_conv(heatmap))
        heatmap = F.relu(self.fuse_heatmap_conv_1(heatmap))
        heatmap = F.relu(self.fuse_heatmap_conv_2(heatmap))
        heatmap = self.fuse_heatmap_conv_3(heatmap).view(heatmap.shape[0], -1)

        new_box_pos = box_pos.clone()
        new_box_pos[:,0] = (0.5*(box_pos[:,0]+box_pos[:,2]) - self.camera_intrisic[2] * (1500 / 2710)) / self.camera_intrisic[0]
        new_box_pos[:,1] = (0.5*(box_pos[:,1]+box_pos[:,3]) - self.camera_intrisic[3] * (1500 / 2710)) / self.camera_intrisic[1]
        new_box_pos[:,2] = (box_pos[:,2] - box_pos[:,0]) / self.camera_intrisic[0]
        new_box_pos[:,3] = (box_pos[:,3] - box_pos[:,1]) / self.camera_intrisic[1]

        new_box_pos = self.relu(self.fuse_box_pos(new_box_pos))
        new_box_pos = self.fuse_box_pos_1(new_box_pos)
  
        new_keypoint_pos = keypoint_pos.clone()
        new_keypoint_pos[:,:,0] = (keypoint_pos[:,:,0] - self.camera_intrisic[2] * (1500 / 2710)) / self.camera_intrisic[0]
        new_keypoint_pos[:,:,1] = (keypoint_pos[:,:,1] - self.camera_intrisic[3] * (1500 / 2710)) / self.camera_intrisic[1]

        new_keypoint_pos = new_keypoint_pos.view(keypoint_pos.shape[0], -1)
        new_keypoint_pos = self.relu(self.fuse_kpts_pos(new_keypoint_pos))
        new_keypoint_pos = self.fuse_kpts_pos_1(new_keypoint_pos)

        translation_feature = torch.cat((roi_feature, new_box_pos, new_keypoint_pos), 1)
        rotation_feature = torch.cat((roi_feature, heatmap, new_keypoint_pos), 1)
        shape_feature = torch.cat((roi_feature, heatmap), 1)

        pred_trans = self.regress_translation(translation_feature)
        pred_rotation = self.regress_rotation(rotation_feature)

        car_params_0  = self.regress_car_params_0(shape_feature)
        car_params_1  = self.regress_car_params_1(shape_feature)
        car_params_2  = self.regress_car_params_2(shape_feature)
        car_params_3  = self.regress_car_params_3(shape_feature)
        car_cluster_types  = self.regress_car_cluster_type(shape_feature)
        car_cluster_probs = nn.Softmax()(car_cluster_types)

        shape_0_dim = self.mesh_0_vertices.shape[0]

        predict_vertices_0 = torch.bmm(car_cluster_probs[:,0:1].unsqueeze(1),(self.mesh_0_vertices.reshape(shape_0_dim, -1) + torch.mm(car_params_0, self.eigen_basis_0.reshape(10,-1))).unsqueeze(1)).squeeze(1)
        predict_vertices_1 = torch.bmm(car_cluster_probs[:,1:2].unsqueeze(1),(self.mesh_1_vertices.reshape(shape_0_dim, -1) + torch.mm(car_params_1, self.eigen_basis_1.reshape(5,-1))).unsqueeze(1)).squeeze(1)
        predict_vertices_2 = torch.bmm(car_cluster_probs[:,2:3].unsqueeze(1),(self.mesh_2_vertices.reshape(shape_0_dim, -1) + torch.mm(car_params_2, self.eigen_basis_2.reshape(5,-1))).unsqueeze(1)).squeeze(1)
        predict_vertices_3 = torch.bmm(car_cluster_probs[:,3:4].unsqueeze(1),(self.mesh_3_vertices.reshape(shape_0_dim, -1) + torch.mm(car_params_3, self.eigen_basis_3.reshape(8,-1))).unsqueeze(1)).squeeze(1)

        predict_vertices = (predict_vertices_0 + predict_vertices_1 + predict_vertices_2 + predict_vertices_3).reshape(car_params_0.shape[0],-1,3)
        
        instances_per_image[0].predict_trans = pred_trans
        instances_per_image[0].predict_rotation = pred_rotation
        instances_per_image[0].predict_vertices = predict_vertices

        return instances_per_image
