import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import math

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.framework.resnet import resnet101
import cv2


class HKA(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        return u * attn

class Attention(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = HKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
class HKA2(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv_spatial(x)
        attn = self.conv1(attn)
        return u * attn

class Attention2(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = HKA2(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
class _hANMCL(nn.Module):
    def __init__(self, classes, attention_type, rpn_reduce_dim, rcnn_reduce_dim, gamma, n_way=2, n_shot=5, pos_encoding=True):
        super(_hANMCL, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_way = n_way
        self.n_shot = n_shot
        self.attention_type = attention_type
        self.channel_gamma = gamma
        self.rpn_reduce_dim = rpn_reduce_dim
        self.rcnn_reduce_dim = rcnn_reduce_dim
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # pooling or align
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        # few shot rcnn head
        self.pool_feat_dim = 1024
        self.rcnn_dim = 64
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        dim_in = self.pool_feat_dim
        

        self.rpn_unary_layer = nn.Linear(400, 1)
        init.normal_(self.rpn_unary_layer.weight, std=0.01)
        init.constant_(self.rpn_unary_layer.bias, 0)
        self.rcnn_unary_layer = nn.Linear(49, 1)
        init.normal_(self.rcnn_unary_layer.weight, std=0.01)
        init.constant_(self.rcnn_unary_layer.bias, 0)
        
        self.rpn_adapt_q1_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_q1_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_q1_layer.bias, 0)
        self.rpn_adapt_k1_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_k1_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_k1_layer.bias, 0)
        
        self.rpn_adapt_q2_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_q2_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_q2_layer.bias, 0)
        self.rpn_adapt_k2_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_k2_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_k2_layer.bias, 0)

        self.rpn_adapt_q3_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_q3_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_q3_layer.bias, 0)
        self.rpn_adapt_k3_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_k3_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_k3_layer.bias, 0)
        
        
        self.rcnn_adapt_q1_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_q1_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_q1_layer.bias, 0)
        self.rcnn_adapt_k1_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_k1_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_k1_layer.bias, 0)

        self.rcnn_adapt_q2_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_q2_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_q2_layer.bias, 0)
        self.rcnn_adapt_k2_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_k2_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_k2_layer.bias, 0)

        
        self.rcnn_adapt_q3_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_q3_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_q3_layer.bias, 0)
        self.rcnn_adapt_k3_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_k3_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_k3_layer.bias, 0)
        
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        if self.attention_type == 'concat':
            self.RCNN_rpn = _RPN(2048)
            self.rcnn_transform_layer = nn.Linear(2048, self.rcnn_dim)
        elif self.attention_type == 'product':
            self.RCNN_rpn = _RPN(1024)
            self.rcnn_transform_layer = nn.Linear(1024, self.rcnn_dim)
        
        self.output_score_layer = FFN(64* 49, dim_in)
        self.rcnn_transform_layer2 = nn.Linear(1024, 128)
        # positional encoding
        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pos_encoding_layer = PositionalEncoding()
            self.rpn_pos_encoding_layer = PositionalEncoding(max_len=400)
        
        self.attention = Attention()
        self.attention2 = Attention2()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, support_ims, all_cls_gt_boxes=None):
        if self.training:
            self.num_of_rois = cfg.TRAIN.BATCH_SIZE
        else:
            self.num_of_rois = cfg.TEST.RPN_POST_NMS_TOP_N 
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        base_feat = self.RCNN_base(im_data)
        base_feat = self.attention(base_feat)
        base_feat = self.attention2(base_feat)
        if self.training:
            support_ims = support_ims.view(-1, support_ims.size(2), support_ims.size(3), support_ims.size(4))
            support_feats = self.RCNN_base(support_ims)  # [B*2*shot, 1024, 20, 20]
            support_feats = self.attention(support_feats)
            support_feats = self.attention2(support_feats)
            

            support_feats = support_feats.contiguous().view(-1, self.n_way*self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :].contiguous()  # [B, shot, 1024, 20, 20]
            neg_support_feat = support_feats[:, self.n_shot:self.n_way*self.n_shot, :, :, :].contiguous()
            
            pos_support_feat_pooled = self.avgpool(pos_support_feat.view(-1, 1024, 20, 20))
            neg_support_feat_pooled = self.avgpool(neg_support_feat.view(-1, 1024, 20, 20))
            pos_support_feat_pooled = pos_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)  # [B, shot, 1024, 7, 7]
            neg_support_feat_pooled = neg_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)
            
            
        else:
            support_ims = support_ims.view(-1, support_ims.size(2),  support_ims.size(3),  support_ims.size(4)) 
            support_feats = self.RCNN_base(support_ims)
            support_feats = self.attention(support_feats)
            support_feats = self.attention2(support_feats)
            support_feats = support_feats.view(-1, self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :]
            pos_support_feat_pooled = self.avgpool(pos_support_feat.view(-1, 1024, 20, 20))
            pos_support_feat_pooled = pos_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)

        batch_size = pos_support_feat.size(0)
        feat_h = base_feat.size(2)
        feat_w = base_feat.size(3)
        support_mat = pos_support_feat.transpose(0, 1).view(self.n_shot, batch_size, 1024, -1).transpose(2, 3)
        query_mat = base_feat.view(batch_size, 1024, -1).transpose(1, 2)
        dense_support_feature = []

        q1_matrix = self.rpn_adapt_q1_layer(query_mat)  # [B, hw, 256]
        q1_matrix = q1_matrix - q1_matrix.mean(1, keepdim=True)
        
        q2_matrix = self.rpn_adapt_q2_layer(query_mat)  # [B, hw, 256]
        q2_matrix = q2_matrix - q2_matrix.mean(1, keepdim=True)
        
        q3_matrix = self.rpn_adapt_q3_layer(query_mat)  # [B, hw, 256]
        q3_matrix = q3_matrix - q3_matrix.mean(1, keepdim=True)
        
        for i in range(self.n_shot):
            if self.pos_encoding:
                single_s_mat = self.rpn_pos_encoding_layer(support_mat[i])  # [B, 400, 1024]
            else:
                single_s_mat = support_mat[i]
            
            k1_matrix = self.rpn_adapt_k1_layer(single_s_mat)  # [B, hw, 256]
            k1_matrix = k1_matrix - k1_matrix.mean(1, keepdim=True)
            k2_matrix = self.rpn_adapt_k2_layer(single_s_mat)  # [B, hw, 256]
            k2_matrix = k2_matrix - k2_matrix.mean(1, keepdim=True)
            
            k3_matrix = self.rpn_adapt_k3_layer(single_s_mat)  # [B, hw, 256]
            k3_matrix = k3_matrix - k3_matrix.mean(1, keepdim=True)

            support_adaptive_attention_weight1 = torch.bmm(q1_matrix, q2_matrix.transpose(1, 2)) / math.sqrt(self.rpn_reduce_dim) 
            support_adaptive_attention_weight1 = F.softmax(support_adaptive_attention_weight1, dim=2)
            
            support_adaptive_attention_weight2 = torch.bmm(k1_matrix, k2_matrix.transpose(1, 2)) / math.sqrt(self.rpn_reduce_dim) 
            support_adaptive_attention_weight2 = F.softmax(support_adaptive_attention_weight2, dim=2)

            support_adaptive_attention_weight3 = torch.bmm(q3_matrix, k3_matrix.transpose(1, 2)) / math.sqrt(self.rpn_reduce_dim) 
            support_adaptive_attention_weight3 = F.softmax(support_adaptive_attention_weight3, dim=2)
            
            support_adaptive_attention_feature1 = torch.bmm(support_adaptive_attention_weight1, query_mat)  # [B, hw, 1024]

            unary_term = self.rpn_unary_layer(support_adaptive_attention_weight2)  # [n_roi, 49, 1]
            unary_term = F.softmax(unary_term, dim=1)
            support_adaptive_attention_weight = support_adaptive_attention_weight3 + unary_term.transpose(1, 2)
            
            support_adaptive_attention_feature2 = torch.bmm(support_adaptive_attention_weight, single_s_mat)  # [B, hw, 1024]
            
            support_adaptive_attention_feature = torch.cat([support_adaptive_attention_feature1,
                                                             support_adaptive_attention_feature2],dim=2)
            dense_support_feature += [support_adaptive_attention_feature]
        
        dense_support_feature = torch.stack(dense_support_feature, 0).mean(0)  # [B, hw, 1024]
        dense_support_feature = dense_support_feature.transpose(1, 2).contiguous().view(batch_size, 1024*2, feat_h, feat_w)
        
        if self.attention_type == 'concat':
            correlation_feat = dense_support_feature
        elif self.attention_type == 'product':
            correlation_feat = base_feat * dense_support_feature
        
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(correlation_feat, im_info, gt_boxes, num_boxes)

        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        rois = Variable(rois)

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # rcnn head
        if self.training:
            bbox_pred, cls_prob, cls_score_all, correlation_feat2 = self.rcnn_head(pooled_feat, pos_support_feat_pooled)
            _, neg_cls_prob, neg_cls_score_all, neg_correlation_feat2 = self.rcnn_head(pooled_feat, neg_support_feat_pooled)
            cls_prob = torch.cat([cls_prob, neg_cls_prob], dim=0)
            cls_score_all = torch.cat([cls_score_all, neg_cls_score_all], dim=0)
            neg_rois_label = torch.zeros_like(rois_label)
            rois_label = torch.cat([rois_label, neg_rois_label], dim=0)
            fg_inds = (rois_label == 1).nonzero().squeeze(-1)
            roi = torch.stack((correlation_feat2[fg_inds],neg_correlation_feat2[fg_inds]),dim=0)
            roi = roi.permute(1,0,2)

        else:
            bbox_pred, cls_prob, cls_score_all, correlation_feat2 = self.rcnn_head(pooled_feat, pos_support_feat_pooled)

        # losses
        if self.training:
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            fg_inds = (rois_label == 1).nonzero().squeeze(-1)
            bg_inds = (rois_label == 0).nonzero().squeeze(-1)
            cls_score_softmax = torch.nn.functional.softmax(cls_score_all, dim=1)
            bg_cls_score_softmax = cls_score_softmax[bg_inds, :]
            bg_num_0 = max(1, min(fg_inds.shape[0] * 2, int(rois_label.shape[0] * 0.25)))
            bg_num_1 = max(1, min(fg_inds.shape[0], bg_num_0))
            _sorted, sorted_bg_inds = torch.sort(bg_cls_score_softmax[:, 1], descending=True)
            real_bg_inds = bg_inds[sorted_bg_inds]  # sort the real_bg_inds
            real_bg_topk_inds_0 = real_bg_inds[real_bg_inds < int(rois_label.shape[0] * 0.5)][:bg_num_0]  # pos support
            real_bg_topk_inds_1 = real_bg_inds[real_bg_inds >= int(rois_label.shape[0] * 0.5)][:bg_num_1]  # neg_support
            topk_inds = torch.cat([fg_inds, real_bg_topk_inds_0, real_bg_topk_inds_1], dim=0)
            RCNN_loss_cls = F.cross_entropy(cls_score_all[topk_inds], rois_label[topk_inds])
        else:
            RCNN_loss_cls = 0
            RCNN_loss_bbox = 0
            rois_label2=0
            roi=[]
            roi_temp=0
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, roi#, roi_temp

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def rcnn_head(self, pooled_feat, support_feat):
        # box regression
        bbox_pred = self.RCNN_bbox_pred(self._head_to_tail(pooled_feat))  # [B*128, 4]
        # classification
        n_roi = pooled_feat.size(0)
        support_mat = []
        query_mat = []
        batch_size = support_feat.size(0)
        for query_feat, target_feat in zip(pooled_feat.chunk(batch_size, dim=0), support_feat.chunk(batch_size, dim=0)):
            # query_feat [128, c, 7, 7], target_feat [1, shot, c, 7, 7]
            query_feat = self.attention(query_feat)
            target_feat = self.attention(target_feat.squeeze(0))
            query_feat = self.attention2(query_feat)
            target_feat = self.attention2(target_feat)
            target_feat = target_feat.unsqueeze(0)
            target_feat = target_feat.view(1, self.n_shot, 1024, -1).transpose(2, 3)  # [1, shot, 49, c]
            target_feat = target_feat.repeat(query_feat.size(0), 1, 1, 1)  # [128, shot, 49, c]
            query_feat = query_feat.view(query_feat.size(0), 1024, -1).transpose(1, 2)  # [128, 49, c]
            if self.pos_encoding:
                target_feat = self.pos_encoding_layer(target_feat.view(-1, 49, 1024)).view(-1, self.n_shot, 49, 1024)
                query_feat = self.pos_encoding_layer(query_feat)
            support_mat += [target_feat]
            query_mat += [query_feat]
        support_mat = torch.cat(support_mat, 0).transpose(0, 1)  # [shot, B*128, 49, c]
        query_mat = torch.cat(query_mat, 0)  # [B*128, 49, c]
        dense_support_feature = []
        q1_matrix = self.rcnn_adapt_q1_layer(query_mat)
        q1_matrix = q1_matrix - q1_matrix.mean(1, keepdim=True)
        q2_matrix = self.rcnn_adapt_q2_layer(query_mat)
        q2_matrix = q2_matrix - q2_matrix.mean(1, keepdim=True)
        q3_matrix = self.rcnn_adapt_q3_layer(query_mat)
        q3_matrix = q3_matrix - q3_matrix.mean(1, keepdim=True)  
        
        for i in range(self.n_shot):
            single_s_mat = support_mat[i]
            k1_matrix = self.rcnn_adapt_k1_layer(single_s_mat)
            k1_matrix = k1_matrix - k1_matrix.mean(1, keepdim=True)
            k2_matrix = self.rcnn_adapt_k2_layer(single_s_mat)
            k2_matrix = k2_matrix - k2_matrix.mean(1, keepdim=True)
            k3_matrix = self.rcnn_adapt_k3_layer(single_s_mat)
            k3_matrix = k3_matrix - k3_matrix.mean(1, keepdim=True)

            support_adaptive_attention_weight1 = torch.bmm(q1_matrix, q2_matrix.transpose(1, 2)) / math.sqrt(self.rcnn_reduce_dim) 
            support_adaptive_attention_weight1 = F.softmax(support_adaptive_attention_weight1, dim=2)

            
            support_adaptive_attention_weight2 = torch.bmm(k1_matrix, k1_matrix.transpose(1, 2)) / math.sqrt(self.rcnn_reduce_dim) 
            support_adaptive_attention_weight2 = F.softmax(support_adaptive_attention_weight2, dim=2)            

            support_adaptive_attention_weight3 = torch.bmm(q3_matrix, k3_matrix.transpose(1, 2)) / math.sqrt(self.rcnn_reduce_dim) 
            support_adaptive_attention_weight3 = F.softmax(support_adaptive_attention_weight3, dim=2)
            
            support_adaptive_attention_feature1 = torch.bmm(support_adaptive_attention_weight1, query_mat)  # [B, hw, 1024]

            unary_term = self.rcnn_unary_layer(support_adaptive_attention_weight2)  # [n_roi, 49, 1]
            unary_term = F.softmax(unary_term, dim=1)
            
            support_adaptive_attention_weight = support_adaptive_attention_weight3 + unary_term.transpose(1, 2)

            support_adaptive_attention_feature2 = torch.bmm(support_adaptive_attention_weight, single_s_mat)  # [B, hw, 1024]
            
            support_adaptive_attention_feature = torch.cat([support_adaptive_attention_feature1,
                                                             support_adaptive_attention_feature2],dim=2)
            
            dense_support_feature += [support_adaptive_attention_feature]
        dense_support_feature = torch.stack(dense_support_feature, 0).mean(0)  # [n_roi, 49, 4096]
        
        if self.attention_type == 'concat':
            correlation_feat = dense_support_feature
        elif self.attention_type == 'product':
            correlation_feat = query_mat * dense_support_feature           
            
        correlation_feat2 = correlation_feat # [n_roi, 49, 128]
        correlation_feat2 = F.normalize(self.avgpool2(correlation_feat2.transpose(1,2).view(-1,2048,7,7)).view(-1,2048))
        f1, f2 = torch.split(correlation_feat2, [1024, 1024], dim=1)
        f1 = self.rcnn_transform_layer2(f1)
        f2 = self.rcnn_transform_layer2(f2)
        
        correlation_feat2 = torch.cat([f1,f2],dim=1)
        correlation_feat = self.rcnn_transform_layer(correlation_feat)  # [B*128, 49, rcnn_d]
        cls_score = self.output_score_layer(correlation_feat.view(n_roi, -1))
        cls_prob = F.softmax(cls_score, 1)  # [B*128, 1]

        return bbox_pred, cls_prob, cls_score, correlation_feat2

class FFN(nn.Module):
    def __init__(self, in_channel, hidden, drop_prob=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_channel, hidden)
        self.linear2 = nn.Linear(hidden, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model=1024, max_len=49):
        super(PositionalEncoding, self).__init__()
        #print('positionencoding')
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / float(d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = Variable(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        x = x + self.pe.to(x.device)
        return x


class hANMCL(_hANMCL):
    def __init__(self, classes, attention_type, rpn_reduce_dim=256, rcnn_reduce_dim=256, gamma=0.1, 
                num_layers=101, pretrained=False, num_way=2, num_shot=5, pos_encoding=True):
        self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        _hANMCL.__init__(self, classes, attention_type, rpn_reduce_dim, rcnn_reduce_dim, gamma, 
                                    n_way=num_way, n_shot=num_shot, pos_encoding=pos_encoding)

    def _init_modules(self):
        resnet = resnet101()
        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet. (base -> top -> head)
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
        self.RCNN_top = nn.Sequential(resnet.layer4)  # 1024 -> 2048
        # build rcnn head
        self.RCNN_bbox_pred = nn.Linear(2048, 4)

        # Fix blocks 
        for p in self.RCNN_base[0].parameters(): p.requires_grad=False
        for p in self.RCNN_base[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)  # [128, 2048]
        return fc7
