#_base_ = './mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
_base_ = './mask_rcnn_swin-s-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed.py'
# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
