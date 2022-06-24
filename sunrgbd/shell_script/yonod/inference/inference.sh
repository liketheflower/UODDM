<<comment
# yonod swin-s on rgb
path=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed/
cfg=${path}mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed_test_on_rgb.py
checkpoint=${path}epoch_100.pth
show_dir=./vis/yonod_swin_s_rgb/
mkdir $show_dir
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox --show --show-dir ${show_dir} --gpu-collect > ./test_logs_show/test_swin_s_yonod_on_rgb_at_epoch100_show.log

# yonod swin-s on rgb
path=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed/
cfg=${path}mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed.py
checkpoint=${path}epoch_100.pth
show_dir=./vis/yonod_swin_s_dhs/
mkdir $show_dir
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox --show --show-dir ${show_dir} --gpu-collect > ./test_logs_show/test_swin_s_yonod_on_dhs_at_epoch100_show.log
comment
# rgb only swin-t on rgb
path=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from/
cfg=${path}mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py
checkpoint=${path}epoch_50.pth
show_dir=./vis/yonod_swin_t_rgb_only/
mkdir $show_dir
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox --show --show-dir ${show_dir} --gpu-collect > ./test_logs_show/test_swin_t_rgbonly_at_epoch100_show.log
