<<comment
# Test swin with rgb2dhs on RGB images
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs/epoch_69.pth
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox  > ./test_logs_unshow/test_swin_rgb_yonod_at_epoch100_unshow.log
comment
# Test swin with rgb2dhs on DHS images from the first 50 epoch
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs.py
res_savepath=./test_logs_unshow/rgb2dhs_on_dhs_first50epochs/
mkdir -p ${res_savepath}
for (( i=1; i<50; i++ ))
do
echo "epoch ${i}"
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from/epoch_${i}.pth
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox  > ${res_savepath}test_swin_rgb2dhs_yonod_on_dhs_first_50epochs_epoch_${i}.log
done

# Test swin with rgb2dhs on RGB images from the second 50 epoch
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py
res_savepath=./test_logs_unshow/rgb2dhs_on_rgb_second50epochs/
mkdir -p ${res_savepath}
for (( i=1; i<50; i++ ))
do
echo "epoch ${i}"
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs/epoch_${i}.pth
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox  > ${res_savepath}test_swin_rgb2dhs_yonod_on_rgb_second_50epochs_epoch_${i}.log
done
