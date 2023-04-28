# Test swin with rgb_dhs_mixed with sff checkpoints on RGB images                    
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py
res_savepath=./test_logs_unshow/rgb_dhs_mixed_with_sff_on_rgb/
mkdir -p ${res_savepath}
for (( i=100; i>0; i-- ))
do
echo "epoch ${i}"
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed_with_sff/epoch_${i}.pth
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox  > ${res_savepath}test_swin_rgb_dhs_mixed_with_rff_yonod_on_rgb_epoch_${i}.log
done

