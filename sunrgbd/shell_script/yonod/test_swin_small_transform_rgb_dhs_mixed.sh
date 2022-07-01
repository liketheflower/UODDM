# Test swin with rgb_dhs_mixed with sff checkpoints on RGB images                    
cfg=./work_dirs/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed_test_on_rgb.py
res_savepath=./test_logs_unshow/rgb_dhs_mixed_swin_s_on_rgb/
mkdir -p ${res_savepath}
for (( i=100; i>99; i-- ))
do
echo "epoch ${i}"
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed/epoch_${i}.pth
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox --gpu-collect  > ${res_savepath}test_swin_rgb_dhs_mixed_with_chess1_yonod_on_rgb_epoch_${i}_gpu.log
done

