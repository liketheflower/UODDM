def extract_mAP50(path, fileprefix, linenumber):
    """
    Extract the mAP50 info
    ./../../test_logs_unshow/rgb2dhs_on_rgb_second50epochs/test_swin_rgb2dhs_yonod_on_rgb_second_50epochs_epoch_6.log

    """
    mAP50 = []
    for epoch in range(1, 51):
        fn = (
            path
            + fileprefix
            + str(epoch)
            + ".log"
        )
        file = open(fn, "r")
        lines = file.readlines()
        """
        2637  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.091

2638  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.189

2639  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.077
        for i, l in enumerate(lines):
            if "Average Precision" in l:print(i, l) 
        """
        mAP = float(lines[linenumber].split()[-1])
        mAP50.append(mAP)
    print(mAP50)


if __name__ == "__main__":
    path = "./../../test_logs_unshow/rgb2dhs_on_rgb_second50epochs/"
    fileprefix = "test_swin_rgb2dhs_yonod_on_rgb_second_50epochs_epoch_"
    extract_mAP50(path, fileprefix, 2638)
