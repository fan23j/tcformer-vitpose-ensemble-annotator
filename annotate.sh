python pose/tools/top_down_img_demo_with_yolov5.py \
    --vitpose-config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py \
    --vitpose-checkpoint ../two-stage-annotator/pretrained/wholebody.pth \
    --tcformer-config pose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_large_mta_coco_wholebody_384x288.py \
    --tcformer-checkpoint pose/weights/tcformer_large_coco-wholebody_384x288-b3b884c8_20220627.pth \
    --img-root /dfs/data/datasets/infinity_train \
    --out-img-root vis/infinity_train_vis \