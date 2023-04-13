python pose/tools/top_down_img_demo_with_yolov5.py \
    --vitpose-config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py \
    --vitpose-checkpoint ${VITPOSE_WEIGHTS} \
    --tcformer-config pose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_large_mta_coco_wholebody_384x288.py \
    --tcformer-checkpoint ${TCFORMER_WEIGHTS} \
    --img-root ${CUSTOM_DATASET} \
    --out-img-root ${OUTPUT_FOLDER} \
