### Motivation
This repo utilizes a simple ensemble pipeline that combines pose predictions of [TCFormer](https://github.com/zengwang430521/TCFormer) and [ViTPose](https://github.com/ViTAE-Transformer/ViTPose). Both approaches independently produce acceptable inference results, but annecdotally, it seems ViTPose performs better on the classic COCO keypoints (17) while TCFormer outperforms ViTPose on the foot keypoints provided in COCO Wholebody (6). This annotation tool can hopefully save a significant amount of annotation time for custom pose datasets. YoloV5l6 is used as the detector for top-down inference.

### Selection Criteria
I only apply a simple criteria that chooses the prediction result with higher confidence (for each keypoint). Although there is no guarantee that higher confidence translates to more accurate prediction, the ensemble results do (anectodally) seem more accurate.

## Setup / Installation
We use PyTorch 1.9.0 or NGC docker 21.06, and mmcv 1.3.9 for the experiments.
```bash
git clone https://github.com/fan23j/tcformer-vitpose-ensemble-annotator.git
cd tcformer-vitpose-ensemble-annotator
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
# RTX 30 series cards use:
# MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e .
cd ..
pip install -v -e .
```

After install the two repos, install timm and einops, i.e.,
```bash
pip install timm==0.4.9 einops
```

## Download pre-trained models
Download [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) COCO Wholebody pretrained model.

Download [TCFormer](https://github.com/zengwang430521/TCFormer) COCO Wholebody pretrained model.

## Run Annotation
Replace placeholders with paths to the pretrained models, custom dataset, and output folder in `annotate.sh`.

```bash
python pose/tools/top_down_img_demo_with_yolov5.py \
    --vitpose-config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py \
    --vitpose-checkpoint ${VITPOSE_WEIGHTS} \
    --tcformer-config pose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_large_mta_coco_wholebody_384x288.py \
    --tcformer-checkpoint ${TCFORMER_WEIGHTS} \
    --img-root ${CUSTOM_DATASET} \
    --out-img-root ${OUTPUT_FOLDER} \
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [YoloV5](https://github.com/ultralytics/yolov5)
* [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
* [TCFormer](https://github.com/zengwang430521/TCFormer)
* [mmpose](https://github.com/open-mmlab/mmpose)
</details>
