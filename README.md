# [LayoutTransformer-Scene-Layout-Generation-with-Conceptual-and-Spatial-Diversity](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.pdf)
Cheng-Fu Yang*, Wan-Cyuan Fan*, Fu-En Yang, Yu-Chiang Frank Wang, "LayoutTransformer: Scene Layout Generation with Conceptual and Spatial Diversity", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

# LayoutTransformer 
Pytorch implementation for LT-Net. The goal is to generate scene layout with conceptual and spatial diversity.

### Overview
<img src="./figures/archi.png" width="940px" height="360px"/>

### Data
- COCO dataset
    - Download the annotations from [COCO](https://cocodataset.org/#download).
    - i.e., 2017 Train/Val annotations [241MB] and 2017 Stuff Train/Val annotations [1.1GB]
    - Extract the annotations to `data/coco/`
- VG-MSDN dataset
    - Download the VG-MSDN dataset from [VG-MSDN](https://drive.google.com/file/d/1WjetLwwH3CptxACrXnc1NCcccWUVDO76/view). (This dataset origins from [FactorizableNet](https://github.com/yikang-li/FactorizableNet))
    - Extract the annotations (i.e., all json files) to `data/vg_msdn/`

### Training
All code was developed and tested on Ubuntu 20.04 with Python 3.7 (Anaconda) and PyTorch 1.7.1.

#### Pre-train the Obj/Rel Rredictor
- Pre-train Predictor module for COCO dataset:
```
python3 train.py --cfg_path ./configs/coco/coco_pretrain.yaml
```
- Pre-train Predictor model for VG-MSDN dataset: 
```
python3 train.py --cfg_path ./configs/vg_msdn/vg_msdn_pretrain.yaml
```
#### Full module
- Train full model for COCO dataset:
```
python3 train.py --cfg_path ./configs/coco/coco_seq2seq_v9_ablation_4.yaml
```
- Train full model for VG-MSDN dataset: 
```
python3 train.py --cfg_path ./configs/vg_msdn/vg_msdn_seq2seq_v24.yaml
```

`*.yml` files include configuration for training and testing.


### Pretrained Model Weights (TODO) 
#### Obj/Rel Predictor 
- [COCO](https://github.com/davidhalladay/LayoutTransformer). Download and save it to `saved/coco_model_weights`
- [VG-MSDN](https://github.com/davidhalladay/LayoutTransformer). Download and save it to `saved/vg_model_weights`
#### LT-Net Full Model 
- [COCO](https://github.com/davidhalladay/LayoutTransformer). Download and save it to `saved/coco_model_weights`
- [VG-MSDN](https://github.com/davidhalladay/LayoutTransformer). Download and save it to `saved/vg_model_weights`

### Evaluation

TODO

### Citation

If you find this useful for your research, please use the following.

```
@InProceedings{Yang_2021_CVPR,
    author    = {Yang, Cheng-Fu and Fan, Wan-Cyuan and Yang, Fu-En and Wang, Yu-Chiang Frank},
    title     = {LayoutTransformer: Scene Layout Generation With Conceptual and Spatial Diversity},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3732-3741}
}
```

### Acknowledgements
This code borrows heavily from [Transformer](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py) repository. Many thanks.