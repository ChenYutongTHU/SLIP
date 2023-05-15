# 中文CLIP

## 训练
本方法分为两步：（1）中文文本编码器的蒸馏学习（2）图像-文本的比较学习

### 1. Bilingual distillation learning
```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port 2990 --use_env main.py \
    --training_mode distillation \
    --dataset_distillation aic,coco_sbu_vg,cc3m,cc12m,tsl2019,newsv16,wikititles,wikimatrix,thunmt \
    --output-dir experiments/outputs/distill/Vit_l_14_bs512-8-wd0.1-lr5e-4-ep25_aic-csv-cc15m-tsl2019-newsv16-wiki2m-thunmt \
    --model TRIPLET \
    --lr 5e-4 --wd 0.1 --epochs 25 \
    --batch-size 512 \
    --model_cfg_path experiments/configs_distill/Vit_L_14.yaml \
    --need_only_text --read_tsv 
```
该训练在8张GTX3090上运行。如果使用A100，可增大batch_size，减少节点数

### 2. Vision-language contrastive learning
