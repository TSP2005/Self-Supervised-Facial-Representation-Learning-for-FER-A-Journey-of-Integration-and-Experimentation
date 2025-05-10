For Training:
nohup env CUDA_VISIBLE_DEVICES=5,6 \
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=29500 \
main.py \
    --arch FRAB \
    --backbone resnet18_encoder \
    --dataset vggface2 \
    --data-root ./data/VGG-Face2-crop/train \
    --lr 0.9 \
    -b 128 \
    --wd 0.000001 \
    --epochs 30 \
    --cos \
    --warmup-epoch 10 \
    --workers 16 \
    --enc-m 0.996 \
    --norm SyncBN \
    --lewel-loss-weight 0.5 \
    --mask_type="attn" \
    --num_proto 4 \
    --teacher_temp 0.04 \
    --loss_w_cluster 0.1 \
    --amp \
    --save-dir ./ckpts \
    --save-freq 50 \
    --print-freq 100 \
    --eval-freq 5 \
    --seed 23456 \
    --proj-dim 256 \
    --lewel-num-heads 2 \
    --diversity_lambda 0.001 \
    --multiproc \
> t1.log 2>&1 &

For Fine tuning :
python3 launch.py --device=${DEVICES} --launch main_fer.py \
    -a resnet18 \
    --dataset rafdb --data-root ${FER_DATA_ROOT} \
    --lr 0.0002 --lr_head 0.0002 --optimizer adamw --weight-decay 0.05 --scheduler cos \
    --finetune \
    --epochs 100 --batch-size 256 \
    --amp \
    --workers 16 \
    --eval-freq 5 \
    --model-prefix online_net.backbone \
    --pretrained ${PRETRAINED} \
    --image_size 224 \
    --multiprocessing_distributed