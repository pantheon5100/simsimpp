python3 ../../../main_pretrain.py \
    --dataset $1 \
    --encoder resnet18 \
    --data_dir datasets \
    --max_epochs 1000 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 3 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --name mocov2plus-$1 \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --method mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 32768 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier