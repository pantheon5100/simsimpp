python3 ../../../main_pretrain.py \
    --dataset cifar100 \
    --encoder resnet18 \
    --data_dir ../../../datasets \
    --max_epochs 1000 \
    --gpus 7 \
    --distributed_backend ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.5 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --zero_init_residual \
    --name moverage-normz-random_z \
    --project solo-learn-simsiammova \
    --entity kaistssl \
    --wandb \
    --method simsiammova \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --output_dim 2048\
    --normz\
    --random_z
