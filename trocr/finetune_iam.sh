export DATASET=web
export MODEL_NAME=ft_${DATASET}
export SAVE_PATH=./saved/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
export DATA=/home/duyx/workspace/data/benchmark_dataset/${DATASET}/
mkdir ${LOG_DIR}
export BSZ=32
export valid_BSZ=32

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
    $(which fairseq-train) \
    --data-type ${DATASET} --user-dir ./ --task text_recognition \
    --arch trocr_base \
    --seed 1111 --optimizer adam --lr 2e-05 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} \
    --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp \
    --num-workers 8 --preprocess DA2 --update-freq 1 \
    --bpe gpt2 --decoder-pretrained roberta \
    --finetune-from-model /home/duyx/workspace/code/OCR/unilm/trocr/saved/trocr-base-printed.pt --fp16 \
    --dict-path-or-url  /home/duyx/workspace/code/OCR/unilm/trocr/config/vocab.txt \
    ${DATA} 