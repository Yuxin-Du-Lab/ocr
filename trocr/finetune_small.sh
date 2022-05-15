export MODEL_NAME=ft_document
export SAVE_PATH=./saved/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
export DATA=/home/duyx/data/benchmark_dataset/document/
mkdir ${LOG_DIR}
export BSZ=8
export valid_BSZ=16

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
    $(which fairseq-train) \
    --data-type document --user-dir ./ --task text_recognition \
    --arch trocr_small \
    --seed 1111 --optimizer adam --lr 2e-05 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} \
    --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp \
    --num-workers 8 --preprocess DA2 --update-freq 1 \
    --bpe sentencepiece --sentencepiece-model ./unilm3-cased.model --decoder-pretrained unilm \
    --finetune-from-model /home/duyx/code/OCR/unilm/trocr/saved/trocr-small-printed.pt --fp16 \
    --dict-path-or-url  /home/duyx/code/OCR/my_vocab.txt\
    ${DATA} 