export DATASET=web
export DATA=/home/duyx/workspace/data/benchmark_dataset/${DATASET}/
export MODEL=/home/duyx/workspace/code/OCR/unilm/trocr/saved/ft_web/checkpoint_best.pt
export RESULT_PATH=/home/duyx/workspace/code/OCR/unilm/trocr/saved
export BSZ=32
export CUDA_VISIBLE_DEVICES=0

$(which fairseq-generate) \
        --data-type ${DATASET} --user-dir ./ --task text_recognition --input-size 384 \
        --beam 10 --scoring acc_ed --gen-subset test --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess DA2 \
        --bpe gpt2 --decoder-pretrained roberta \
        --dict-path-or-url /home/duyx/workspace/code/OCR/unilm/trocr/config/vocab.txt \
        --fp16 \
        ${DATA}