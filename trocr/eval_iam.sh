export DATA=/home/duyx/data/benchmark_dataset/document/
export MODEL=/home/duyx/code/OCR/unilm/trocr/saved/trocr-base-printed.pt
export RESULT_PATH=/home/duyx/code/OCR/unilm/trocr/saved
export BSZ=32
export CUDA_VISIBLE_DEVICES=0

$(which fairseq-generate) \
        --data-type document --user-dir ./ --task text_recognition --input-size 384 \
        --beam 10 --scoring sroie --gen-subset test --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess DA2 \
        --bpe gpt2 --decoder-pretrained roberta \
        --dict-path-or-url /home/duyx/code/OCR/my_vocab.txt \
        --fp16 \
        ${DATA}