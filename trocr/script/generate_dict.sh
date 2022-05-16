export NWORDSSRC=21128

fairseq-preprocess 	\
--source-lang cn 	\
--target-lang cn 	\
--trainpref /home/duyx/workspace/code/OCR/unilm/trocr/data/train 	\
--validpref /home/duyx/workspace/code/OCR/unilm/trocr/data/valid 	\
--testpref /home/duyx/workspace/code/OCR/unilm/trocr/data/test 		\
--task multilingual_language_modeling 	\
--nwordssrc ${NWORDSSRC} 		\
--joined-dictionary 	\
--only-source \
--destdir ../dict_${NWORDSSRC} 
