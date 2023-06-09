CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/roberta_wwm_large_ext
export GLUE_DIR=$CURRENT_DIR/CLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"

python run_ner_softmax.py --model_type=bert --model_name_or_path=bert-pretrain --task_name=ccks --do_train --do_eval --loss_type=ce --do_predict --data_dir=ccks/ --train_max_seq_length=512 --eval_max_seq_length=512 --per_gpu_train_batch_size=32 --per_gpu_eval_batch_size=1 --learning_rate=3e-5 --num_train_epochs=4.0 --output_dir=outputs/ccks_output/ --overwrite_output_dir --seed=42

#python run_ner_softmax.py \
#  --model_type=bert \
#  --model_name_or_path=$BERT_BASE_DIR \
#  --task_name=$TASK_NAME \
#  --do_predict \
#  --do_lower_case \
#  --loss_type=ce \
#  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
#  --train_max_seq_length=128 \
#  --eval_max_seq_length=512 \
#  --per_gpu_train_batch_size=24 \
#  --per_gpu_eval_batch_size=24 \
#  --learning_rate=3e-5 \
#  --num_train_epochs=4.0 \
#  --logging_steps=224 \
#  --save_steps=224 \
#  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#  --overwrite_output_dir \
#  --seed=42