CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/bert-pretrain/
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"

python run_ner_softmax.py --model_type=bert --model_path=bert-pretrain/ --config_name=bert-pretrain/bert_config.json --task_name=cluener --do_eval --do_lower_case --data_dir=datasets/cluener/ --train_batch_size=48 --eval_batch_size=48 --learning_rate=3e-5 --num_train_epochs=1.0 --logging_steps=224 --save_steps=224 --output_dir=outputs/cluener_output/ --overwrite_output_dir --seed=42 --tokenizer_name=token_save/
