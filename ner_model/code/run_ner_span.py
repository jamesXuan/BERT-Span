import argparse
import glob
import logging
import os
import json

import torch
import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from callback.adversarial import FGM
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME,BertConfig,AlbertConfig
from models.bert_for_ner import BertSpanForNer
from models.albert_for_ner import AlbertSpanForNer
from processors.utils_ner import CNerTokenizer
from processors.ner_span import convert_examples_to_features
from processors.ner_span import ner_processors as processors
from processors.ner_span import collate_fn
from metrics.ner_metrics import SpanEntityScore
from processors.utils_ner import bert_extract_item

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,AlbertConfig)), ())

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertSpanForNer, CNerTokenizer),
    'albert': (AlbertConfig,AlbertSpanForNer,CNerTokenizer)
}

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = model.bert.named_parameters()
    start_parameters = model.start_fc.named_parameters()
    end_parameters = model.end_fc.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': args.learning_rate},

        {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': 0.001},

        {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': 0.001},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=600,
                                                num_training_steps=4500)
    # Check if saved optimizer or scheduler states exist

    # fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    best_f = 0.0
    for epoch in range(int(args.num_train_epochs)):
        steps = 0
        total_loss = 0
        for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
            # Skip past any already trained steps if resuming training
            steps += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "start_positions": batch[3],"end_positions": batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss.backward()

            # fgm.attack()
            # loss_adv = model(**inputs)[0]
            # loss_adv.backward()
            # fgm.restore()

            total_loss += loss
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
        print('第{}epoch的loss为{}'.format(epoch, total_loss / float(steps)))
        if epoch >= 0:
            eval_info = evaluate(args,model,tokenizer)
            eval_f = eval_info['f1']

            if eval_f>best_f:
                best_f = eval_f
                if epoch<21:
                    output_dir = os.path.join(args.output_dir, "best_checkpoint_10")
                    with open('best_checkpoint/best_epoch_10.txt','w',encoding='utf8') as f:
                        f.write(str(epoch))
                elif epoch<28:
                    output_dir = os.path.join(args.output_dir, "best_checkpoint_20")
                    with open('best_checkpoint/best_epoch_20.txt','w',encoding='utf8') as f:
                        f.write(str(epoch))
                elif epoch<34:
                    output_dir = os.path.join(args.output_dir, "best_checkpoint_30")
                    with open('best_checkpoint/best_epoch_30.txt','w',encoding='utf8') as f:
                        f.write(str(epoch))
                else:
                    output_dir = os.path.join(args.output_dir, "best_checkpoint_40")
                    with open('best_checkpoint/best_epoch_40.txt','w',encoding='utf8') as f:
                        f.write(str(epoch))
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # 直接模型保存，如果分布式有另外表达方式
                model.save_pretrained(output_dir)
                # 保存定义的args参数
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                # 保存tokenizer，如果tokenizer改变了就加载保存的，而不是直接从s3加载
                tokenizer.save_vocabulary(output_dir)
                # 优化器和schdule保存
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                
        print("训练完毕")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

def evaluate(args, model, tokenizer, prefix=""):
    metric = SpanEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    eval_features = load_and_cache_examples(args, args.task_name,tokenizer, data_type='dev')
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for step, f in enumerate(eval_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(args.device)
        subjects = f.subjects
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "start_positions": start_ids,"end_positions": end_ids}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, start_logits, end_logits = outputs[:3]
            R = bert_extract_item(start_logits, end_logits)
            T = subjects
            metric.update(true_subject=T, pred_subject=R)
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    print()
    return eval_info

def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,collate_fn=collate_fn)
    # Eval!
    results = []
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],"start_positions": None,"end_positions": None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        start_logits, end_logits = outputs[:2]
        R = bert_extract_item(start_logits, end_logits)
        if R:
            label_entities = [[args.id2label[x[0]],x[1],x[2]] for x in R]
        else:
            label_entities = []
        results.append(label_entities)
    contexts = []
    with open('datasets/test.json', 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            contexts.append(line['text'])

    output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit_checkpoint_40_data_6.json")
    with open(output_submit_file, 'w', encoding='utf8') as f1:
        for i in range(len(contexts)):
            v = {'text': contexts[i]}
            v["id"] = i
            en = []
            for j in results[i]:
                en.append({'start_pos': j[1], 'end_pos': j[-1], 'label_type': j[0], "entity": contexts[i][j[1]:j[-1]+1]})
            v['entities'] = en
            f1.write(json.dumps(v, ensure_ascii=False))
            f1.write('\n')


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    processor = processors[task]()
    # Load data features from cache or dataset file
    label_list = processor.get_labels()
    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                else args.eval_max_seq_length,
                                            cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )
    if data_type=='dev':
        return features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_input_lens)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    # Other parameters
    parser.add_argument('--markup', default='bio', type=str, choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.", )
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    # parser.add_argument("--cache_dir",default="",type=str,
    #                     help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    # parser.add_argument("--do_lower_case", defalult= True,
    #                     help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.005, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}.log'.format(args.model_type))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name, num_labels=num_labels, loss_type=args.loss_type,soft_label = True)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,do_lower_case=True)
    model = model_class.from_pretrained(args.model_path, config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        train(args, train_dataset, model, tokenizer)
    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=True)

        checkpoints = [os.path.join(args.output_dir, path) for path in os.listdir(args.output_dir)]
        checkpoints = [path for path in checkpoints if os.path.isdir(path)]
        print(f"checkpoints:{checkpoints}")

        for checkpoint in checkpoints:
            print("checkpoint: ",checkpoint)
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer)

            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
        checkpoints = [os.path.join(args.output_dir, path) for path in os.listdir(args.output_dir)]
        checkpoints = ["/home/ljy/ccf_6/ner-copy/ccks2020-ner/outputs/ccf_roberta_span_adv_6/bert/best_checkpoint_40"]

        for checkpoint in checkpoints:
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer)

if __name__ == "__main__":
    main()
