import os
import sys
import logging
import argparse
import datasets as datasets_module
import random
import torch
import numpy as np
import transformers
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from utils import AnswerExtractor
from models import PretrainedModel
from utils import convert_masked_to_unmasked
from utils import convert_to_sacrebleu_format
from constants import QUESTION_GENERATION_TASK
from utils import load_huggingface_pretrained_object
from utils import batch_to_device, get_num_batches, create_directories
from constants import ANSWER_TOKEN, ANSWERS_TOKEN, SEP_ANSWER_TOKEN, DATASETS
from metrics import PapineniBleuScore, SacreBleuScore, MeteorScore, ChrfScore, TerScore


class Experiment:
    def __init__(self, path, args):

        self.path = path 
        self.args = args

        self.log_path = self.path + self.args.exp_name + '.log'
        if os.path.isfile(self.log_path):
            raise OSError("An existing Experiment was already done with same log name. Rename your experiment.")
    
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO, filename=self.log_path)
        self.logger = logging.getLogger(__name__)


        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        if torch.cuda.is_available() and self.args.use_cuda:
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
        
        os.environ['TRANSFORMERS_CACHE'] = self.args.cache_dir

    def launch(self):
        
        self.logger.info("******************** Running Experiment ********************")
        
        self.logger.info("\t\t\t + + + + + + ARGS + + + + + +")
        self.logger.info(f"\t{self.args}")

        # collect dataset
        datasets = getattr(datasets_module, 'Datasets')(self.args.training_datasets)
        # dataset = getattr(datasets, args.datasets)(args.dataset, args.dataset_path)
        train_data = datasets.get_training_data(self.args.training_ratio)
        self.logger.info("\tData Loaded!")
        self.logger.info(f"\tNum training examples = {len(train_data)}")
        # test_data = dataset.get_testing_data()

        # load tokenizer
        tokenizer = load_huggingface_pretrained_object(self.args.tokenizer, self.args.config)

        # add tokens + resizing model nn.Embeddings
        tokenizer.add_tokens(ANSWER_TOKEN, special_tokens=False)
        tokenizer.add_tokens(ANSWERS_TOKEN, special_tokens=False)
        tokenizer.add_tokens(SEP_ANSWER_TOKEN, special_tokens=False)

        # load model
        # model = load_huggingface_pretrained_object(args.model, args.config)
        model = PretrainedModel(
                self.args.model, 
                self.args.config, 
                len(tokenizer), 
                self.args.tasks)
        model = model.cuda() if self.args.use_cuda else model
        
        # INFOS
        self.logger.info("\tModel Loaded!") 
        self.logger.info(f"\tNumber of parameters:{model.get_num_trainable_parameters()}")
        self.logger.info("\tTraining Example:")
        self.logger.info(f"\t\tInput Question:{getattr(train_data[20], self.args.input_type + '_lowercase') if self.args.lowercase else getattr(train_data[0], self.args.input_type)}")
        self.logger.info(f"\t\tOutput Answer:{getattr(train_data[20], self.args.verbalization_type + '_lowercase') if self.args.lowercase else getattr(train_data[0], self.args.verbalization_type)}")

        if QUESTION_GENERATION_TASK in self.args.tasks:
            self.logger.info(f"\t\tOutput Question:{getattr(train_data[20], 'question')}")

        # preprocess dataset

        # TODO tokenize once and for all the train  + test

        # optimizer
        optimizer = getattr(optim, self.args.optimizer)(params=model.parameters(), lr=self.args.lr)

        # loss
        loss = getattr(nn, self.args.loss)(ignore_index=tokenizer.pad_token)

        pbar = trange(int(args.max_epochs), desc="Epoch")
        
        self.logger.info(f"Training on {self.args.training_ratio} of each of {self.args.training_datasets} datasets. . . \n")

        for epoch in pbar:
            # shuffle training data
            random.shuffle(train_data)

            self.train(epoch, train_data, self.args.batch_size, model, tokenizer, loss, optimizer, pbar)

            if (epoch + 1) % args.eval_every == 0 or epoch+1 == args.max_epochs:
                self.logger.info(f"# # # Epoch {epoch+1} # # #\n")
                for dataset in DATASETS:
                    self.logger.info(f"Evaluating on {dataset}. . . ")
                    test_data = getattr(datasets, dataset).testing_examples
                    scores = self.evaluate(test_data, self.args.batch_size, model, tokenizer, self.args.use_cuda)
                    self.logger.info(f"Bleu:{scores['BLEU']}\tSacreBLeu:{scores['SACREBLEU']}\tMeteor:{scores['METEOR']}\tChrf:{scores['CHRF']}\tTER:{scores['TER']}\n")
                    #TODO save best model on test
                self.save_ckpt(epoch+1, model, optimizer, loss)
        

    def train(self, epoch, train_data, batch_size, model, tokenizer, loss, optimizer, pbar):

        num_batches = get_num_batches(len(train_data), batch_size)
        model.train()
        losses = 0
        for i in range(num_batches):
            batch = train_data[i * batch_size: (i+1) * batch_size]

            # input_questions = [example.question_lowercase if self.args.lowercase else example.question for example in batch]
            input_questions = [getattr(example, self.args.input_type + '_lowercase') if self.args.lowercase else getattr(example, self.args.input_type) for example in batch]
            target_verbalized_answer = [getattr(example, self.args.verbalization_type + '_lowercase') if self.args.lowercase else getattr(example, self.args.verbalization_type) for example in batch]

            input_questions = tokenizer(input_questions,
                                        padding=True,
                                        is_split_into_words=False,
                                        return_tensors='pt')

            target_verbalized_answer = tokenizer(target_verbalized_answer,
                                                 padding=True,
                                                 is_split_into_words=False,
                                                 return_tensors='pt')

            # put data on device
            input_questions = batch_to_device(input_questions, self.args.use_cuda)
            target_verbalized_answer = batch_to_device(target_verbalized_answer, self.args.use_cuda)
            output = model(input_questions, labels=target_verbalized_answer)

            loss = output.loss
            pbar.set_postfix(loss=loss.item())
            losses += loss.item()
            
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.args.gradient_clipping)
            optimizer.step()

    def infer(self, test_data, batch_size, model, tokenizer, use_cuda):
        pass

    def evaluate(self, test_data, batch_size, model, tokenizer, use_cuda):

        bleu_scorer = PapineniBleuScore()
        sacrebleu_scorer = SacreBleuScore()
        meteor_scorer = MeteorScore()
        ter_scorer = TerScore()
        chrf_scorer = ChrfScore()

        num_batches = get_num_batches(len(test_data), batch_size)

        model.eval()

        predictions = []
        tokenized_predictions = []
        # references = [sample.verbalized_answers for sample in test_data]
        references = []
        
        # first generate predictions (some metrics require the whole corpus)
        for i in range(num_batches):
            batch = test_data[i * batch_size: (1+i)*batch_size]
            input_questions = []
            raw_answers = [] 
            for sample in batch:
                if self.args.lowercase:
                    input_questions.append(getattr(sample, args.input_type + '_lowercase'))
                    references.append(sample.verbalized_answers_lowercase)
                    raw_answers.append(sample.raw_answer_lowercase)
                
                else:
                    input_questions.append(getattr(sample, args.input_type))
                    references.append(sample.verbalized_answers)
                    raw_answers.append(sample.raw_answer)

            input_questions = tokenizer(input_questions,
                                        padding=True,
                                        is_split_into_words=False,
                                        return_tensors='pt')
    
            input_questions = batch_to_device(input_questions, use_cuda)
            generated_ids = model.generate(input_questions)
            
            generated_verbalized_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)             
            
            if self.args.lowercase:
                generated_verbalized_answers = [prediction.lower() for prediction in generated_verbalized_answers]

            if self.args.verbalization_type == 'masked_verbalization':
                generated_verbalized_answers = convert_masked_to_unmasked(generated_verbalized_answers, raw_answers)
            
            predictions.extend(generated_verbalized_answers)
        
        sacrebleu_refs = convert_to_sacrebleu_format(predictions, references)

        bleu_scorer(predictions, references)
        sacrebleu_scorer(predictions, sacrebleu_refs)
        meteor_scorer(predictions, references)
        chrf_scorer(predictions, sacrebleu_refs)
        ter_scorer(predictions, sacrebleu_refs)

        return {"BLEU": bleu_scorer.score,
                "SACREBLEU": sacrebleu_scorer.score,
                "METEOR": meteor_scorer.score,
                "CHRF":chrf_scorer.score,
                "TER": ter_scorer.score}


    def save_ckpt(self, epoch, model, optimizer, loss):
        saved_model_path = self.path + 'ckpts/' 
        filename = 'epoch_' + str(epoch) + '.ckpt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, saved_model_path + filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Answer Verbalization')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--use_cuda', action='store_true')

    # experiments
    parser.add_argument('--base_dir', default='/data/cmjt4501/answerverbalization/exp6', type=str)
    parser.add_argument('--exp_name', default='run1', type=str)

    # dataset
    parser.add_argument('--training_datasets', nargs='+')
    parser.add_argument('--use_corrected', action='store_true')
    parser.add_argument('--training_ratio', default=100, type=int)
    

    # model
    parser.add_argument('--model', default='T5ForConditionalGeneration', type=str)
    parser.add_argument('--tokenizer', default='T5Tokenizer', type=str)
    parser.add_argument('--config', default='t5-base', type=str)
    parser.add_argument('--cache_dir', default='./', type=str)
    
    # training
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--eval_every', default=2, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--loss', default='CrossEntropyLoss', type=str)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--gradient_clipping', default=5.0, type=float)
    parser.add_argument('--input_type', default='question', type=str)
    parser.add_argument('--verbalization_type', default='masked_verbalization', type=str)
    parser.add_argument('--lowercase', action='store_true')

    # args = parser.parse_args(['--use_cuda', '--training_datasets', 'VANiLLa', '--tasks', 'AV', '--cache_dir', '/data/cmjt4501/', '--lowercase'])
    
    args = parser.parse_args()
    exp_path = create_directories(args)
    exp = Experiment(exp_path, args)
    exp.launch()






