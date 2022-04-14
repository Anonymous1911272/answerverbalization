import re
import json
import random
from utils import default_answer_masking
from constants import ANSWER_REGEX, ANSWER_TOKEN, ANSWERS_TOKEN, SEP_ANSWER_TOKEN, DATASETS, DATA_DIR


class TrainingSample:
    def __init__(self, question, raw_answer, verbalized_answer, masked_verbalization, is_spurious, prefix='question:'):
        self.question = question
        self.raw_answer = raw_answer
        self.verbalized_answer = verbalized_answer
        self.masked_verbalization = masked_verbalization
        self.question_and_concat_answer = self.question + " " + SEP_ANSWER_TOKEN + " " + self.raw_answer
        self.question_with_prefix = prefix + ' ' + self.question
        self.question_with_prefix_and_concat_answer = prefix + ' ' + self.question_and_concat_answer

        self.question_lowercase = self.question.lower()
        self.raw_answer_lowercase = self.raw_answer.lower()
        self.verbalized_answer_lowercase = self.verbalized_answer.lower()
        self.masked_verbalization_lowercase = self.masked_verbalization.lower()
        self.question_and_concat_answer_lowercase = self.question_and_concat_answer.lower()
        self.question_with_prefix_lowercase = self.question_with_prefix.lower()
        self.question_with_prefix_and_concat_answer_lowercase = self.question_with_prefix_and_concat_answer.lower()

        self.is_spurious = is_spurious


class TestingSample:
    def __init__(self, question, raw_answer, verbalized_answers, masked_verbalizations, prefix='question:'):

        assert len(verbalized_answers) == len(masked_verbalizations)

        self.question = question
        self.raw_answer = raw_answer
        self.verbalized_answers = verbalized_answers
        self.masked_verbalizations = masked_verbalizations
        self.question_and_concat_answer = self.question + " " + SEP_ANSWER_TOKEN + " " + self.raw_answer
        self.question_with_prefix = prefix + ' ' + self.question
        self.question_with_prefix_and_concat_answer = prefix + ' ' + self.question_and_concat_answer
        
        self.question_lowercase = self.question.lower()
        self.raw_answer_lowercase = self.raw_answer.lower()
        self.verbalized_answers_lowercase = [verbalization.lower() for verbalization in self.verbalized_answers]
        self.masked_verbalizations_lowercase = [verbalization.lower() for verbalization in self.masked_verbalizations]
        self.question_and_concat_answer_lowercase = self.question_and_concat_answer.lower() 
        self.question_with_prefix_lowercase = self.question_with_prefix.lower()
        self.question_with_prefix_and_concat_answer_lowercase = self.question_with_prefix_and_concat_answer.lower()

class BaseDataset:
    def __init__(self, name, path, github):
        self.name = name
        self.path = path
        self.github = github
        self.training_examples = []
        self.testing_examples = []

    def __len__(self):
        return len(self.training_examples + self.testing_examples)

    def load_data(self):
        raise NotImplementedError

    def get_training_data(self, ratio):
        
        random.shuffle(self.training_examples)
        if ratio != 100:
            return [sample for sample in self.training_examples[:int(len(self.training_examples) * ratio/100)] if not sample.is_spurious]

        return [sample for sample in self.training_examples if not sample.is_spurious]

    def get_testing_data(self):
        return self.testing_examples


class VANiLLa(BaseDataset):
    def __init__(self, name, path, github='https://github.com/AskNowQA/VANiLLa'):
        super(VANiLLa, self).__init__(name=name, path=path, github=github)
        
        self.load_training_data()
        self.load_testing_data()

    def load_training_data(self):
        training_data = []
        with open(self.path + '/train.json', 'r', encoding='utf-8') as f:
            for line in f:
                training_data.append(json.loads(line))

        for sample in training_data:
            
            verbalized_answer = sample['answer_sentence']
            question = sample['question']
            raw_anwser = sample['answer']
            mask = self.mask_answer(verbalized_answer, raw_anwser)
            
            new_sample = TrainingSample(question=question,
                                        raw_answer=raw_anwser,
                                        verbalized_answer=verbalized_answer,
                                        masked_verbalization=mask['masked_verbalization'],
                                        is_spurious=mask['is_masking_spurious'])
            
            self.training_examples.append(new_sample)

    def load_testing_data(self):
        testing_data = []
        with open(self.path + '/test.json', 'r', encoding='utf-8') as f:
            for line in f:
                testing_data.append(json.loads(line))

        for sample in testing_data:
            question = sample['question']
            raw_answer = sample['answer']
            verbalized_answers = []
            masked_verbalizations = []

            verbalized_answer = sample['answer_sentence']
            mask = self.mask_answer(verbalized_answer, raw_answer)

            verbalized_answers.append(verbalized_answer)
            masked_verbalizations.append(mask['masked_verbalization'])

            new_sample = TestingSample(question=question,
                                       raw_answer=raw_answer,
                                       verbalized_answers=verbalized_answers,
                                       masked_verbalizations=masked_verbalizations,
                                       )

            self.testing_examples.append(new_sample)

    def mask_answer(self, verbalized_answer, raw_answer):
        # print(verbalized_answer)
        is_spurious = False
        masked_verbalization = verbalized_answer.replace(raw_answer.lower(), ANSWER_TOKEN)
        
        if len(ANSWER_REGEX.findall(masked_verbalization)) != 1:
            is_spurious = True
        # print(masked_verbalization)
        # print(raw_answer)
        return {"masked_verbalization": masked_verbalization,
                "is_masking_spurious": is_spurious}



class ParaQA(BaseDataset):
    def __init__(self, name, path, github='https://github.com/barshana-banerjee/ParaQA'):
        super(ParaQA, self).__init__(name=name, path=path, github=github)

        self.load_training_data()
        self.load_testing_data()

    def load_training_data(self):
        with open(self.path + '/train.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        for sample in training_data:
            question = sample['question']
            verbalized_answer = sample['verbalized_answer']
            # print("Question:{}".format(question))
            # print("verbalized_answer:{}".format(verbalized_answer))
            # print(sample['uid'])

            mask = self.mask_answer(verbalized_answer)
            # print("Mask answer:{}".format(mask['raw_answer']))
            new_sample = TrainingSample(question=question,
                                        raw_answer=mask['raw_answer'],
                                        verbalized_answer=verbalized_answer,
                                        masked_verbalization=mask['masked_verbalization'],
                                        is_spurious=mask['is_masking_spurious']
                                        )
            self.training_examples.append(new_sample)

            for i in range(2, 9):
                verbalized_answer = sample['verbalized_answer_' + str(i)]
                if verbalized_answer != "":
                    mask = self.mask_answer(verbalized_answer)
                    # print(mask)
                    new_sample = TrainingSample(question=question,
                                                raw_answer=mask['raw_answer'],
                                                verbalized_answer=verbalized_answer,
                                                masked_verbalization=mask['masked_verbalization'],
                                                is_spurious=mask['is_masking_spurious']
                                                )
                    self.training_examples.append(new_sample)

    def load_testing_data(self):

        with open(self.path + '/test.json', 'r', encoding='utf-8') as f:
            testing_data = json.load(f)

        for sample in testing_data:
            question = sample['question']
            verbalized_answers = []
            masked_verbalizations = []

            verbalized_answer = sample['verbalized_answer']
            mask = self.mask_answer(verbalized_answer)
            verbalized_answers.append(verbalized_answer)
            masked_verbalizations.append(mask['masked_verbalization'])

            for i in range(2, 9):
                verbalized_answer = sample['verbalized_answer_' + str(i)]

                if verbalized_answer != "":
                    mask  = self.mask_answer(verbalized_answer)
                    verbalized_answers.append(sample['verbalized_answer'])
                    masked_verbalizations.append(mask['masked_verbalization'])

            new_sample = TestingSample(question=question,
                                       raw_answer=mask['raw_answer'],
                                       verbalized_answers=verbalized_answers,
                                       masked_verbalizations=masked_verbalizations
                                       )
            self.testing_examples.append(new_sample)

    def mask_answer(self, verbalized_answer):
        
        mask = default_answer_masking(verbalized_answer)

        return mask


class VQuAnDa(BaseDataset):
    def __init__(self, name, path, github='https://github.com/AskNowQA/VQUANDA'):
        super(VQuAnDa, self).__init__(name=name, path=path, github=github)

        self.load_training_data()
        self.load_testing_data()

    def load_training_data(self):

        with open(self.path + '/train.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        for sample in training_data:
            question = sample['question']
            verbalized_answer = sample['verbalized_answer']
            if question == "" or verbalized_answer == "":
                continue

            mask = self.mask_answer(verbalized_answer)
            new_sample = TrainingSample(question=question,
                                        raw_answer=mask['raw_answer'],
                                        verbalized_answer=verbalized_answer,
                                        masked_verbalization=mask['masked_verbalization'],
                                        is_spurious=mask['is_masking_spurious']
                                        )
            self.training_examples.append(new_sample)

    def load_testing_data(self):
        with open(self.path + '/test.json', 'r', encoding='utf-8') as f:
            testing_data = json.load(f)

        for sample in testing_data:
            question = sample['question']
            verbalized_answer = sample['verbalized_answer']
            if question == "" or verbalized_answer == "":
                continue

            verbalized_answers = []
            masked_verbalizations = []

            mask = self.mask_answer(verbalized_answer)
            verbalized_answers.append(verbalized_answer)
            masked_verbalizations.append(mask['masked_verbalization'])

            new_sample = TestingSample(question=question,
                                       raw_answer=mask['raw_answer'],
                                       verbalized_answers=verbalized_answers,
                                       masked_verbalizations=masked_verbalizations
                                       )
            self.testing_examples.append(new_sample)

    def mask_answer(self, verbalized_answer):

        mask = default_answer_masking(verbalized_answer)

        return mask


class Datasets:
    def __init__(self, training_datasets):
        
        self.training_datasets = training_datasets
        
        self.VQuAnDa = VQuAnDa('VQUANDA', DATA_DIR + 'VQUANDA') 
        self.RawVANiLLa = VANiLLa('RawVANiLLa', DATA_DIR + 'VANiLLa/raw/')
        self.CorrectedVANiLLa = VANiLLa('CorrectedVANiLLa', DATA_DIR + 'VANiLLa/corrected/') 
        self.ParaQA = ParaQA('ParaQA', DATA_DIR + 'ParaQA/')

        self.training_examples = []

    def get_training_data(self, ratio):

        for dataset in self.training_datasets:
            self.training_examples.extend(getattr(self, dataset).get_training_data(ratio))

        return self.training_examples


