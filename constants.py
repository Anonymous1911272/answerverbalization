import re

ANSWER_REGEX = re.compile("\[.+\]")
ANSWER_TOKEN = '[answer]'
ANSWERS_TOKEN = '[answers]'
SEP_ANSWER_TOKEN = '[sep_answer]'
DATASETS = ['RawVANiLLa', 'CorrectedVANiLLa','VQuAnDa', 'ParaQA']
DATA_DIR = './data/'
QUESTION_GENERATION_TASK = 'RQ'    # stands for 'Repeat Question'
ANSWER_GENERATION_TASK = 'AV'   # stands for 'Answer Verbalization'
TASKS = [QUESTION_GENERATION_TASK, ANSWER_GENERATION_TASK]

