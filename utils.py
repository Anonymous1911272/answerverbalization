import re
import transformers
from pathlib import Path
from constants import ANSWER_REGEX, ANSWER_TOKEN, ANSWERS_TOKEN


class AnswerExtractor:
    def __init__():
        self.regex = re.compile("\[.+\]")

    def __call__(verbalization):

        matches = self.regex.findall(verbalization)
        if len(matches) == 1:
            return matches[0][1:-1]

        else:
            return None

def load_huggingface_pretrained_object(object_name, config):
    return getattr(transformers, object_name).from_pretrained(config)

def batch_to_device(input_dict, use_cuda):

    for k, v in input_dict.items():
       if use_cuda:
           input_dict[k] = v.cuda()

    return input_dict

def get_num_batches(corpus_size, batch_size):
    """Computes the correct number of batch for a given corpus size.
    
    Args:
        corpus_size (int): The size of the corpus.
        batch_size (int): The number of examples in a batch.

    Returns:
        num_batches (int): The accurate number of batches necessary.
    """

    ratio = corpus_size // batch_size
    modulo = corpus_size % batch_size
    num_batches = ratio + 1 if modulo > 0 else ratio
    return num_batches

def create_directories(args):
    
    base_dir = args.base_dir
    dataset_dir = '_'.join(args.training_datasets)
    ratio_dataset_dir = str(args.training_ratio)
    task_dir = '_'.join(args.tasks)
    model_dir = args.model
    verbalization_type_dir = args.verbalization_type
    nested_dirs = base_dir + "/" + dataset_dir + "/" + ratio_dataset_dir + "/" + task_dir + "/" + model_dir + "/" + verbalization_type_dir + "/" + args.exp_name + "/"

    Path(nested_dirs).mkdir(parents=True, exist_ok=True) 
    Path(nested_dirs + 'ckpts/').mkdir(parents=True, exist_ok=True)
    
    return nested_dirs


def default_answer_masking(verbalized_answer):

    is_spurious = False
    answer_list = ANSWER_REGEX.findall(verbalized_answer)

    
    if len(answer_list) != 1:
        is_spurious = True

        if len(answer_list) == 0:
            return {"masked_verbalization": verbalized_answer,
                    "is_masking_spurious":is_spurious, 
                    "raw_answer": ""}

            
    raw_answer = answer_list[0]

    if ', ' in raw_answer:
        masked_verbalization = verbalized_answer.replace(raw_answer, ANSWERS_TOKEN)
    else:
        masked_verbalization = verbalized_answer.replace(raw_answer, ANSWER_TOKEN)

            
    num_masks = ANSWER_REGEX.findall(masked_verbalization)
    if len(num_masks) != 1:
        is_spurious = True
    
    return {"masked_verbalization": masked_verbalization, 
            "is_masking_spurious": is_spurious, 
            "raw_answer": raw_answer}


def convert_masked_to_unmasked(masked_answers, raw_answers):
    
    unmasked_answers = []
    for (masked_answer, raw_answer) in zip(masked_answers, raw_answers):
        unmasked_answer = masked_answer.replace(ANSWER_TOKEN, raw_answer)
        unmasked_answer = unmasked_answer.replace(ANSWERS_TOKEN, raw_answer)
     
        unmasked_answers.append(unmasked_answer)
    
    return unmasked_answers

def convert_to_sacrebleu_format(preds, refs):
    """
    preds should be of the form : [pred1, pred2, pred3]
    refs should be of the form [ [pred1_ref1, pred1_ref2], [pred2_ref1], [pred3_ref1, pred3_ref2]]
    
    Outputs:
    refs should be of the form : [ [pred1_ref1, pred2_ref1, pred3_ref1],
                                    [pred1_ref2, pred2_ref2, pred3_ref2],
                                    [ pred1_refn, pred2, refn, pred3_ref3]
                                ]

    """

    assert len(preds) == len(refs)
    sacrebleu_refs = []
    max_num_refs = max([len(ref) for ref in refs])
    for ref_id in range(max_num_refs):
        sacrebleu_ref = []
        for i in range(len(preds)):
           
            if len(refs[i]) < ref_id + 1:
                sacrebleu_ref.append("")
            else:
                sacrebleu_ref.append(refs[i][ref_id])

        sacrebleu_refs.append(sacrebleu_ref)
    
    return sacrebleu_refs
        





