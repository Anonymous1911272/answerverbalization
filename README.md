# Answer Verbalization

## Installation

### Create a virtual environment with conda
``
conda create -n answerverbalization python=3.6
``

### Activate Environment

``
conda activate answerverbalization
``

### Install packages in the requirements.txt file using pip

``
pip install -r requirements.txt
``

## Training + Testing
To train a T5 model on ParaQA with the whole training data (--training\_ratio set to 100) for the Answer Verbalization task (--tasks set to 'AV'), you can execute the command below.

``
CUDA_VISIBLE_DEVICES=0 python train.py --seed "42" --exp_name run"42" --training_datasets ParaQA --training_ratio "100" --lr "0.0001" --tasks "AV" --model "T5ForConditionalGeneration" --tokenizer "T5Tokenizer" --config "t5-base" --cache_dir "data/" --lowercase --use_cuda
``

## Task Definition
The task of **A**nswer **V**erbalization consists in generating a natural language response given the previous question and the raw answer. Given a question, we aim at generating a masked verbalization. 
### Example
**Input**: Where is the headquarters of Sigma TV ? 

**Output**: The headquarters of Sigma TV is in **\[ANSWER\]**.

## Datasets
Recently, several works contributed to provide verbalized response with its corresponding question and raw ranswer
### [VQuAnDa](https://github.com/AskNowQA/VQUANDA)

### [VANiLLA](https://github.com/AskNowQA/VANiLLa)

### [ParaQA](https://github.com/barshana-banerjee/ParaQA)

## Models
### State-of-the-art
Recently, [1](https://arxiv.org/abs/2106.13316) proposed VOGUE (**V**erbalization thr**O**uGh m**U**lti-task  l**E**arning). Their model takes as input the question, its logical form (*a query*) and the raw answer. Throughout cross-attention and thresholding, they finally generate the verbalization directly (no masking) with a **hybrid** decoding. They compared their approach with RNN [https://arxiv.org/pdf/2106.13316.pdf], Convolutional[https://arxiv.org/pdf/2106.13316.pdf], Transformer[https://arxiv.org/pdf/2106.13316.pdf] and BERT[https://arxiv.org/pdf/2106.13316.pdf].
### Results
First value and second value stand for the BLEU score (Papineni et al) and METEOR respectively.
| Models | VQuAnDa | ParaQA | VANiLLa |
| ------ | ------ | ------ | ------ |
| RNN (Q) | 15.43<br>53.15 | 22.45<br>58.41 | 16.66<br>58.67 |
| Transformer (Q) | 18.37<br>56.83 | 23.61<br>59.63 | 30.80<br>62.16 |
| (***SOTA***) VOGUE (H) | 28.76<br>67.21 | 32.05<br>68.85 | 35.46<br>65.04 |
| Akermi et al. | 22.70<br>48.04 | 18.25<br>44.27 | 18.30<br>48.27 |
| T5 (masking) | 39.07<br>67.70 | 30.62<br>59.81 | 45.87<br>67.15 |
| BART (masking) | 43.90><br>71.92 | 35.57<br>65.40 | 45.69<br> 66.71 |
