import sacrebleu
import nltk


class BaseScore:
    def __init__(self, name):
        self.name = name
        self.score = None

    def __call__(self, hypotheses, references):
        raise NotImplementedError


class SacreBleuScore(BaseScore):
    def __init__(self):
        super(SacreBleuScore, self).__init__(name='SacreBleu')
        self.score = 0

    def __call__(self, hypotheses, references):
        self.score = sacrebleu.corpus_bleu(hypotheses, references)


class PapineniBleuScore(BaseScore):
    def __init__(self):
        super(PapineniBleuScore, self).__init__(name='PapineniBleu')
        self.score = 0

    def __call__(self, hypotheses, references):
        num_test_samples = 0

        for i, (hypothesis, hypothesis_references) in enumerate(zip(hypotheses, references)):
            cur_references = [reference.split() for reference in hypothesis_references]
            cur_hypothesis = hypothesis.split()

            self.score += nltk.translate.bleu_score.sentence_bleu(cur_references, cur_hypothesis)
            num_test_samples += 1
        self.score /= num_test_samples


class MeteorScore(BaseScore):
    def __init__(self):
        super(MeteorScore, self).__init__(name='Meteor')
        self.score = 0

    def __call__(self, hypotheses, references):
        num_test_samples = 0
        for i, (hypothesis, hypothesis_references) in enumerate(zip(hypotheses, references)):
            self.score += nltk.meteor(hypothesis_references, hypothesis)
            num_test_samples += 1
        self.score /= num_test_samples


class TerScore(BaseScore):
    def __init__(self):
        super(TerScore, self).__init__(name='TER')
        self.score = 0
        self.ter = sacrebleu.metrics.TER()

    def __call__(self, hypotheses, references):
        self.score = self.ter.corpus_score(hypotheses, references)


class ChrfScore(BaseScore):
    def __init__(self):
        super(ChrfScore, self).__init__(name='CHRF')
        self.score = 0
        self.chrf = sacrebleu.metrics.CHRF()

    def __call__(self, hypotheses, references):
        self.score = self.chrf.corpus_score(hypotheses, references)

        



