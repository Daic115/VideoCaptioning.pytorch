import cv2
import numpy as np
from ..pycocoevalcap.bleu.bleu import Bleu
from ..pycocoevalcap.cider.cider import Cider
from ..pycocoevalcap.cider.cider_scorer import CiderScorer
from ..pycocoevalcap.bleu.bleu_scorer import BleuScorer

def sample_frames_metafunc(video_path, stride):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames.append(frame)
        frame_count += 1

    indices = list(range(8, frame_count - 7, stride))

    frames = np.array(frames)
    frame_list = frames[indices]

    return frame_list, frame_count


def bleu(gts, res):
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    print('belu = %s' % score)


class BleuS():
    def __init__(self, n=4, option='closest'):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}
        self.option = option

    def compute_score(self, gts, res):
        bleu_scorer = BleuScorer(n=self._n)
        # for gt in gts:
        hypo = res
        ref = gts
        bleu_scorer += (hypo, ref)

        # score, scores = bleu_scorer.compute_score(option='shortest')
        # score, scores = bleu_scorer.compute_score(option='closest', verbose=1)
        score, scores = bleu_scorer.compute_score(option=self.option, verbose=1)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"


class CiderS:
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        hypo = res
        ref = gts
        cider_scorer += (hypo, ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"


def get_caption_scores(gts, res):
    scorerB = BleuS()
    scorerC = CiderS()
    scoreB, _ = scorerB.compute_score(gts, res)
    scoreC, _ = scorerC.compute_score(gts, res)

    return [scoreB[4], scoreC]


def bleuD(gts, res, N=4, opt='average'):
    '''
    mean the belu scores between one generated caption and others as it's diversity score.
    :param gts: list of refs
    :param res: a string
    :param N: Belu num
    :return: score [b1,b2,b3,b4]
    '''
    scorer = BleuS(n=N, option=opt)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, _ = scorer.compute_score(gts, res)
    # print(score)
    return score