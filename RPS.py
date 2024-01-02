import itertools
import numpy as np
import math
from functools import total_ordering


HISTORY_LENGHT = 3
VOCAB = [0, 1, 2]
VOCAB_MAX = len(VOCAB)-1
VOCAB_INDEX = {'R': 0, 'P': 1, 'S': 2}
VOCAB_DICT = list(VOCAB_INDEX)
KEYS = [''.join(str(n) for n in p) for p in itertools.product(
    *[VOCAB for _ in range(HISTORY_LENGHT)])]

def beat(i):
    return i+1 if i < VOCAB_MAX else 0


@total_ordering
class Node:
    def __init__(self, learn_rate):
        self.score = 0.0
        self.times = 0
        self.learn_rate = learn_rate
        self._last_learn = 0

    def __repr__(self):
        return repr(self.score)

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __add__(self, other):
        n = Node(0)
        n.score = self.score + other.score
        return n

    def snapshot(self):
        n = Node(self.learn_rate)
        n.times = self.times
        n.score = self.score
        return n

    def update(self):
        self.times += 1
        t = math.exp(-self.learn_rate * self.times)
        self._last_learn = t / (1+t)
        self.score += self._last_learn


def _createScores(learn_rate):
    return np.array([Node(learn_rate), Node(learn_rate), Node(learn_rate)])


class MarkovChain():

    def __init__(self, learn_rate=0.001, name="MarkovChain",verbose=False):
        if (learn_rate <= 0):
            raise ValueError("learn rate must above 0")
        if (learn_rate >= 0.01):
            raise ValueError("learn rate must less then 0.01")

        self._matrix = {key: _createScores(learn_rate) for key in KEYS}
        self._name=name
        self._verbose=verbose

    def update_matrix(self, key, i):
        scores = self._matrix[key]
        scores[i].update()
        self._verbose and print("== {}: new {}'s scores: {}".format(self._name, key, scores))

    def get_scores(self, key):
        return self._matrix[key]

class Model:
    HISTORY_BUFFER_SIZE=HISTORY_LENGHT*10
    def __init__(self, opinionated=False, verbose=False):
        self._opinionated = opinionated
        self._verbose=verbose
        self._step=0
        self.oppo = MarkovChain(name="Opposite",verbose=verbose)
        self.oppo_history = np.zeros(HISTORY_LENGHT, dtype=np.int8).tolist()
        self.mirror = MarkovChain(name="Mirror",verbose=verbose)
        self.mirror_history = np.ones(HISTORY_LENGHT+1, dtype=np.int8).tolist()
        self.last_choose_mirror = None

    def _append_to_history(self, move, use_mirror=False):
        history=self.mirror_history if use_mirror else self.oppo_history
        history.append(move)
        if(len(history)>Model.HISTORY_BUFFER_SIZE):
            new_history=history[-HISTORY_LENGHT-1:].copy()
            if use_mirror:
                self.mirror_history=new_history
            else:
                self.oppo_history=new_history
            self._verbose and print(f"<<<<Truncate {'Mirror' if use_mirror else 'Opposite'} history buffer from {history} to {new_history}")

    def oppo_move(self, prev_oppo):
        self._step+=1
        self._append_to_history(prev_oppo)
        oppo_seqence = ''.join(str(n) for n in self.oppo_history[-HISTORY_LENGHT-1:-1])
        mirror_seqence = ''.join(str(n) for n in self.mirror_history[-HISTORY_LENGHT-1:-1])
        last_move=self.mirror_history[-1]

        self._verbose and print(f"==========Round {self._step}: prev_oppo_move={prev_oppo},last_move={last_move} =============================")
        if (self._opinionated):
            self.oppo.update_matrix(oppo_seqence, prev_oppo)
            return
        
        # introspective updates
        if (self.last_choose_mirror == None):
            return
        
        if (last_move==beat(prev_oppo)):
            # last_move is win
            mod = self.mirror if self.last_choose_mirror else self.oppo
            seqence = mirror_seqence if self.last_choose_mirror else oppo_seqence
            mod.update_matrix(seqence, prev_oppo)
        elif (prev_oppo == beat(last_move)):
            # last_move is loose
            self.mirror.update_matrix(mirror_seqence, prev_oppo)
        else:
            self.oppo.update_matrix(oppo_seqence, prev_oppo)
        
    def predict(self):
        oppo_seqence = ''.join(str(n) for n in self.oppo_history[-HISTORY_LENGHT:])
        oppo_scores = self.oppo.get_scores(oppo_seqence)
        if (self._opinionated):
            return False, np.argmax(oppo_scores)

        # introspective prediction
        mirror_seqence = ''.join(str(n) for n in self.mirror_history[-HISTORY_LENGHT:])
        mirror_scores = self.mirror.get_scores(mirror_seqence)
        scores = oppo_scores + mirror_scores
        prediction = np.argmax(scores)

        choose_mirror = mirror_scores[prediction] > oppo_scores[prediction]
        return choose_mirror, prediction

    def move(self):
        choose_mirror,prediction=self.predict()
        move=beat(prediction)

        self._append_to_history(move, use_mirror=True)
        self.last_choose_mirror=choose_mirror
        self._verbose and print(f"<========= Choose mirror: {self.last_choose_mirror} ; Predict: {prediction}; Move: {move} =============================")
        return move

model = Model(opinionated=False, verbose=False)


def player(prev_play):
    prev_play = prev_play if prev_play != "" else 'R'
    prev_oppo_move = VOCAB_INDEX[prev_play]

    model.oppo_move(prev_oppo_move)
    movement = model.move()

    return VOCAB_DICT[movement]
