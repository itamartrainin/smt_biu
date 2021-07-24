# Author: Itamar Trainin 315425967

from datetime import datetime
import numpy as np

NULL = '<NULL>'

def evaluate(pred, actual):
    sure, possible = actual
    (size_a, size_s, size_a_and_s, size_a_and_p) = (0.0, 0.0, 0.0, 0.0)
    for pred_align, sure_align, possible_align in zip(pred, sure, possible):
        size_a += len(pred_align)
        size_s += len(sure_align)
        size_a_and_s += len(pred_align & sure_align)
        size_a_and_p += len(pred_align & possible_align) + len(pred_align & sure_align)

    precision = size_a_and_p / size_a
    recall = size_a_and_s / size_s
    aer = 1 - ((size_a_and_s + size_a_and_p) / (size_a + size_s))

    return precision, recall, aer


def save_alignments(output_dir, alignments):
    with open(output_dir + '/alignments_{}.txt'.format(datetime.now().strftime('%d%m%y_%H%M')), 'w') as f:
        for sentence in alignments:
            sentence = list(sorted(sentence))
            f.write(' '.join(list(map(lambda x: str(x[0]) + '-' + str(x[1]), sentence))) + '\n')
    print('Alignments saved.')


class IBM_M1():
    def __init__(self, f_word_to_ix, e_word_to_ix, f_data, e_data, f_max, e_max, test_data, num_epochs, debug=False,
                 t_pretrained=None, smoothing_coef=0):
        super(IBM_M1, self).__init__()

        self.f_word_to_ix = f_word_to_ix
        self.e_word_to_ix = e_word_to_ix

        self.f_data = f_data
        self.e_data = e_data
        self.test_data = test_data

        self.f_max = f_max
        self.e_max = e_max

        self.t_pretrained = t_pretrained

        self.num_epochs = num_epochs
        self.debug = debug
        self.smoothing_coef = smoothing_coef

    def align(self, t):
        time_align = datetime.now()
        alignments = []
        for f_sent, e_sent in zip(self.f_data, self.e_data):
            sent_aligns = []
            for i, f in enumerate(f_sent):
                max_val = 0
                max_pos = 0
                max_idx = 0
                for j, e in enumerate(e_sent):
                    if t[f][e] > max_val:
                        max_val = t[f][e]
                        max_pos = j
                        max_idx = e
                if NULL in self.e_word_to_ix:
                    if max_idx != self.e_word_to_ix[NULL]:
                        sent_aligns.append((i, max_pos))
                else:
                    sent_aligns.append((i, max_pos))
            alignments.append(set(sent_aligns))
        if self.debug:
            print("Align time: {}".format(datetime.now() - time_align))
        return alignments

    def train(self):
        if self.t_pretrained is None:
            t = np.random.rand(len(self.f_word_to_ix), len(self.e_word_to_ix))  # |t| = |f|*|e|
        else:
            t = self.t_pretrained

        precisions = np.zeros(self.num_epochs)
        recalls = np.zeros(self.num_epochs)
        aers = np.zeros(self.num_epochs)

        for epoch in range(self.num_epochs):

            time_epoch = datetime.now()

            # Initialize counters
            c_f_e = np.zeros((len(self.f_word_to_ix), len(self.e_word_to_ix)))
            c_f = np.zeros(len(self.f_word_to_ix))

            for sent_ix, (f_sent, e_sent) in enumerate(zip(self.f_data, self.e_data)):
                s_c_e = t[np.ix_(f_sent, e_sent)].sum(0)
                delta = t[np.ix_(f_sent, e_sent)]/s_c_e
                delta_sum = delta.sum(1)
                np.add.at(c_f_e, np.ix_(f_sent, e_sent), delta)
                np.add.at(c_f, f_sent, delta_sum)

            if self.smoothing_coef != 0:
                np.add.at(c_f_e, np.where(c_f_e != 0), self.smoothing_coef)
                np.add.at(c_f, np.where(c_f != 0), self.smoothing_coef*len(self.e_word_to_ix))

            t = (c_f_e.T / c_f).T

            if epoch % 5 == 0:
                print('Model {}\tEpoch #{}/{}'.format('IBM_M1', epoch, self.num_epochs))

            if self.debug:
                print("Epoch time: {}".format(datetime.now() - time_epoch))

                time_testing = datetime.now()

                alignments = self.align(t)
                precision, recall, aer = evaluate(alignments, self.test_data)

                precisions[epoch] = precision
                recalls[epoch] = recall
                aers[epoch] = aer

                print("Testing time: {}".format(datetime.now() - time_testing))
                print("Epoch: {},\tPrecision: {},\tRecall: {},\tAER = {}".format(epoch, precision, recall, aer))

        return t, precisions, recalls, aers


class IBM_M2():
    def __init__(self, f_word_to_ix, e_word_to_ix, f_data, e_data, f_max, e_max, test_data, num_epochs, debug=False,
                 t_pretrained=None):
        super(IBM_M2, self).__init__()

        self.f_word_to_ix = f_word_to_ix
        self.e_word_to_ix = e_word_to_ix

        self.f_data = f_data
        self.e_data = e_data
        self.test_data = test_data

        self.f_max = f_max
        self.e_max = e_max

        self.t_pretrained = t_pretrained

        self.num_epochs = num_epochs
        self.debug = debug

    def align(self, parameters):
        t, q = parameters
        time_align = datetime.now()
        alignments = []
        for f_sent, e_sent in zip(self.f_data, self.e_data):
            sent_aligns = []
            for i, f in enumerate(f_sent):
                max_val = 0
                max_pos = 0
                max_idx = 0
                for j, e in enumerate(e_sent):
                    prod = t[f][e] * q[i][j]
                    if prod > max_val:
                        max_val = prod
                        max_pos = j
                        max_idx = e
                if NULL in self.e_word_to_ix:
                    if max_idx != self.e_word_to_ix[NULL]:
                        sent_aligns.append((i, max_pos))
                else:
                    sent_aligns.append((i, max_pos))
            alignments.append(set(sent_aligns))

        if self.debug:
            print("Align time: {}".format(datetime.now() - time_align))
        return alignments

    def train(self):
        if self.t_pretrained is None:
            t = np.random.rand(len(self.f_word_to_ix), len(self.e_word_to_ix))  # |t| = |f|*|e|
        else:
            t = self.t_pretrained
        q = np.random.rand(self.f_max, self.e_max)  # |q| = max(I)*max(J)

        precisions = np.zeros(self.num_epochs)
        recalls = np.zeros(self.num_epochs)
        aers = np.zeros(self.num_epochs)

        for epoch in range(self.num_epochs):

            time_epoch = datetime.now()

            # Initialize counters
            c_f_e = np.zeros((len(self.f_word_to_ix), len(self.e_word_to_ix)))
            c_f = np.zeros(len(self.f_word_to_ix))
            c_j_i = np.zeros((self.f_max, self.e_max))
            c_j = np.zeros(self.f_max)

            for sent_ix, (f_sent, e_sent) in enumerate(zip(self.f_data, self.e_data)):
                # Extract relevant indices
                data_idx = np.ix_(f_sent, e_sent)
                len_idx = np.ix_(range(len(f_sent)), range(len(e_sent)))

                # Compute helper matrices
                tq = t[data_idx] * q[len_idx]
                sum = tq.sum(0)
                delta = tq/sum
                delta_sum = delta.sum(1)

                # Update counts
                np.add.at(c_f_e, data_idx, delta)
                np.add.at(c_f, f_sent, delta_sum)
                np.add.at(c_j_i, len_idx, delta)
                np.add.at(c_j, range(len(f_sent)), delta_sum)

            t = (c_f_e.T / c_f).T
            q = (c_j_i.T / c_j).T

            if epoch % 5 == 0:
                print('Model {}\tEpoch #{}/{}'.format('IBM_M2', epoch, self.num_epochs))

            if self.debug:
                print("Epoch time: {}".format(datetime.now() - time_epoch))

                time_testing = datetime.now()

                alignments = self.align((t, q))
                precision, recall, aer = evaluate(alignments, self.test_data)

                precisions[epoch] = precision
                recalls[epoch] = recall
                aers[epoch] = aer

                print("Testing time: {}".format(datetime.now() - time_testing))

                print("Epoch: {},\tPrecision: {},\tRecall: {},\tAER = {}".format(epoch, precision, recall, aer))

        return (t, q), precisions, recalls, aers