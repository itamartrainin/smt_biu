# Itamar Trainin 315425967

import sacrebleu
import optparse
import torch
import preprocess
import os

def to_sent(x, ix_to_word):
    return ' '.join([ix_to_word[int(ix)] for ix in x])


def score(model, X, Y, ix_to_word_src, ix_to_word_trg, print_translations=False, debug=False):

    sent_src = []
    sents_pred = []
    sents_true = []
    losses = []

    for ix in range(len(X)):
        x = X[ix]
        y = Y[ix]
        y_true = y[1:]

        y_pred, loss, _ = model.forward_predict(x, y_true)
        _, loss = model.forward_train(x, y_true)

        losses.append(loss)

        x_sent = to_sent(x, ix_to_word_src)
        sent_src.append(x_sent)

        y_pred_sent = to_sent(y_pred, ix_to_word_trg)
        sents_pred.append(y_pred_sent)

        y_sent = to_sent(y, ix_to_word_trg)
        sents_true.append(y_sent)

        if debug:
            print('Eval x:\t{}\nEval y_pred:\t{}\nEval y_true:\t{}\n'.format(x_sent, y_pred_sent, y_sent))

    loss = float(sum(losses)) / float(len(losses))
    blue = sacrebleu.corpus_bleu(sents_pred, [sents_true]).score

    if print_translations:
        print('\n'.join(sents_pred))

    return loss, blue


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("--ts", dest="test_src", default="../data/test.src", help="Test source filename")
    optparser.add_option("--tt", dest="test_trg", default="../data/test.trg", help="Test target filename")
    optparser.add_option("--mf", dest="model_file", default="../models/model_no_attention_72_50_200_500_0.001", help="Path to model file")
    (opts, args) = optparser.parse_args()
    print('opts: {}'.format(opts))

    model, word_to_ix_src, word_to_ix_trg = torch.load(opts.model_file)

    X_test = preprocess.read_data(opts.test_src, word_to_ix_src)
    Y_test = preprocess.read_data(opts.test_trg, word_to_ix_trg)

    ix_to_word_src = {v: k for k, v in word_to_ix_src.items()}
    ix_to_word_trg = {v: k for k, v in word_to_ix_trg.items()}

    dev_loss, dev_bleu = score(model, X_test, Y_test, ix_to_word_src, ix_to_word_trg, print_translations=True, debug=False)

    print('\ntest-loss: {:.4f}, test-bleu: {:.4f}'.format(dev_loss, dev_bleu))

