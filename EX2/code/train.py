# Itamar Trainin 315425967

import optparse
import torch
import torch.nn as nn
import torch.optim as optim
from seq2seq import Seq2Seq
import settings
from tqdm import tqdm
import evaluate
import sacrebleu
import os
import copy
import preprocess
from pathlib import Path
import random
from datetime import datetime
import heatmap


if __name__ == '__main__':

    print(datetime.now().isoformat())

    optparser = optparse.OptionParser()

    optparser.add_option("--ts", dest="train_src", default="../data/train.src", help="Train source filename")
    optparser.add_option("--tt", dest="train_trg", default="../data/train.trg", help="Train target filename")
    optparser.add_option("--ds", dest="dev_src", default="../data/dev.src", help="Dev source filename")
    optparser.add_option("--dt", dest="dev_trg", default="../data/dev.trg", help="Dev target filename")
    optparser.add_option("--od", dest="output_dir", default="../models", help="Output dir")

    optparser.add_option("--embd", dest="embd_dim", default=None, help="Embedding dimension", type='int')
    optparser.add_option("--hidden", dest="hidden_dim", default=None, help="Hidden dimension", type='int')
    optparser.add_option("--out", dest="output_dim", default=None, help="Output dimension", type='int')
    optparser.add_option("-t", dest="task", default="no_attention", help="Output dir")

    optparser.add_option("--src", dest="src", default="40 116 104 101 110 ", help="Test source filename")
    optparser.add_option("--trg", dest="trg", default="( t h e n ", help="Test target filename")

    (opts, args) = optparser.parse_args()
    print('opts: {}'.format(opts))

    if opts.embd_dim:
        settings.embd_dim = opts.embd_dim
    if opts.hidden_dim:
        settings.hidden_dim = opts.hidden_dim
    if opts.output_dim:
        settings.output_dim = opts.output_dim

    if opts.task == 'no_attention':
        attention = False
    elif opts.task == 'attention':
        attention = True
    else:
        raise Exception('invalid task type.')

    print('embd_dim: {}\thidden_dim: {}\toutput_dim: {}\tlr: {}'.format(settings.embd_dim, settings.hidden_dim, settings.output_dim, settings.lr))

    Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    word_to_ix_src = preprocess.make_dict(opts.train_src)
    word_to_ix_trg = preprocess.make_dict(opts.train_trg)

    X_train = preprocess.read_data(opts.train_src, word_to_ix_src, add_unk=settings.add_unk)
    Y_train = preprocess.read_data(opts.train_trg, word_to_ix_trg, add_unk=settings.add_unk)

    X_dev = preprocess.read_data(opts.dev_src, word_to_ix_src)
    Y_dev = preprocess.read_data(opts.dev_trg, word_to_ix_trg)

    ix_to_word_src = {v: k for k, v in word_to_ix_src.items()}
    ix_to_word_trg = {v: k for k, v in word_to_ix_trg.items()}

    model = Seq2Seq(word_to_ix_src,
                    word_to_ix_trg,
                    settings.embd_dim,
                    settings.hidden_dim,
                    settings.output_dim,
                    settings.num_layers,
                    attention)
    # optim = optim.SGD(model.parameters(), lr=settings.lr)
    # optim = optim.Adam(model.parameters(), lr=settings.lr)
    optim = optim.Adam(model.parameters())
    nllloss = nn.NLLLoss()

    train_bleu_scores = []
    train_loss_scores = []

    dev_bleu_scores = []
    dev_loss_scores = []

    sent_src = []
    sents_pred = []
    sents_true = []

    model_cps = []

    train_size = len(X_train)
    indices = list(range(train_size))
    for epoch in range(settings.num_epochs):
        total_loss = 0.0
        random.shuffle(indices)
        for ix in tqdm(indices, desc='Train - epoch {}'.format(epoch)):

            x = X_train[ix]
            y = Y_train[ix]
            y_true = y[1:]

            y_pred, loss = model.forward_train(x, y_true)
            total_loss += loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            x_sent = evaluate.to_sent(x, ix_to_word_src)
            sent_src.append(x_sent)

            y_pred_sent = evaluate.to_sent(y_pred, ix_to_word_trg)
            sents_pred.append(y_pred_sent)

            y_sent = evaluate.to_sent(y, ix_to_word_trg)
            sents_true.append(y_sent)

            if settings.debug_train:
                print('Train: x: ' + x_sent)
                print('Train: y_pred: ' + y_pred_sent)
                print('Train: y_true: ' + y_sent)
                print()

        dev_loss, dev_bleu = evaluate.score(model, X_dev, Y_dev, ix_to_word_src, ix_to_word_trg, debug=settings.debug_eval)

        total_loss = total_loss / train_size
        train_loss_scores.append(total_loss)
        total_bleu = sacrebleu.corpus_bleu(sents_pred, [sents_true]).score
        train_bleu_scores.append(total_bleu)
        dev_loss_scores.append(dev_loss)
        dev_bleu_scores.append(dev_bleu)

        if attention:
            x_heatmap = preprocess.read_line(opts.src, word_to_ix_src)
            y_heatmap = preprocess.read_line(opts.trg, word_to_ix_trg)

            heatmap.show_heat_map(model, x_heatmap, y_heatmap, ix_to_word_src, ix_to_word_trg, opts.output_dir, epoch)

        model_cps.append(copy.deepcopy(model))

        print('\nEpoch {}/{}\ttrain-loss: {:.4f}, train-bleu: {:.4f}, dev-loss: {:.4f}, dev-bleu: {:.4f}'.format(epoch + 1, settings.num_epochs, total_loss, total_bleu, dev_loss, dev_bleu))

    fname = os.path.join(opts.output_dir, 'model_' + opts.task + '_' + str(round(max(dev_bleu_scores))) + '_' + str(settings.embd_dim) + '_' + str(settings.hidden_dim) + '_' + str(settings.output_dim) + '_' + str(settings.lr))
    torch.save((model_cps[int(torch.tensor(dev_bleu_scores).argmax())], word_to_ix_src, word_to_ix_trg), fname)

    print('train_bleu_scores = {}'.format(train_bleu_scores))
    print('train_loss_scores = {}'.format(train_loss_scores))
    print('dev_bleu_scores = {}'.format(dev_bleu_scores))
    print('dev_loss_scores = {}'.format(dev_loss_scores))
