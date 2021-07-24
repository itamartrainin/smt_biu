# Itamar Trainin 315425967

import matplotlib.pyplot as plt
import optparse
import torch
import preprocess
import evaluate
import numpy as np
import os
import settings


def show_heat_map(model, x, y, ix_to_word_src, ix_to_word_trg, output=None, epoch=None):

    fname = 'model_attention_ep' + str(epoch) + '_' + str(settings.embd_dim) + '_' + str(settings.hidden_dim) + '_' + str(settings.output_dim) + '_' + str(settings.lr)

    y_true = y[1:]
    y_pred, _, weights = model.forward_predict(x, y_true)

    x_sent = evaluate.to_sent(x, ix_to_word_src).split(preprocess.word_sep)[1:-1]
    y_sent = evaluate.to_sent(y, ix_to_word_trg).split(preprocess.word_sep)[1:-1]

    fig, ax = plt.subplots()

    weights = np.array(list(map(lambda weight: weight.squeeze(1).detach().numpy()[1:-1], weights)))
    np.save(os.path.join(output, fname + '.npy'), weights)
    im = ax.imshow(weights)

    ax.set_xticks(np.arange(len(x_sent)))
    ax.set_yticks(np.arange(len(y_sent)))

    ax.set_xticklabels(x_sent)
    ax.set_yticklabels(y_sent)

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(x_sent)):
        for j in range(len(y_sent)):
            text = ax.text(i, j, '{:.2}'.format(weights[j, i]),
                           ha="center", va="center", color="w")

    ax.set_title("attention-based alignment: {} -> {}".format(preprocess.word_sep.join(x_sent), preprocess.word_sep.join(y_sent)))
    fig.tight_layout()

    if output and epoch != None:
        plt.savefig(os.path.join(output, fname + '.png'))
    else:
        plt.show()


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("--src", dest="src", default="66 111 110 100 ", help="Test source filename")
    optparser.add_option("--trg", dest="trg", default="B o n d ", help="Test target filename")
    optparser.add_option("--mf", dest="model_file", default="../models/model_attention_93_50_200_500_0.001", help="Path to model file")
    optparser.add_option("--od", dest="output_dir", default="../models", help="Output dir")
    (opts, args) = optparser.parse_args()
    print('opts: {}'.format(opts))

    model, word_to_ix_src, word_to_ix_trg = torch.load(opts.model_file)

    x = preprocess.read_line(opts.src, word_to_ix_src)
    y = preprocess.read_line(opts.trg, word_to_ix_trg)

    ix_to_word_src = {v: k for k, v in word_to_ix_src.items()}
    ix_to_word_trg = {v: k for k, v in word_to_ix_trg.items()}

    show_heat_map(model, x, y, ix_to_word_src, ix_to_word_trg, opts.output_dir)
