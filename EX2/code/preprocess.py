# Itamar Trainin 315425967

import settings
import torch
import random

sent_sep = '\n'
word_sep = ' '


def make_dict(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        return {v: i for i, v in enumerate(set([settings.NULL, settings.SENT_BEG, settings.SENT_END] + f.read().replace(sent_sep, '').split(word_sep)[:-1]))}


def read_data(fname, word_to_ix, add_unk=False):
    data = []
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.read().split(sent_sep)
        for line in lines:
            data.append(read_line(line, word_to_ix, add_unk))

    data = data

    return data[:-1]


def read_line(line, word_to_ix, add_unk=False):
    return torch.tensor([_get_token(word, word_to_ix, add_unk) for word in [settings.SENT_BEG] + line[:-1].split(word_sep) + [settings.SENT_END]])


def _should_unk(dist):
    return random.randint(1, 1 / settings.unk_dist) == 1


def _get_token(word, word_to_ix, add_unk):
    if add_unk and _should_unk(settings.unk_dist):
        return word_to_ix[settings.NULL]
    elif word not in word_to_ix:
        return word_to_ix[settings.NULL]
    else:
        return word_to_ix[word]
