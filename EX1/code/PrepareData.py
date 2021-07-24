# Author: Itamar Trainin 315425967

import os
import optparse
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
nltk.download('punkt')

NULL = '<NULL>'
PUNCT = '<PUNCT>'
STOP = '<STOP>'
UNKNOWN = '<UKN>'

use_stopwords = False
use_stemming = True
use_punct = True

snowball = {
    'english': SnowballStemmer("english"),
    'french': SnowballStemmer("french")
}


def get_words(sent, lang):
    words = sent.strip().split()
    return [clean_word(word, lang) for word in words]


def clean_word(word, lang):
    if use_punct and word in string.punctuation:
        return PUNCT
    elif use_stopwords and (word in stopwords.words(lang)):
        return STOP
    else:
        if use_stemming:
            return snowball[lang].stem(word.lower())
        else:
            return word.lower()


def get_lang(fname):
    return 'french' if fname.split('.')[-1] == 'f' else 'english'


def make_dict(fname, null_word=False, line_limit=-1, override=False):
    if override:
        os.remove(fname + '.dict')
    try:
        word_to_ix = pickle.load(open(fname + '.dict', 'rb'))
        print('Word to index of {} has been read from disk.'.format(fname))
    except:
        word_to_ix = {}
        lang = get_lang(fname)
        for sent_ix, sent in enumerate(open(fname)):
            if line_limit != -1 and sent_ix > line_limit:
                break
            words = get_words(sent, lang)
            for word in words:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        if null_word:
            word_to_ix[NULL] = len(word_to_ix)

        pickle.dump(word_to_ix, open(fname + '.dict', 'wb'))
        print('Word to index of {} has been created and saved.'.format(fname))

    return word_to_ix


def read_data(fname, dictionary, null_word=False, line_limit=-1, override=False):
    if override:
        os.remove(fname + '.data')
    try:
        data, max_sent = pickle.load(open(fname + '.data', 'rb'))
        print('Data from {} has been read from disk.'.format(fname))
    except:
        data = []
        max_sent = 0
        lang = get_lang(fname)

        # Compute data matrix, that is for each line and word offset the index of the word.
        for sent_ix, sent in enumerate(open(fname)):
            if line_limit != -1 and sent_ix > line_limit:
                break
            sent_data = []
            words = get_words(sent, lang)
            if null_word:
                words = words + [NULL]
            if len(words) > max_sent:
                max_sent = len(words)
            for word in words:
                sent_data.append(dictionary[word])
            data.append(sent_data)

        pickle.dump((data, max_sent), open(fname + '.data', 'wb'))
        print('Data from {} has been created and saved.'.format(fname))

    return data, max_sent


def read_test(fname):
    try:
        sure, possible = pickle.load(open(fname + '.test.data', 'rb'))
        print('Test data from {} has been read from disk.'.format(fname))
    except:
        sure = []
        possible = []
        for i, g in enumerate(open(fname)):
            sure.append(set([tuple(map(int, x.split("-"))) for x in filter(lambda x: x.find("-") > -1, g.strip().split())]))
            possible.append(set([tuple(map(int, x.split("?"))) for x in filter(lambda x: x.find("?") > -1, g.strip().split())]))

        pickle.dump((sure, possible), open(fname + '.test.data', 'wb'))
        print('Test data from {} has been computed and saved.'.format(fname))

    return sure, possible


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-f", dest="f_file", default="data/hansards.f", help="F filename")
    optparser.add_option("-e", dest="e_file", default="data/hansards.e", help="E filename")
    optparser.add_option("-a", dest="a_file", default="data/hansards.a", help="Gold alignments filename")
    (opts, args) = optparser.parse_args()

    f_word_to_ix = make_dict(opts.f_file)
    e_word_to_ix = make_dict(opts.e_file, null_word=True, override=True)

    f_data = read_data(opts.f_file, f_word_to_ix)
    e_data = read_data(opts.e_file, e_word_to_ix, null_word=True, override=True)

    sure, possible = read_test(opts.a_file)

    print('Done')
