# Author: Itamar Trainin 315425967

import optparse
import IBM_Models
from PrepareData import make_dict, read_data, read_test
import numpy as np
from datetime import datetime
from pathlib import Path
import os

# https://docs.google.com/document/d/16NOAMZ836C-AecSH8FWH-v6iDoELGFPXs8kJeRpzQcU/edit?ts=5eb10dd6#heading=h.bna1re8nv3k0

if __name__ == "__main__":

    total_time = datetime.now()

    optparser = optparse.OptionParser()
    optparser.add_option("-f", dest="f_file", default="data/hansards.f", help="F filename")
    optparser.add_option("-e", dest="e_file", default="data/hansards.e", help="E filename")
    optparser.add_option("-a", dest="a_file", default="data/hansards.a", help="Gold alignments filename")
    optparser.add_option("-t", dest="t_pretrained", default=None, help="Path to pre-trained t parameters (.npy)")
    optparser.add_option("--out", dest="output_dir", default="data/output", help="Dir to save outputs")
    optparser.add_option("-d", dest="debug", action="store_true", default=False, help="Print debugging output")
    optparser.add_option("-m", dest="model", default="1", type="int", help="Index of IBM Model to run (1/2)")
    optparser.add_option("--ep", dest="num_epochs", default="100", type="int", help="Number of epochs")
    optparser.add_option("-n", dest="n", default="0", type="float", help="Smoothing coefficient")
    optparser.add_option("-o", dest="override", action="store_true", default=False, help="Override data")
    optparser.add_option("--ll", dest="lines_limit", default="-1", type="int", help="Number data lines to use")
    (opts, args) = optparser.parse_args()

    print('opts: {}'.format(opts))

    f_word_to_ix = make_dict(opts.f_file, line_limit=opts.lines_limit, override=opts.override)
    e_word_to_ix = make_dict(opts.e_file, line_limit=opts.lines_limit, override=opts.override, null_word=True)

    f_data, f_max = read_data(opts.f_file, f_word_to_ix, line_limit=opts.lines_limit, override=opts.override)
    e_data, e_max = read_data(opts.e_file, e_word_to_ix, line_limit=opts.lines_limit, override=opts.override,
                              null_word=True)

    sure, possible = read_test(opts.a_file)

    if opts.t_pretrained:
        t_pretrained = np.load(opts.t_pretrained)
        print('Pre-trained t was found and loaded.')
    else:
        t_pretrained = None
        print('No pre-training used.')

    if opts.model == 1:
        print('Running IBM Model 1')
        output_dir = os.path.join(opts.output_dir, 'model_1')
        model = IBM_Models.IBM_M1(f_word_to_ix, e_word_to_ix, f_data, e_data, f_max, e_max, (sure, possible),
                                  opts.num_epochs, opts.debug, t_pretrained, opts.n)
    else:
        print('Running IBM Model 2')
        output_dir = os.path.join(opts.output_dir, 'model_2')
        model = IBM_Models.IBM_M2(f_word_to_ix, e_word_to_ix, f_data, e_data, f_max, e_max, (sure, possible),
                                  opts.num_epochs, opts.debug, t_pretrained)

    train_time = datetime.now()

    parameters, precisions, recalls, aers = model.train()

    print('Train time: {}'.format(datetime.now() - train_time))

    alignments = model.align(parameters)
    precision, recall, aer = IBM_Models.evaluate(alignments, (sure, possible))

    print("Precision: {},\tRecall: {},\tAER = {}".format(precision, recall, aer))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    IBM_Models.save_alignments(output_dir, alignments)

    np.save(output_dir + '/precision.eval_{}.data'.format(datetime.now().strftime('%d%m%y_%H%M')), precisions)
    print('Precisions saved.')
    np.save(output_dir + '/recall.eval_{}.data'.format(datetime.now().strftime('%d%m%y_%H%M')), recalls)
    print('Recalls saved.')
    np.save(output_dir + '/aer.eval_{}.data'.format(datetime.now().strftime('%d%m%y_%H%M')), aers)
    print('AERs saved')
    np.save(output_dir + '/parameters_{}_{}'.format(opts.model, datetime.now().strftime('%d%m%y_%H%M')), parameters)
    print('Parameters saved')

    print('Done- total time: {}'.format(datetime.now() - total_time))
