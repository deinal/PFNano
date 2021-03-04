import os
import argparse
import pickle


def sampler(infile, outdir, splits):

    with open(infile, 'rb') as f:
        data = pickle.load(f)

    globals = data['globals']
    constituents = data['constituents']

    for n in range(splits):
        globals_sample = globals.sample(frac=1/(splits-n))
        constituents_sample = constituents.loc[globals_sample.index, :]
        globals = globals.drop(globals_sample.index)

        data_sample = {
            'globals': globals_sample,
            'constituents': constituents_sample
        }

        with open(f'{outdir}/{n+1}.pkl', 'wb') as f:
                pickle.dump(data_sample, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help='Event-by-event NanoAOD root file')
    parser.add_argument('-o', '--outdir', required=True, help='Jet-by-jet pickle out directory')
    parser.add_argument('-s', '--splits', required=True, type=int, help='Number of sample splits')
    args = parser.parse_args()

    try:
        os.mkdir(args.outdir)
    except FileExistsError:
        pass

    sampler(args.infile, args.outdir, args.splits)
