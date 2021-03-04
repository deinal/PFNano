"""
Select variables in NanoAOD root files and convert from event-by-event format 
to a jet-by-jet dataframe for further ml analyzes
"""

import os
import argparse
import glob
import uproot
import awkward as ak
import pandas as pd
import pickle
import yaml


def jetter(outdir, root_files, variables):
    flat_variables = [val for sublist in variables.values() for val in sublist]

    for path in root_files:
        name = path.split('/')[-1].split('.root')[0]

        with uproot.open(path) as file:
            events = file['Events'].arrays(flat_variables)

        # Concatenate leading jets vertically
        jets = events[variables['jets']]
        jets = ak.concatenate((jets[:,0], jets[:,1]), axis=0)
        jets_df = ak.to_pandas(jets)

        # Sort gen jets according to the jet collection and concatenate leading gen jets vertically
        gen_jets = events[variables['gen_jets']][events.Jet_genJetIdx]
        gen_jets = ak.concatenate((gen_jets[:,0], gen_jets[:,1]), axis=0)
        gen_jets_df = ak.to_pandas(gen_jets)

        # Concatentate jets and gen jets horizontally
        globals = pd.concat((jets_df, gen_jets_df), axis=1)

        # Select jet pf cands
        jet_pf_cands = events[variables['jet_pf_cands']]
        jet_pf_cands_df = ak.to_pandas(jet_pf_cands)

        # Sort pf cands according to jet pf cands collection
        pf_cands = events[variables['pf_cands']][events.JetPFCands_pFCandsIdx]
        pf_cands_df = ak.to_pandas(pf_cands)

        # Concatenate pf cands and jet pf cands horizontally
        constituents = pd.concat((pf_cands_df, jet_pf_cands_df), axis=1)

        # Make a tuple of leading jet constituents and next-to-leading jet constituents 
        constituents = (constituents[constituents.JetPFCands_jetIdx == 0], constituents[constituents.JetPFCands_jetIdx == 1])

        # Create a new outer multi index for next-to-leading jet constituents in preparation for concatenation
        outer_index = constituents[1].index.get_level_values(0) + constituents[0].index[-1][0] + 1
        inner_index = constituents[1].index.get_level_values(1)
        constituents[1].index = pd.MultiIndex.from_tuples(zip(outer_index, inner_index), names=('entry', 'subentry'))

        # Concatenate leading jet constituents and next-to-leading jet constituents vertically
        constituents = pd.concat(constituents, axis=0)

        result = {
            'constituents': constituents,
            'globals': globals
        }

        with open(f'{outdir}/{name}.pkl', 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', required=True, help='Event-by-event NanoAOD root in directory')
    parser.add_argument('-o', '--outdir', required=True, help='Jet-by-jet pickle out directory')
    args = parser.parse_args()

    try:
        os.mkdir(args.outdir)
    except FileExistsError:
        pass

    with open('jet_ml_config.yaml') as f:
        variables = yaml.safe_load(f)['variables']

    flat_variables = [val for sublist in variables.values() for val in sublist]

    root_files = glob.glob(f'{args.indir}/*.root')

    jetter(args.outdir, root_files, variables)
