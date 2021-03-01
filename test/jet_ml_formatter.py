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

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', required=True, help='Event-by-event NanoAOD root in directory')
parser.add_argument('-o', '--outdir', required=True, help='Jet-by-jet pickle out directory')
args = parser.parse_args()

try:
    os.makedirs(args.outdir)
except FileExistsError:
    pass

variables = ['GenJet_pt', 'GenJet_eta', 'Jet_pt', 'Jet_genJetIdx', 'JetPFCands_pt', 'Jet_rawFactor', 
             'JetPFCands_jetIdx', 'JetPFCands_pFCandsIdx', 'PFCands_mass', 'PFCands_eta']

for path in glob.glob(f'{args.indir}/*.root'):
    name = path.split('/')[1].split('.root')[0]

    with uproot.open(path) as file:
        events = file['Events'].arrays(variables)

    gen_jets = events[['GenJet_pt', 'GenJet_eta']][events.Jet_genJetIdx]
    gen_jets = ak.concatenate((gen_jets[:,0], gen_jets[:,1]), axis=0)
    gen_jets_df = ak.to_pandas(gen_jets)

    jets = events[['Jet_pt', 'Jet_genJetIdx', 'JetPFCands_pt', 'Jet_rawFactor']]
    jets = ak.concatenate((jets[:,0], jets[:,1]), axis=0)
    jets_df = ak.to_pandas(jets)

    globals = pd.concat((jets_df, gen_jets_df), axis=1)

    pf_cands = events[['PFCands_mass', 'PFCands_eta']][events.JetPFCands_pFCandsIdx]
    pf_cands_df = ak.to_pandas(pf_cands)

    jet_pf_cands = events[['JetPFCands_pt', 'JetPFCands_jetIdx', 'JetPFCands_pFCandsIdx']]
    jet_pf_cands_df = ak.to_pandas(jet_pf_cands)

    constituents = pd.concat((pf_cands_df, jet_pf_cands_df), axis=1)

    constituents = (constituents[constituents.JetPFCands_jetIdx == 0], constituents[constituents.JetPFCands_jetIdx == 1])
    outer_index = constituents[1].index.get_level_values(0) + constituents[0].index[-1][0] + 1
    inner_index = constituents[1].index.get_level_values(1)
    constituents[1].index = pd.MultiIndex.from_tuples(zip(outer_index, inner_index), names=('entry', 'subentry'))

    constituents = pd.concat(constituents, axis=0)

    result = {
        'constituents': constituents,
        'globals': globals
    }

    with open(f'{args.outdir}/{name}.pkl', 'wb') as handle:
        pickle.dump(result, handle)
