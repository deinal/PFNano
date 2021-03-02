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

variables = {
    'jets': ['Jet_pt', 'Jet_genJetIdx', 'Jet_rawFactor'],
    'gen_jets': ['GenJet_pt', 'GenJet_eta'],
    'jet_pf_cands': ['JetPFCands_pt', 'JetPFCands_jetIdx', 'JetPFCands_pFCandsIdx', 
                     'JetPFCands_btagEtaRel', 'JetPFCands_btagJetDistVal', 'JetPFCands_btagPParRatio',
                     'JetPFCands_btagPtRatio', 'JetPFCands_btagSip3dSig', 'JetPFCands_btagSip3dVal'],
    'pf_cands': ['PFCands_charge', 'PFCands_d0', 'PFCands_d0Err', 'PFCands_dz', 
                 'PFCands_dzErr', 'PFCands_eta', 'PFCands_lostInnerHits', 'PFCands_mass', 
                 'PFCands_pdgId', 'PFCands_phi', 'PFCands_pt', 'PFCands_puppiWeight',
                 'PFCands_puppiWeightNoLep', 'PFCands_pvAssocQuality', 'PFCands_trkChi2', 
                 'PFCands_trkQuality', 'PFCands_vtxChi2']

}

flat_variables = [val for sublist in variables.values() for val in sublist]

for path in glob.glob(f'{args.indir}/*.root'):
    name = path.split('/')[1].split('.root')[0]

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

    with open(f'{args.outdir}/{name}.pkl', 'wb') as handle:
        pickle.dump(result, handle)
