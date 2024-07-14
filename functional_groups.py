import itertools
import numpy as np
import rdkit
import rdkit.Chem
import rdkit.Chem.Draw
from datasets import FSMolDataSet
import pickle
from pathlib import Path
from utils import read_csv, load_tokenizer_from_file, smiles_valid, calc_tani_sim, get_fp


def find_matched_bonds(mol, matched_atoms):
    the_bonds = set([mol.GetBondBetweenAtoms(a, b)
                     for (a, b) in itertools.combinations(matched_atoms, 2)])
    return [x.GetIdx() for x in the_bonds if x is not None]


# return boolean list whether corresponding functional group is found
def find_functional_groups(mol, fg_list):
    return [len(mol.GetSubstructMatches(rdkit.Chem.MolFromSmarts(the_fg_target))) > 0
            for the_fg_target in fg_list]


# create image of molecule with all matches for functional groups
# in fg_list highlighted in highlight_color
def highlit_image(mol, fg_list, highlight_color=(0.5, 1.0, 1.0), the_size=(500, 500)):
    the_matches = [mol.GetSubstructMatches(rdkit.Chem.MolFromSmarts(the_fg_target)) for the_fg_target in fg_list]
    all_highlighted_atoms = set()
    all_highlighted_bonds = set()
    for the_match in the_matches:
        for the_submatch in the_match:
            all_highlighted_atoms.update(the_submatch)
            matched_bonds = find_matched_bonds(mol, the_submatch)
            all_highlighted_bonds.update(matched_bonds)
    highlighted_image = rdkit.Chem.Draw.MolToImage(mol,
                                                   size=the_size,
                                                   highlightColor=highlight_color,
                                                   highlightAtoms=list(all_highlighted_atoms),
                                                   highlightBonds=list(all_highlighted_bonds))
    highlighted = True if len(all_highlighted_atoms) > 0 or len(all_highlighted_bonds) > 0 else False
    return highlighted_image, highlighted


if __name__ == '__main__':
    with open('fg_trans_dict.pkl', 'rb') as handle:
        fg = pickle.load(handle)
    test_non_chiral_smiles, test_backbones, test_chains, test_assay_ids, test_types, test_labels = read_csv('./data/fsmol/test.csv')
    filtered_fg = list(fg.values())[1:10]
    filtered_assays = ['CHEMBL1119333', 'CHEMBL1614027', 'CHEMBL1614423', 'CHEMBL1738485', 'CHEMBL1963715',
                       'CHEMBL1963723', 'CHEMBL1963731', 'CHEMBL1963741', 'CHEMBL1963756', 'CHEMBL1963810',
                       'CHEMBL1963818', 'CHEMBL1963819', 'CHEMBL1963824', 'CHEMBL1963825', 'CHEMBL1963827',
                       'CHEMBL1963831', 'CHEMBL1964101', 'CHEMBL1964115', 'CHEMBL3214944', 'CHEMBL3431930',
                       'CHEMBL3431932']
    opts_dict = {}
    path = './results/'
    for assay in filtered_assays:
        with open(f'{path}/{assay}/samples.txt', 'r') as f:
            all_pairs = [line.replace('\n', '').split(' ') for line in f.readlines()]
        opt_samples = {}
        for pair in all_pairs:
            if pair[0] in opt_samples:
                opt_samples[pair[0]].append(pair[1])
            else:
                opt_samples[pair[0]] = [pair[1]]
        opts_dict[assay] = opt_samples

    train_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_train.npz')
    valid_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_valid.npz')
    test_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_test.npz')
    tokenizer = load_tokenizer_from_file(f'./models/tokenizer_object.json')
    test_ds = FSMolDataSet(test_non_chiral_smiles, test_backbones, test_assay_ids, test_types, test_labels,
                           test_prot_embeds, tokenizer, calc_rf=True)

    tasks = [task for task in test_ds.tasks if task.get('clf', None) is not None]
    tasks_dict = {task['assay_id']: task for task in tasks}
    cur_num_examples, examples = 0, 1000
    for cur_smiles, label, assay_id in zip(test_non_chiral_smiles, test_labels, test_assay_ids):
        if label == 1 or assay_id not in filtered_assays or opts_dict[assay_id].get(cur_smiles, None) is None:
            continue
        mol = rdkit.Chem.MolFromSmiles(cur_smiles)
        res, highlighted = highlit_image(mol, filtered_fg, highlight_color=(1., 1., 0.))
        if not highlighted:
            for optimized_smiles in opts_dict[assay_id][cur_smiles]:
                if not smiles_valid(optimized_smiles) or optimized_smiles == cur_smiles:
                    continue
                clf = tasks_dict[assay_id]['clf']
                cur_score = clf.predict_proba(np.reshape(get_fp(optimized_smiles), (1, -1)))
                cur_score = cur_score[0][1]
                sim = calc_tani_sim(cur_smiles, optimized_smiles)
                if sim > 0.55 and cur_score > tasks_dict[assay_id]['threshold']:
                    mol2 = rdkit.Chem.MolFromSmiles(optimized_smiles)
                    res2, highlighted2 = highlit_image(mol2, filtered_fg, highlight_color=(1., 1., 0.))
                    if highlighted2:
                        sim = calc_tani_sim(cur_smiles, optimized_smiles)
                        res.save(f'./fg_output/example_{cur_num_examples}_inp_{assay_id}_sim_{sim}.png')
                        res2.save(f'./fg_output/example_{cur_num_examples}_opt_{assay_id}_sim_{sim}.png')
                        cur_num_examples += 1
                        if cur_num_examples > examples:
                            exit()
                        else:
                            break