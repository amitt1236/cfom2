import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_clf, add_attachment_points, get_fp


INTERACTION_TYPE = {'B': 1, 'F': 2, 'A': 3}
UNKNOWN_TYPE = 4
LABEL_MAP = {0: -1, 1: 1}  # negative label should not be zero due to classifier free guidance


class InteractionsDataset(Dataset):
    def __init__(self, smiles, smiles_backbones, smiles_chains, assay_ids, assay_types, labels, protein_embeddings,
                 tokenizer, use_backbone=True):
        self.model_inputs = []
        self.model_outputs = []
        self.interactions = []
        self.labels = []
        for cur_smiles, cur_backbone, cur_chains, cur_assay_id, cur_assay_type, cur_label in zip(smiles, smiles_backbones, smiles_chains, assay_ids, assay_types, labels):
            if cur_backbone == 'None':
                continue
            cur_protein = protein_embeddings[cur_assay_id]
            cur_interaction = np.append(
                np.array([LABEL_MAP[cur_label], INTERACTION_TYPE.get(cur_assay_type, UNKNOWN_TYPE)]), cur_protein)

            if use_backbone:
                cur_tok_chain = np.array(tokenizer.encode(cur_chains).ids)
                cur_tok_backbone = np.array(tokenizer.encode(cur_backbone).ids)
                self.model_inputs.append(torch.from_numpy(cur_tok_backbone).long())
                self.model_outputs.append(torch.from_numpy(cur_tok_chain).long())
                self.interactions.append(torch.from_numpy(cur_interaction).float())
            else:
                cur_fp = get_fp(cur_smiles)
                cur_tok_smiles = np.array(tokenizer.encode(cur_smiles).ids)
                self.model_inputs.append(torch.from_numpy(cur_fp).float())
                self.model_outputs.append(torch.from_numpy(cur_tok_smiles).long())
                self.interactions.append(torch.from_numpy(cur_interaction).float())
            self.labels.append(cur_label)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, item):
        return self.model_inputs[item], self.model_outputs[item], self.interactions[item], self.labels[item]


class FSMolDataSet(Dataset):
    def __init__(self, smiles, smiles_backbones, assay_ids, assay_types, labels, protein_embeddings, tokenizer,
                 calc_rf=False, use_backbone=False, min_clf_samples=300, min_roc_auc=0.75, num_sites=2, seed=0):
        self.assays = {}
        a = 0
        for cur_smiles, cur_backbone, cur_assay_id, cur_assay_type, cur_label in zip(smiles, smiles_backbones, assay_ids, assay_types, labels):
            if cur_assay_id not in self.assays:
                cur_protein = protein_embeddings[cur_assay_id]
                self.assays[cur_assay_id] = {'active': [], 'active_backbone': [], 'inactive': [],
                                             'inactive_backbone': [], 'assay_id': cur_assay_id, 'protein': cur_protein,
                                             'active_smiles': [], 'inactive_smiles': [],
                                             'assay_type': INTERACTION_TYPE.get(cur_assay_type, UNKNOWN_TYPE)}
            if use_backbone:
                if isinstance(cur_backbone, float) and np.isnan(cur_backbone):
                    print(a)
                    a += 1
                    continue
                if cur_backbone == 'None':
                    cur_backbone = add_attachment_points(cur_smiles, n=num_sites, seed=seed)
                model_input = np.array(tokenizer.encode(cur_backbone).ids)
                model_input = torch.from_numpy(model_input).long()
            else:
                model_input = get_fp(cur_smiles)
                model_input = torch.from_numpy(model_input).float()
            if cur_label == 0:
                self.assays[cur_assay_id]['inactive'].append(model_input)
                self.assays[cur_assay_id]['inactive_backbone'].append(cur_backbone)
                self.assays[cur_assay_id]['inactive_smiles'].append(cur_smiles)
            else:
                self.assays[cur_assay_id]['active'].append(model_input)
                self.assays[cur_assay_id]['active_backbone'].append(cur_backbone)
                self.assays[cur_assay_id]['active_smiles'].append(cur_smiles)

        self.tasks = [assay for assay in self.assays.values()]
        if calc_rf:
            for task in self.tasks:
                if len(task['active']) + len(task['inactive']) > min_clf_samples:
                    positive_smiles = [cur_smile for cur_smile in task['active_smiles']]
                    negative_smiles = [cur_smile for cur_smile in task['inactive_smiles']]
                    clf, roc_auc, fpr, tpr, thresholds = get_clf(positive_smiles, negative_smiles)
                    if roc_auc > min_roc_auc:
                        task['clf'] = clf
                        task['threshold'] = thresholds[np.argmax(tpr - fpr)]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):
        return None
