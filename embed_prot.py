import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoModel, pipeline
import re
import numpy as np
import json
from pathlib import Path
import os
import requests
from tqdm.auto import tqdm
import gc


if __name__ == '__main__':
    model_name = 'prot_t5_xl_uniref50'
    data_split = 'valid'
    tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False)
    model = T5EncoderModel.from_pretrained(f"Rostlab/{model_name}")
    gc.collect()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

    with open(f'./data/fsmol/prot_map_{data_split}.json', 'r') as f:
        prot_map = json.load(f)

    prots, assays = [], []
    for key, val in prot_map.items():
        prots.append(val)
        assays.append(key)

    seqs = [[c for c in prot] for prot in prots]
    spaced_seqs = [' '.join(seq) for seq in seqs]
    filtered_seqs = [re.sub(r"[UZOB]", "X", sequence) for sequence in spaced_seqs]
    idx = 0
    batch_size = 1
    all_embeds = []
    while idx < len(filtered_seqs):
        end = min(idx + batch_size, len(filtered_seqs))
        batch = filtered_seqs[idx: end]
        idx = end
        ids = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        features = []

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)

        protein_features = [np.mean(tokens, axis=0) for tokens in features]
        all_embeds.extend(protein_features)
    to_save = {}
    for assay_id, embed in zip(assays, all_embeds):
        to_save[assay_id] = embed
    np.savez_compressed(f'./non_averaged_prot_embeds_fsmol_{data_split}', **to_save)
