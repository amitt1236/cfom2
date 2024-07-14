import numpy as np
import pandas as pd
from pathlib import Path
import copy
import json
from datetime import datetime
from torch.utils.data import DataLoader
from datasets import InteractionsDataset, FSMolDataSet
from utils import read_csv, create_smiles_tokenizer, create_target_masks
from evaluate import generate_molecules
from models import InteractionEncoder, InteractionTranslator, TransformerEncoder, TransformerDecoder, \
    RNNBasedEncoder, RNNBasedDecoder
from evaluate import evaluate_task
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import sys


if __name__ == '__main__':
    cur_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    hyper_params = {
        'bs': 256,
        'lr': 0.0001,
        'weight_decay': 0.,
        'epochs': 100,
        'max_mol_len': 128,
        'embedding_dim': 128,
        'arch_type': 'transformer',  # can be 'transformer', 'gru', 'lstm'
        'decoder_n_layer': 2,
        'decoder_n_head': 4,
        'encoder_n_layer': 2,
        'encoder_n_head': 4,
        'unconditional_percentage': 0.,
        'guidance_scale': 1.,
        'sampling_method': 'top_p',  # can be 'top_p' or 'top_k'
        'num_molecules_generated': 20,
        'p': 1.,
        'k': 1,
        'mol_backbone': True,
        'similarity_threshold': 0.4,
        'num_samples': 10
    }
    print(hyper_params)
    sys.stdout.flush()
    train_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_train.npz')
    valid_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_valid.npz')
    test_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_test.npz')

    train_non_chiral_smiles, train_backbones, train_chains, train_assay_ids, train_types, train_labels = read_csv('./data/fsmol/train.csv')
    valid_non_chiral_smiles, valid_backbones, valid_chains, valid_assay_ids, valid_types, valid_labels = read_csv('./data/fsmol/valid.csv')
    test_non_chiral_smiles, test_backbones, test_chains, test_assay_ids, test_types, test_labels = read_csv('./data/fsmol/test.csv')

    smiles = train_non_chiral_smiles + valid_non_chiral_smiles + test_non_chiral_smiles
    tokenizer = create_smiles_tokenizer(smiles, max_num_chains=25)
    tokenizer.enable_padding(pad_token="<pad>", length=hyper_params['max_mol_len'])

    use_test = False

    train_ds = InteractionsDataset(train_non_chiral_smiles, train_backbones, train_chains, train_assay_ids,
                                   train_types, train_labels, train_prot_embeds, tokenizer,
                                   use_backbone=hyper_params['mol_backbone'])

    valid_ds = FSMolDataSet(valid_non_chiral_smiles, valid_backbones, valid_assay_ids, valid_types, valid_labels,
                            valid_prot_embeds, tokenizer, calc_rf=True, use_backbone=hyper_params['mol_backbone'])
    test_ds = FSMolDataSet(test_non_chiral_smiles, test_backbones, test_assay_ids, test_types, test_labels,
                           test_prot_embeds, tokenizer, calc_rf=True, use_backbone=hyper_params['mol_backbone'])

    eval_ds = test_ds if use_test else valid_ds

    train_dl = DataLoader(train_ds, batch_size=hyper_params['bs'], shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not hyper_params['mol_backbone']:
        mol_encoder = nn.Linear(2048, hyper_params['embedding_dim'])
        mol_decoder = TransformerDecoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                         hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['decoder_n_head'],
                                         n_layers=hyper_params['decoder_n_layer'], max_length=hyper_params['max_mol_len'])
    elif hyper_params['arch_type'] == 'transformer':
        mol_encoder = TransformerEncoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                         hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['encoder_n_head'],
                                         n_layers=hyper_params['encoder_n_layer'],
                                         max_length=hyper_params['max_mol_len'],
                                         pad_token=tokenizer.token_to_id('<pad>'))
        mol_decoder = TransformerDecoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                         hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['decoder_n_head'],
                                         n_layers=hyper_params['decoder_n_layer'], max_length=hyper_params['max_mol_len'])
    elif hyper_params['arch_type'] == 'gru' or hyper_params['arch_type'] == 'lstm':
        mol_encoder = RNNBasedEncoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                      hidden_size=hyper_params['embedding_dim'], n_layers=hyper_params['encoder_n_layer'],
                                      rnn_type=hyper_params['arch_type'])
        mol_decoder = RNNBasedDecoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                      hidden_size=2 * hyper_params['embedding_dim'], n_layers=hyper_params['encoder_n_layer'],
                                      rnn_type=hyper_params['arch_type'])
    else:
        mol_encoder, mol_decoder = None, None

    model = InteractionTranslator(prot_encoder=InteractionEncoder(2 + 1024, hyper_params['embedding_dim']),
                                  mol_encoder=mol_encoder, decoder=mol_decoder)

    print(f'model has: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')
    model.to(device)
    optimizer = Adam(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])
    recon_loss_fn = nn.CrossEntropyLoss()

    tot_epochs = hyper_params['epochs']
    pbar = tqdm([_ for _ in range(tot_epochs * len(train_dl))], position=0, leave=True)

    val_data = {'avg_sim': []}
    val_epochs = [5, 10, 20, 30, 50, 90]

    for epoch in range(tot_epochs):
        recon_losses = []
        model.train()
        for model_inputs, model_outputs, interactions, _ in train_dl:
            optimizer.zero_grad()
            rand_num = np.random.rand()
            inters = interactions.to(device)

            encoder_tokenized_in = model_inputs.to(device)

            decoder_tokenized_in, decoder_tokenized_tgt = model_outputs.to(device)[:, :-1], model_outputs[:, 1:].to(device)
            target_mask, target_padding_mask = create_target_masks(decoder_tokenized_in, device, tokenizer.token_to_id('<pad>'))

            if rand_num < hyper_params['unconditional_percentage']:
                prot_embed = torch.zeros((inters.shape[0], 1, hyper_params['embedding_dim'])).to(device)
                mol_embeds = model.mol_encoder(encoder_tokenized_in)
                if not hyper_params['mol_backbone']:
                    mol_embeds = torch.reshape(mol_embeds, (mol_embeds.shape[0], -1, mol_embeds.shape[1]))
                memory = torch.concat([prot_embed, mol_embeds], dim=1)
            else:
                mol_embeds = model.mol_encoder(encoder_tokenized_in)
                if not hyper_params['mol_backbone']:
                    mol_embeds = torch.reshape(mol_embeds, (mol_embeds.shape[0], -1, mol_embeds.shape[1]))
                    mol_embeds = mol_embeds.repeat((1, hyper_params['max_mol_len'], 1))

                if hyper_params['arch_type'] == 'transformer':
                    prot_embed = torch.unsqueeze(model.prot_encoder(inters), dim=1)
                    memory = torch.concat([prot_embed, mol_embeds], dim=1)
                elif hyper_params['arch_type'] == 'gru':
                    prot_embed = torch.unsqueeze(model.prot_encoder(inters), dim=0)
                    prot_embed = prot_embed.repeat((hyper_params['encoder_n_layer'], 1, 1))
                    memory = torch.concat([prot_embed, mol_embeds], dim=2)
                else:
                    prot_embed = torch.unsqueeze(model.prot_encoder(inters), dim=0)
                    prot_embed = prot_embed.repeat((hyper_params['encoder_n_layer'], 1, 1))
                    memory = (torch.concat([prot_embed, mol_embeds[0]], dim=2),
                              torch.concat([prot_embed, mol_embeds[1]], dim=2))

            logits = model.decoder(decoder_tokenized_in, memory, target_mask=target_mask,
                                   target_padding_mask=target_padding_mask)
            recon_loss = recon_loss_fn(logits.reshape(-1, logits.shape[-1]), decoder_tokenized_tgt.reshape(-1))
            recon_loss.backward()
            optimizer.step()
            recon_losses.append(recon_loss)
            pbar.set_description(f'recon_loss: {sum(recon_losses) / len(recon_losses)}, '
                                 f'epoch: {epoch + 1}')
            pbar.update(1)

        if (epoch + 1) in val_epochs:
            model.eval()
            with torch.no_grad():
                print(f'****************************** EPOCH {epoch + 1} ******************************')
                avg_sims = []
                for task in eval_ds.tasks:
                    if task.get('clf', None) is None:
                        continue
                    if task['assay_id'] not in val_data:
                        val_data[task['assay_id']] = []
                    # cur_interaction = torch.from_numpy(np.append(task['protein'], [1, task['assay_type']]))
                    # cur_interaction = torch.from_numpy(np.append(task['protein'], [1, task['assay_type']]))
                    np_int = np.append(np.array([1, task['assay_type']]), task['protein'])
                    cur_interaction = torch.from_numpy(np_int)

                    cur_interaction = cur_interaction.float().to(device)
                    # success_rate, validity, sims = [], [], []
                    prot_embed = torch.reshape(model.prot_encoder(cur_interaction), (1, 1, -1))
                    if hyper_params['arch_type'] != 'transformer':
                        prot_embed = prot_embed.repeat((hyper_params['encoder_n_layer'], 1, 1))
                    # prot_embed = model.prot_encoder(cur_interaction)

                    # zero_prot_embed = torch.zeros_like(prot_embed).to(device)
                    all_mol_embeds = []
                    idx = 0
                    while idx < len(task['inactive']):
                        end = min(idx + hyper_params['bs'], len(task['inactive']))
                        batch = task['inactive'][idx: end]
                        batch_size = end - idx
                        idx = end
                        # cur_scaffolds = torch.tensor(batch).long().to(device)
                        # cur_mols = torch.from_numpy(np.array(batch)).long().to(device)
                        cur_mols = torch.stack(batch).to(device)
                        cur_mols_embeds = model.mol_encoder(cur_mols)
                        if not hyper_params['mol_backbone']:
                            cur_mols_embeds = torch.reshape(cur_mols_embeds, (cur_mols_embeds.shape[0], -1, cur_mols_embeds.shape[1]))
                            cur_mols_embeds = cur_mols_embeds.repeat((1, hyper_params['max_mol_len'], 1))

                        if hyper_params['arch_type'] == 'transformer':
                            all_mol_embeds.extend([cur_mols_embeds[i, :, :].cpu() for i in range(batch_size)])
                        elif hyper_params['arch_type'] == 'gru':
                            all_mol_embeds.extend([cur_mols_embeds[:, i, :].cpu() for i in range(batch_size)])
                        else:
                            all_mol_embeds.extend([(cur_mols_embeds[0][:, i, :].cpu(),
                                                    cur_mols_embeds[1][:, i, :].cpu()) for i in range(batch_size)])
                    opt_molecules = {}
                    for orig_mol_embed, orig_mol, orig_backbone in zip(all_mol_embeds, task['inactive_smiles'], task['inactive_backbone']):

                        if hyper_params['arch_type'] == 'transformer':
                            orig_mol_embed = orig_mol_embed.to(device)
                            orig_mol_embed = torch.unsqueeze(orig_mol_embed, dim=0)
                            memory = torch.concat([prot_embed, orig_mol_embed], dim=1)
                        elif hyper_params['arch_type'] == 'gru':
                            orig_mol_embed = orig_mol_embed.to(device)
                            orig_mol_embed = torch.unsqueeze(orig_mol_embed, dim=1)
                            memory = torch.concat([prot_embed, orig_mol_embed], dim=2)
                        else:
                            orig_mol_embed_h, orig_mol_embed_c = orig_mol_embed[0].to(device), orig_mol_embed[1].to(device)
                            orig_mol_embed_h, orig_mol_embed_c = torch.unsqueeze(orig_mol_embed_h, dim=1), torch.unsqueeze(orig_mol_embed_c, dim=1)
                            # prot_embed = prot_embed.repeat((hyper_params['encoder_n_layer'], 1, 1))
                            memory = (torch.concat([prot_embed, orig_mol_embed_h], dim=2),
                                      torch.concat([prot_embed, orig_mol_embed_c], dim=2))



                        # memory = torch.concat([prot_embed, orig_mol_embed], dim=1)
                        # uncond_memory = torch.concat([zero_prot_embed, orig_mol_embed], dim=1)
                        # uncond_memory = torch.zeros_like(uncond_memory).to(device)
                        uncond_memory = None
                        generated_mols = generate_molecules(model.decoder, memory, uncond_memory, device, tokenizer,
                                                            hyper_params['max_mol_len'],
                                                            tokenizer.token_to_id('<bos>'),
                                                            tokenizer.token_to_id('<eos>'),
                                                            hyper_params['guidance_scale'],
                                                            hyper_params['num_molecules_generated'],
                                                            hyper_params['sampling_method'],
                                                            hyper_params['p'],
                                                            hyper_params['k'],
                                                            orig_backbone,
                                                            orig_mol,
                                                            mol_backbone=hyper_params['mol_backbone'])
                        opt_molecules[orig_mol] = generated_mols

                    validity, avg_diversity, std_diversity, avg_similarity, std_similarity, avg_success, std_success = \
                        evaluate_task(opt_molecules, task['clf'], threshold=task['threshold'],
                                      similarity_threshold=hyper_params['similarity_threshold'])

                    val_data[task['assay_id']].append(avg_success)
                    avg_sims.append(avg_similarity)

                    print('*'*5)
                    print(task['assay_id'])
                    print(f'thresh: {task["threshold"]}')
                    print(f'success rate: {avg_success}, {std_success}')
                    print(f'diversity: {avg_diversity}, {std_diversity}')
                    print(f'sim: {avg_similarity}, {std_similarity}')
                    print(f'validity: {validity}')
                    sys.stdout.flush()
                sys.stdout.flush()
                val_data['avg_sim'].append(sum(avg_sims) / len(avg_sims))

        if (epoch + 1) in val_epochs:
            output_dir = Path(f'./models/{cur_time}/epoch{epoch + 1}')
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(copy.deepcopy(model.state_dict()), f'{str(output_dir)}/model.pt')
            with open(f'{str(output_dir)}/hyper_params.json', 'w') as f:
                json.dump(hyper_params, f, indent=4)
            tokenizer.save(f'{str(output_dir)}/tokenizer_object.json')
