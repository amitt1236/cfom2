import numpy as np
from pathlib import Path
import copy
import json
from datetime import datetime
from torch.utils.data import DataLoader
from datasets import InteractionsDataset, FSMolDataSet
from utils import read_csv, create_smiles_tokenizer, create_target_masks, load_tokenizer_from_file
from evaluate import generate_molecules
from models import InteractionEncoder, InteractionTranslator, TransformerEncoder, TransformerDecoder
from evaluate import evaluate_task
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import pickle

def training(model, tokenizer, hyper_params, loader, test_ds, epochs, device):

    print(f'model has: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')
    
    optimizer = Adam(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])
    recon_loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        recon_losses = []
        model.train()
        for model_inputs, model_outputs, interactions, _ in loader:
            optimizer.zero_grad()
            rand_num = np.random.rand()
            inters = interactions.to(device)

            encoder_tokenized_in = model_inputs.to(device)

            decoder_tokenized_in, decoder_tokenized_tgt = model_outputs.to(device)[:, :-1], model_outputs[:, 1:].to(device)
            target_mask, target_padding_mask = create_target_masks(decoder_tokenized_in, device, tokenizer.token_to_id('<pad>'))

            if rand_num < hyper_params['unconditional_percentage']:
                prot_embed = torch.zeros((inters.shape[0], 1, hyper_params['embedding_dim'])).to(device)
                mol_embeds = model.mol_encoder(encoder_tokenized_in)
                memory = torch.concat([prot_embed, mol_embeds], dim=1)
            else:
                mol_embeds = model.mol_encoder(encoder_tokenized_in)
                prot_embed = torch.unsqueeze(model.prot_encoder(inters), dim=1)
                memory = torch.concat([prot_embed, mol_embeds], dim=1)

            logits = model.decoder(decoder_tokenized_in, memory, target_mask=target_mask,
                                    target_padding_mask=target_padding_mask)
            
            recon_loss = recon_loss_fn(logits.reshape(-1, logits.shape[-1]), decoder_tokenized_tgt.reshape(-1))
            recon_loss.backward()
            optimizer.step()
            recon_losses.append(recon_loss)

    validation_step(model, tokenizer, hyper_params, test_ds, 5, device)

def validation_step(model, tokenizer, hyper_params, dataset, epoch, device):
    # save model for current validation
    cur_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = Path(f'./models/{cur_time}/epoch{epoch + 1}')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(copy.deepcopy(model.state_dict()), f'{str(output_dir)}/model.pt')
    with open(f'{str(output_dir)}/hyper_params.json', 'w') as f:
        json.dump(hyper_params, f, indent=4)
    tokenizer.save(f'{str(output_dir)}/tokenizer_object.json')
    
    print("*"  * 20 + "model saved" + "*" * 20)

    model.eval()
    with torch.no_grad():
        for task in tqdm(dataset):
            if task.get('clf', None) is None:
                continue            

            cur_interaction = torch.from_numpy(np.append(task['protein'], [1, task['assay_type']]))
            cur_interaction = cur_interaction.float().to(device)
            prot_embed = torch.reshape(model.prot_encoder(cur_interaction), (1, 1, -1))
            all_mol_embeds = []

            idx = 0
            while idx < len(task['inactive']):
                end = min(idx + hyper_params['bs'], len(task['inactive']))
                batch = task['inactive'][idx: end]
                batch_size = end - idx
                idx = end

                cur_mols = torch.stack(batch).to(device)
                cur_mols_embeds = model.mol_encoder(cur_mols)
                all_mol_embeds.extend([cur_mols_embeds[i, :, :].cpu() for i in range(batch_size)])
                
            opt_molecules = {}
            for orig_mol_embed, orig_mol, orig_backbone in zip(all_mol_embeds, task['inactive_smiles'], task['inactive_backbone']):

                orig_mol_embed = orig_mol_embed.to(device)
                orig_mol_embed = torch.unsqueeze(orig_mol_embed, dim=0)
                memory = torch.concat([prot_embed, orig_mol_embed], dim=1)

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

            print('*'*10)
            print(task['assay_id'])
            print(f'thresh: {task["threshold"]}')
            print(f'success rate: {avg_success}, {std_success}')
            print(f'diversity: {avg_diversity}, {std_diversity}')
            print(f'sim: {avg_similarity}, {std_similarity}')
            print(f'validity: {validity}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hyper_params = {
        'bs': 256,
        'lr': 0.0001,
        'weight_decay': 0.,
        'epochs': 100,
        'max_mol_len': 128,
        'embedding_dim': 512,
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
        'k': 40,
        'mol_backbone': True,
        'similarity_threshold': 0.4,
        'num_samples': 10
    }
    print(hyper_params)

    # train_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_train.npz')
    with open('prot_emb_train.pkl', 'rb') as file:
        train_prot_embeds = pickle.load(file)
    
    # valid_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_valid.npz')
    with open('prot_emb.pkl', 'rb') as file:
        test_prot_embeds = pickle.load(file)

    # test_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_test.npz')

    train_non_chiral_smiles, train_backbones, train_chains, train_assay_ids, train_types, train_labels = read_csv('./data/fsmol/train.csv')
    valid_non_chiral_smiles, valid_backbones, valid_chains, valid_assay_ids, valid_types, valid_labels = read_csv('./data/fsmol/valid.csv')
    test_non_chiral_smiles, test_backbones, test_chains, test_assay_ids, test_types, test_labels = read_csv('./data/fsmol/test.csv')

    smiles = train_non_chiral_smiles + valid_non_chiral_smiles + test_non_chiral_smiles
    tokenizer = create_smiles_tokenizer(smiles, max_num_chains=25)
    tokenizer.enable_padding(pad_token="<pad>", length=hyper_params['max_mol_len'])
    
    train_ds = InteractionsDataset(train_non_chiral_smiles, train_backbones, train_chains, train_assay_ids,
                                train_types, train_labels, train_prot_embeds, tokenizer)
    
    # valid_ds = FSMolDataSet(valid_non_chiral_smiles, valid_backbones, valid_assay_ids, valid_types, valid_labels,
    #                         valid_prot_embeds, tokenizer, calc_rf=True, use_backbone=hyper_params['mol_backbone'])
    test_ds = FSMolDataSet(test_non_chiral_smiles, test_backbones, test_assay_ids, test_types, test_labels,
                            test_prot_embeds, tokenizer, calc_rf=True, use_backbone=hyper_params['mol_backbone'])

    model = InteractionTranslator(prot_encoder=InteractionEncoder(2 + 1280, hyper_params['embedding_dim']),
                                mol_encoder=TransformerEncoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                                                hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['encoder_n_head'],
                                                                n_layers=hyper_params['encoder_n_layer'],
                                                                max_length=hyper_params['max_mol_len'],
                                                                pad_token=tokenizer.token_to_id('<pad>')),
                                decoder=TransformerDecoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                                            hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['decoder_n_head'],
                                                            n_layers=hyper_params['decoder_n_layer'],
                                                            max_length=hyper_params['max_mol_len']))
    
    model_path = "./models/my_embd"
    # model.load_state_dict(torch.load(f'{model_path}/model.pt', map_location=device))
    # tokenizer = load_tokenizer_from_file(f'{model_path}/tokenizer_object.json')

    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=hyper_params['bs'], shuffle=True)
    training(model, tokenizer, hyper_params, train_loader, test_ds, 20,device)

    # validation_step(model, tokenizer, hyper_params, test_ds, 5, device)

if __name__ == "__main__":
    main()
