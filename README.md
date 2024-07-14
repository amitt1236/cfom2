# CFOM: Lead Optimization For Drug Discovery With Limited Data

> **Abstract:**
> Developing drugs is a resource-intensive endeavor, that may take many years to complete. The main objective of lead 
> optimization is to generate a novel molecule, that is chemically similar to the input molecule but with an enhanced 
> property. One of the desired properties for a chemical compound is to be active against a target protein associated with 
> the disease. Often, machine learning techniques are used in the process of discovering and improving potential drug 
> candidates. We introduce a new molecular representation, which takes inspiration from techniques employed by experts in 
> the field. This unique representation significantly improves the performance of conventional neural network architectures 
> during the lead optimization phase of the drug discovery process. Moreover, we incorporate various data modalities, 
> including information related to proteins from previous experiments, to boost the generalization capabilities of 
> models, especially in situations where data is scarce.

This repository provides a reference implementation of CFOM as described in the [paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614807).

## Requirements

Python 3.7 was used. You can find the libraries used in the requirements.txt file. To set up the environment use the command:
`pip install -r requirements.txt`

## Usage

### Training
You can adjust the training parameters in the train.py file. To start training run:
```python train.py```
A checkpoint of the trained model will be saved in the directory 'models'.

### Evaluating
For evaluating a model use the script evaluate.py which will use the model to generate the optimized molecule and then 
evaluate the outputs. To start evaluating run: ```python evaluate.py ./models/your_model```

## Cite

Please cite our paper if you find this work useful:
```
@inproceedings{kaminsky2023cfom,
  title={CFOM: Lead Optimization For Drug Discovery With Limited Data},
  author={Kaminsky, Natan and Singer, Uriel and Radinsky, Kira},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={1056--1066},
  year={2023}
}
```