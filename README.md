# CompeteSMoE - Effective Training of Sparse Mixture of Experts via Competition
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper [CompeteSMoE - Effective Training of Sparse Mixture of Experts via Competition]()</br>
Our implementation is based on the [Sandwich Transformer](https://github.com/ofirpress/sandwich_transformer). More training scripts and datasets are coming soon. 

## Prerequisites
- [FastMoE](https://github.com/laekov/fastmoe): A fast MoE impl for PyTorch

## Running Experiments in the Paper

#### Pre-training
- Download the enwik8, text8, wikitext-2 dataset from [here](https://github.com/laekov/fastmoe/blob/master/examples/transformer-xl/scripts/getdata.sh), then change bash scripts based on your local data paths`</br>
```bash
data_folder/
└── pretraining
    └── enwik8
        ├── test.txt
        ├── train.txt
        └── valid.txt
    └── text8
        ├── test.txt
        ├── train.txt
        └── valid.txt
    └── wikitext-2
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

- Select the Transformer architecture, its scale, and the type of SMoE layer. We support:

|                     | SMoE | SMoE-Dropout | XMoE | StableMoE | CompeteSMoE |
|---------------------|------|--------------|------|-----------|-------------|
| Transformer (S/M/L) |  ✅  |     ✅       |  ✅  |     ✅    |             |
| GLAM (S/M/L)        |  ✅  |     ✅       |  ✅  |     ✅    |             |

- Run all corresponding scripts: </br>
`bash enwik8_exp.sh`
`bash text8_exp.sh`
`bash wikitext2_exp.sh`

- The checkpoint will be saved at `checkpoints/enwik8/transformers-s` during training. 

## Citation