# OOD-failures
Reproduction of Nagarajan et al. 2021 ICLR [paper](https://arxiv.org/abs/2010.15775).

## Setup
Installation requirements:
- `python > 3`

Create a virtual environment:
```bash
virtualenv -p python3 venv_ood
```

Activate environment:
```bash
source venv_ood/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Install jupyter notebook kernel:
```bash
ipython kernel install --name "local-venv" --user
```

After starting the jupyter notebook server, select the `local-venv` kernel to run the experiments.

