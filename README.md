# Projet-Data-Science
mon-projet/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ pyproject.toml
├─ configs/
│  ├─ default.yaml
│  └─ train.yaml
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  ├─ 02_features.ipynb
│  └─ 03_modeling.ipynb
├─ src/
│  └─ mon_projet/
│     ├─ __init__.py
│     ├─ data/
│     │  ├─ load_data.py
│     │  └─ preprocess.py
│     ├─ features/
│     │  └─ build_features.py
│     ├─ models/
│     │  ├─ train.py
│     │  ├─ predict.py
│     │  └─ evaluate.py
│     └─ utils/
│        └─ helpers.py
├─ tests/
│  ├─ test_preprocess.py
│  └─ test_train.py
├─ outputs/
│  ├─ figures/
│  ├─ models/
│  └─ reports/
└─ scripts/
   ├─ run_train.py
   └─ run_inference.py