# RelaxRD

This repository contains the reference implementation for the paper  
"Storage-Centric Relation Design via High-Quality Approximate Functional Dependencies".

RelaxRD studies how to design relational schemas guided by approximate functional dependencies (AFDs) in order to reduce storage redundancy while guaranteeing lossless reconstruction.

---

## Repository Structure


RelaxRD/
├── AFD_discovery/          # Step 1: Approximate FD discovery
│   ├── afd_pkl/            # Discovered AFDs (serialized)
│   │   └── fraudtrain.pkl
│   └── tane.py             # TANE-based AFD discovery
│
├── CoreAFD/                # Step 2: Core AFD set selection
│   ├── source/
│   │   ├── naivecore.py
│   │   ├── quickcore.py
│   │   ├── selectingcoreafd.py
│   │   ├── selectingcoreafd_update.py
│   │   └── main.py
│   └── output/
│       └── fraud.pkl       # Selected core AFD set
│
├── RelaxRD/                # Step 3: RelaxRD schema design & materialization
│   ├── source/
│   │   ├── schemadesign.py
│   │   ├── RelaxRD.py
│   │   └── main.py
│   └── output/
│       ├── r0.csv
│       ├── r1.csv
│       ├── r2.csv
│       ├── r3.csv
│       └── r4.csv
│
└── dataset/                # Input datasets
└── fraud/
└── fraudTrain.csv


## Dataset

We include a **representative dataset (`fraud`)** in this repository as a **running example** for all three steps of RelaxRD:

- AFD discovery
- Core AFD set selection
- RelaxRD schema construction and materialization

All scripts are pre-configured to run on this dataset by default.  
Users may replace the dataset with their own data by modifying the dataset paths accordingly.

For convenience and reproducibility, **the outputs produced on the fraud dataset are already included** in each step’s `output/` directory.

---

## Step 1: Approximate Functional Dependency Discovery

Approximate functional dependencies can be:
- provided by domain experts, or
- automatically discovered using existing algorithms.

In this repository, we use a **TANE-based approach** to discover AFDs.

### Run

```bash
cd AFD_discovery
python tane.py

	•	Input datasets are read from the dataset/ directory.
	•	Discovered AFDs are serialized and stored in:


```bash
python train.py --data_dir "your data dir" --data enron --prefix bandrank --verbose 1