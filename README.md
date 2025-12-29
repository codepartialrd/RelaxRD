# RelaxRD

This repository contains the reference implementation for the paper  
"Storage-Centric Relation Design via High-Quality Approximate Functional Dependencies".

RelaxRD studies how to design relational schemas guided by approximate functional dependencies (AFDs) in order to reduce storage redundancy while guaranteeing lossless reconstruction.



## Dataset

We include a **representative dataset (`fraud`)** (\href{https://www.kaggle.com/datasets/kartik2112/fraud-detection}) in this repository as a **running example** for all three steps of RelaxRD:

- AFD discovery
- Core AFD set selection
- RelaxRD Schema Construction and Relation Instantiation


All scripts are pre-configured to run on this dataset by default.  
Users may replace the dataset with their own data by modifying the dataset paths accordingly.

For convenience and reproducibility, **the outputs produced on the fraud dataset are already included** in each step’s `output/` directory.

---

## Step 1: AFD Discovery

In this step, we obtain a pool of candidate approximate functional dependencies (AFDs).
AFDs can be provided by domain experts or discovered automatically.
In our implementation, we use the classical TANE-based algorithm to discover AFDs.


From the project root directory, run:

```bash
python -m AFD_discovery.main_tane \
  --data_dir dataset/fraud/fraudTrain.csv \
  --output_dir AFD_discovery/Sigma \
  --error 0.05
```

	--data_dir: path to the input CSV dataset
	--output_dir: directory to store discovered AFDs
	--error: error threshold ε for approximate FDs


The discovered AFDs are stored as a pickle file in **AFD_discovery/Sigma/fraudTrain_sigma.pkl**.


## Step 2: Core AFD Set Selection


Given a pool of candidate AFDs $\Sigma$, We provide two algorithms for selecting the core AFD set: **NaiveCore** and **QuickCore**.

From the project root directory, run:

```bash
python -m CoreAFD.main_coreafd \
  --data_dir dataset/fraud/fraudTrain.csv \
  --Sigma AFD_discovery/Sigma/fraudtrain.pkl \
  --method quickcore
```

	--data_dir: path to the input dataset.
	--Sigma: path to the discovered AFD file produced in Step 1.
	--method: Core AFD selection algorithm: quickcore or naivecore.

The resulting core AFD set will be stored in **/CoreAFD/output/fraudTrain_coreafd.pkl**


## Step 3: RelaxRD Schema Construction and Relation Instantiation

Given an input dataset and a selected core AFD set, RelaxRD constructs a relaxed logical schema
and materializes the relation accordingly.


From the project root directory, run:


```bash
python -m RelaxRD.main_relaxrd \
  --data_dir dataset/fraud/fraudTrain.csv \
  --coreafd CoreAFD/output/fraudTrain_coreafd.pkl \
  --output_dir RelaxRD/output
```

	--data_dir: path to the input dataset
	--coreafd: path to the selected core AFD set
	--output_dir: root directory for storing decomposed relations

The decomposed relations will be stored in **RelaxRD/output/fraudTrain/**



