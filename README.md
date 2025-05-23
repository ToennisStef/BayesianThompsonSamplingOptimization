# Thompson Sampling pyproject

This project explores the use of Thompson Sampling in combination with Bayesian Optimization to solve optimization problems involving both continuous and categorical variables. Specifically, we aim to evaluate the effectiveness of this approach using Gaussian Process Regression models.

## Problem Setup

We consider an optimization problem that resembles a multi-armed bandit problem, where the objective is to maximize an unknown function $f$ that depends on both continuous variables $\mathbf{x}$ and categorical variables $\mathbf{z}$. In the following and throughout this project the categorical variable $\mathbf{z}$ will be called arm, based on multi-armed bandit optimization. The problem can be formulated as:

$$
\max_{\mathbf{x}, \mathbf{z}} f(\mathbf{x}, \mathbf{z})
$$

where:
- $\mathbf{x} \in \mathbb{R}^d$ represents the continuous variables
- $\mathbf{z} \in \{1, 2, \ldots, k\}$ represents the categorical variables

The goal is to find the optimal combination of $\mathbf{x^*}$ and $\mathbf{z^*}$ that maximizes the function $f$.

## Solution Approach

We solve the problem by iterative observation of the categrical variable/ arm $\hat{\mathbf{z}}$ and the corresponding new candidate for the continous variable $\hat{\mathbf{x}}$.

To solve the problem setup, we follow these steps:

1. **Space Filling Sampling**: Perform space filling sampling on the continuous variables $\mathbf{x}$ on each arm $\mathbf{z}$.
2. **Gaussian Process Construction**: Construct a Gaussian Process (GP) for each arm $\mathbf{z}$ based on the initial samples for each arm.
3. **Acquisition Function Creation**: Create the acquisition function for each GP.
4. **Maximization of Acquisition Function**: Obtain the argmax of each acquisition function.
5. **Thompson Sampling**: Sample from the GP at the argmax of the corresponding acquisition function. The sample argmax with respect to the arm is selected as the next sample.
6. **Next Training Point Selection**: The next training point is generated at $\mathbf{z}$ where the Thompson sample of the GP at $\mathbf{x}^*$ was highest, and $\mathbf{x}^*$ is selected by maximizing the acquisition function.

This iterative process continues until the optimal combination of $\mathbf{x}$ and $\mathbf{z}$ that maximizes the function $f$ is found.


# Branch Specific

## Optimization of Lactic Acid Extraction – Problem Statement & Approaches

### Goal:
- Optimize the extraction of lactic acid from a contaminated aqueous solution (the fermentation broth) by selecting an optimal solvent, extraction temperature and initial lactic acid concentration.

### Problem Statement

We try to optimize the extraction efficiency

### Initial Approach:


- Perform thermodynamic calculations using COSMOtherm.
- Apply Bayesian Optimization (BO) for optimization.


Model the problem as a Multi-Armed Bandit (MAB) with one continuous variable (temperature).
- Combine Bayesian Optimization with Thompson Sampling:
    1. Treat each solvent as a discrete variable, with a separate Gaussian Process (GP) for temperature for each solvent.
    2. Determine the best next temperature candidate for each solvent.
    3. Extract the Gaussian Random Variable (GRV).
    4. Select the next experiment using Thompson Sampling.

### Problem with the Approach:
- Solvent similarities are not considered.
- High initial measurement effort, as each solvent is treated separately.

### Possible Alternatives for Solvent Encoding:
1. **One-Hot Encoding** → Unsuitable, as it ignores chemical similarities.
2. **Molecular Descriptors / Fingerprints** (e.g., SMILES, ECFP, logP, HBD/HBA) to capture chemical properties.
3. **Latent Embeddings** similar to word embeddings in NLP, to better capture structural similarities.

### Open Question:
- Which encoding method is best suited for BO?
- Are there better strategies to integrate solvent similarities into the optimization process?

## Following plan

- encoding of solvents with ordinal-encoding either by:
    - arbitrary ordinal encoding
    - ordinal encoding of sorted solvent dataframe with respect to SMILES configuration



## Notes

dodcanone: 
- 2-dodecanone|CAS_number:6175-49-1
- 3-dodecanone|CAS_Number:1534-27-6
- 5-dodecanone|CAS_Number:19780-10-0
Nicht in COSMO Datenbank enthalten!

Methylisobutylketone:
- COSMO_name: 4-methy-2-pentanone | CAS_Number: 108-10-1