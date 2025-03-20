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
