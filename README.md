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

Optimierung der Extraktionseffizienz von Milchsäure – Problemstellung & Ansätze

## Ziel:
- Optimierung der Extraktion von Milchsäure aus einer kontaminierten wässrigen Lösung durch Auswahl eines optimalen Lösungsmittels und einer optimalen Extraktionstemperatur.
- Thermodynamische Berechnungen erfolgen mit COSMOtherm.
- Anwendung von Bayesian Optimization (BO) zur Optimierung.
## Ursprünglicher Ansatz:
- Modellierung als Multi-Armed Bandit (MAB) mit einer kontinuierlichen Variablen (Temperatur).
- Kombination aus Bayesian Optimization und Thompson Sampling:
    1. Jedes Lösungsmittel als diskrete Variable, für jedes Lösungsmittel ein separater GP über Temperatur.
    2. Bestimmung des besten nächsten Temperaturkandidaten pro Lösungsmittel.
    3. Extraktion der Gaußschen Verteilung (GRV).
    4. Auswahl des nächsten Experiments per Thompson Sampling.
## Problem des Ansatzes:
- Keine Berücksichtigung von Lösungsmittel-Ähnlichkeiten.
- Hoher initialer Messaufwand, da jedes Lösungsmittel separat betrachtet wird.
## Mögliche Alternativen für die Lösungsmittel-Kodierung:
1. One-Hot-Encoding → ungeeignet, da chemische Ähnlichkeiten ignoriert werden.
2. Molekulare Deskriptoren / Fingerprints (z. B. SMILES, ECFP, logP, HBD/HBA) zur Erfassung chemischer Eigenschaften.
3. Latente Einbettungen ähnlich zu Word-Embeddings in NLP, um strukturelle Ähnlichkeiten besser zu erfassen.
## Offene Frage:
- Welche Kodierungsmethode eignet sich am besten für BO?
- Gibt es bessere Strategien zur Integration von Lösungsmittel-Ähnlichkeiten in die Optimierung?


## Vorläufiger plan
- Nur betrachtung von c6-c11:

| Kettenlänge | Alkan   | Alkohol   | Keton    |
|-------------|---------|-----------|----------|
| C6          | Hexan   | Hexanol   | Hexanon  |
| C7          | Heptan  | Heptanol  | Heptanon |
| C8          | Octan   | Octanol   | Octanon  |
| C9          | Nonan   | Nonanol   | Nonanon  |
| C10         | Decan   | Decanol   | Decanon  |
| C11         | Undecan | Undecanol | Undecanon|

Ordnung/Enkodierung nach Kettenlänge und Endgruppe:

| Kettenlänge | Alkan   | Alkohol   | Keton    |
|-------------|---------|-----------|----------|
| C6          | 1,1     | 1,2       | 1,3      |
| C7          | 2,1     | 2,2       | 2,3      |
| C8          | 3,1     | 3,2       | 3,3      |
| C9          | 4,1     | 4,2       | 4,3      |
| C10         | 5,1     | 5,2       | 5,3      |
| C11         | 6,1     | 6,2       | 6,3      |

- Andere gruppen ver