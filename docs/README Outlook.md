# Extending to Multi-Fidelity Optimization

The lactic acid extraction optimization can be extended to a multi-fidelity framework. This approach allows the integration of high-fidelity COSMOtherm calculation data alongside lower-fidelity real-life measurements and experimental data. By leveraging these multiple sources of information, the optimization process can achieve improved accuracy and efficiency, balancing computational cost with predictive performance.

## Semi-Automatic Implementation Process

The implementation of this multi-fidelity optimization is likely a semi-automatic process. Initially, Bayesian optimization is executed using the current simulation and experimental data. Based on the results, the algorithm suggests a new experimental data point. This suggested data point must then be manually evaluated. Once the experimental data is generated, it needs to be manually added to the training dataset. After updating the training data, the Bayesian optimization process can be rerun with the newly augmented dataset.


# Better Encoding

Another way to extend the current Optimization Algorithm is to use a different encoding method. The currently employed method is very basic—a good first approach—but it can be improved. Instead of encoding the solvents via a SMILES ordering, the solvents can be encoded using chemical fingerprints. 

## Challenges in Fingerprint-Based Encoding

A major challenge in this approach is determining what constitutes a good fingerprint. The encoding must be:
- Clearly attributable, ensuring that the representation accurately reflects the molecular structure and that the numerical representation can be decoded back into a valid molecule.
- Capable of capturing the relevant chemical properties of the molecule to enhance the optimization process.

By addressing these challenges, fingerprint-based encoding could provide a more robust and informative representation of the solvents, potentially improving the optimization outcomes.

## References

- [Capecchi et al. (2020) - One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00445-4)