# Intersort

Repository for the paper "Deriving Causal Order from Single-Variable Interventions: Guarantees &amp; Algorithm"

## Introduction

Targeted and uniform interventions to a system are crucial for unveiling causal relationships. While several methods have been developed to leverage interventional data for causal structure learning, their practical application in real-world scenarios often remains challenging. Recent benchmark studies have highlighted these difficulties, even when large numbers of single-variable intervention samples are available. In this work, we demonstrate, both theoretically and empirically, that such datasets contain a wealth of causal information that can be effectively extracted under realistic assumptions about the data distribution. More specifically, we introduce the notion of interventional faithfulness, which relies on comparisons between the marginal distributions of each variable across observational and interventional settings, and we introduce a score on causal orders. Under this assumption, we are able to prove strong theoretical guarantees on the optimum of our score that also hold for large-scale settings. To empirically verify our theory, we introduce Intersort, an algorithm designed to infer the causal order from datasets containing large numbers of single-variable interventions by approximately optimizing our score. Intersort outperforms baselines (GIES, PC and EASE) on almost all simulated data settings replicating common benchmarks in the field. Our proposed novel approach to modeling interventional datasets thus offers a promising avenue for advancing causal inference, highlighting significant potential for further enhancements under realistic assumptions.

## Citation

Please consider citing, if you reference or use our methodology, code or results in your work:

TBD

## Authors

Mathieu Chevalley, GSK plc and ETH Zurich<br/>
Arash Mehrjou, GSK plc<br/>
Patrick Schwab, GSK plc<br/>

## Acknowledgements

MC, AM and PS are employees and shareholders of GSK plc.
