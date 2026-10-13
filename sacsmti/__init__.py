"""Stable augmenting chains (SAC) and SAC-ILS for MAX-SMTI, with the methods they are compared with.

sac          stable augmenting chains: chain search, SAC, SAC-ILS (and its variants)
tbls         TBLS and the TBLS-ILS control (sparse); Gale-Shapley start and TBLS disruption
exact        CP-SAT on the integer programming model of MAX-SMTI
ilp_delorme  preprocessing and models M1-M4 of Delorme et al. (2019), solved with CP-SAT
lns          exact neighbourhood re-optimisation (SAC-ILS-LNS, repair analysis)
gen, io      instance generators and the JSON format of the benchmark
stability    independent weak-stability checkers
baselines    published methods: CKY, HR, MCS, AS, LTIU
"""
