#!/bin/bash

# ============================================================================
# CascadeUCB1 algorithm
# ============================================================================

./RankingBanditExperiment.py -q all -m CM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/CM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/CM@5 experiments/CascadeUCB1Algorithm/CM@5

./RankingBanditExperiment.py -q all -m DCM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/DCM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/DCM@5 experiments/CascadeUCB1Algorithm/DCM@5

./RankingBanditExperiment.py -q all -m DBN -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/DBN@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/DBN@5 experiments/CascadeUCB1Algorithm/DBN@5

./RankingBanditExperiment.py -q all -m CCM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/CCM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/CCM@5 experiments/CascadeUCB1Algorithm/CCM@5

./RankingBanditExperiment.py -q all -m PBM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/PBM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/PBM@5 experiments/CascadeUCB1Algorithm/PBM@5

./RankingBanditExperiment.py -q all -m UBM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/UBM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/UBM@5 experiments/CascadeUCB1Algorithm/UBM@5

# ============================================================================
# CascadeKL-UCB algorithm
# ============================================================================

./RankingBanditExperiment.py -q all -m CM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/CM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/CM@5 experiments/CascadeKLUCBAlgorithm/CM@5

./RankingBanditExperiment.py -q all -m DCM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/DCM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/DCM@5 experiments/CascadeKLUCBAlgorithm/DCM@5

./RankingBanditExperiment.py -q all -m DBN -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/DBN@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/DBN@5 experiments/CascadeKLUCBAlgorithm/DBN@5

./RankingBanditExperiment.py -q all -m CCM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/CCM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/CCM@5 experiments/CascadeKLUCBAlgorithm/CCM@5

./RankingBanditExperiment.py -q all -m PBM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/PBM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/PBM@5 experiments/CascadeKLUCBAlgorithm/PBM@5

./RankingBanditExperiment.py -q all -m UBM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/UBM@5
./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/UBM@5 experiments/CascadeKLUCBAlgorithm/UBM@5

# Run an experiment with CascadeUCB1 on query 1153 with clicks generated from
# UBM model for 100 iterations:
# ./RankingBanditExperiment.py -q 1153 -m UBM -n 100 CascadeUCB1Algorithm --alpha 1.0 experiment_results

# Do the same with a Thompson like algorithm:
# ./RankingBanditExperiment.py -q 1153 -m UBM -n 100 CascadeThompsonSamplerAlgorithm --alpha 1.0 --beta 1.0 experiment_results

# You can choose from 10 different queries and 6 different click models
# or you can experiment with an algorithm on all combinations of these
# by running the following:
# ./RankingBanditExperiment.py -q all -m all -n 100 CascadeUCB1Algorithm --alpha 1.0 experiment_results

# To run an experimental algorithm, such as CopelandRakingAlgorithm, that
# uses a specific click-based lambdas algorithm, e.g. SkipClickLambdasAlgorithm,
# the command line would read:
# ./RankingBanditExperiment.py -q 1153 -m UBM -n 100 CopelandRakingAlgorithm --feedback SkipClickLambdasAlgorithm --alpha 1.0 experiment_results
