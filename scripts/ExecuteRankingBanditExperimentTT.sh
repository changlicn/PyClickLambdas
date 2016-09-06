#!/bin/bash

echo "Started at `date`"

# ============================================================================
# RankingBanditsGangAlgorithm algorithm
# ============================================================================
#python2.7 ./RankingBanditExperiment.py -q all -m UBM -n $1 -w 1 -c 5 --regret RankingBanditsGangAlgorithm -m KL-UCB experiments/RankingBanditsGangAlgorithm_Horizon$1
#date

# ============================================================================
# StackedRankingBanditsAlgorithm algorithm
# ============================================================================
# mkdir -p experiments/StackedRankingBanditsAlgorithm
# python2.7 ./RankingBanditExperiment.py -q all -m PBM -n 1000000 -w 8 -c 5 -r StackedRankingBanditsAlgorithm -m KL-UCB -f experiments/StackedRankingBanditsAlgorithm


# ============================================================================
# CascadeUCB1 algorithm
# ============================================================================
# for i in `seq 10`
# do
#     echo "Run #${i}..."
#     mkdir -p experiments/CascadeUCB1Algorithm/run${i}
#     python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} CascadeUCB1Algorithm -a 0.51 experiments/CascadeUCB1Algorithm/run${i}
# done

# ============================================================================
# CascadeKL-UCB algorithm
# ============================================================================
# for i in `seq 10`
# do
#     echo "Run #${i}..."
#     mkdir -p experiments/CascadeKLUCBAlgorithm${i}
#     python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm${i}
# done

# ============================================================================
# RelativeCascadeUCB1 algorithm
# ============================================================================
# for i in `seq 10`
# do
#     echo "Run #${i}..."
#     mkdir -p experiments/RelativeCascadeUCB1Algorithm/run${i}
#     python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} RelativeCascadeUCB1Algorithm -a 0.51 experiments/RelativeCascadeUCB1Algorithm/run${i}
# done

# ============================================================================
# RelativeRanking algorithm
# ============================================================================
# mkdir -p experiments/RelativeRankingAlgorithm_Q59560_PBM
# python2.7 ./RankingBanditExperiment.py -q 59560 -m PBM -n 10000 -w 8 -c 5 -r RelativeRankingAlgorithm -m uniform -e 1000 -f RefinedSkipClickLambdasAlgorithm experiments/RelativeRankingAlgorithm_Q59560_PBM

# mkdir -p experiments/RelativeRankingAlgorithm
# python2.7 ./RankingBanditExperiment.py -q all -m PBM -n 50000000 -w 8 -c 5 -r RelativeRankingAlgorithm -m uniform -e 500000 -f RefinedSkipClickLambdasAlgorithm experiments/RelativeRankingAlgorithm
# python2.7 ./RankingBanditExperiment.py -q all -m PBM -n 1000000 -w 8 -c 5 -r RelativeRankingAlgorithm -m uniform -e 100000 -f RefinedSkipClickLambdasAlgorithm experiments/RelativeRankingAlgorithm

# ============================================================================
# CoarseRelativeRanking algorithm
# ============================================================================
# python2.7 ./RankingBanditExperiment.py -q all -m all -n $1 -w 8 -c 5 --regret CoarseRelativeRankingAlgorithm -a 0.05 -f SkipClickLambdasAlgorithm experiments/CoarseRelativeRankingAlgorithm_Horizon$1

# echo

# ============================================================================
# RelativeRanking algorithm
# ============================================================================

#./RankingBanditExperiment.py -q all -m CM -n 100 -w 8 -c 5 RelativeRankingAlgorithm -a 0.5 -g 0.0 -f RefinedSkipClickLambdasAlgorithm experiments/RelativeRankingAlgorithm/CM@5

# ============================================================================
# CascadeUCB1 algorithm
# ============================================================================

# ./RankingBanditExperiment.py -q all -m CM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/CM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/CM@5 experiments/CascadeUCB1Algorithm/CM@5

# ./RankingBanditExperiment.py -q all -m DCM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/DCM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/DCM@5 experiments/CascadeUCB1Algorithm/DCM@5

# ./RankingBanditExperiment.py -q all -m DBN -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/DBN@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/DBN@5 experiments/CascadeUCB1Algorithm/DBN@5

# ./RankingBanditExperiment.py -q all -m CCM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/CCM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/CCM@5 experiments/CascadeUCB1Algorithm/CCM@5

# ./RankingBanditExperiment.py -q all -m PBM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/PBM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/PBM@5 experiments/CascadeUCB1Algorithm/PBM@5

# ./RankingBanditExperiment.py -q all -m UBM -n 10000000 -w 8 -c 5 CascadeUCB1Algorithm -a 1.5 experiments/CascadeUCB1Algorithm/UBM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeUCB1Algorithm/UBM@5 experiments/CascadeUCB1Algorithm/UBM@5

# ============================================================================
# CascadeKL-UCB algorithm
# ============================================================================

# ./RankingBanditExperiment.py -q all -m CM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/CM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/CM@5 experiments/CascadeKLUCBAlgorithm/CM@5

# ./RankingBanditExperiment.py -q all -m DCM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/DCM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/DCM@5 experiments/CascadeKLUCBAlgorithm/DCM@5

# ./RankingBanditExperiment.py -q all -m DBN -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/DBN@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/DBN@5 experiments/CascadeKLUCBAlgorithm/DBN@5

# ./RankingBanditExperiment.py -q all -m CCM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/CCM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/CCM@5 experiments/CascadeKLUCBAlgorithm/CCM@5

# ./RankingBanditExperiment.py -q all -m PBM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/PBM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/PBM@5 experiments/CascadeKLUCBAlgorithm/PBM@5

# ./RankingBanditExperiment.py -q all -m UBM -n 10000000 -w 8 -c 5 CascadeKLUCBAlgorithm experiments/CascadeKLUCBAlgorithm/UBM@5
# ./RankingRegretEvaluation.py -m all -c 5 -w 8 experiments/CascadeKLUCBAlgorithm/UBM@5 experiments/CascadeKLUCBAlgorithm/UBM@5

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

echo "Finished at `date`"
