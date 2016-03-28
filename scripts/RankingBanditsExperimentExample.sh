#!/bin/bash

# Run an experiment with CascadeUCB1 on query 1153 with clicks generated from
# UBM model for 100 iterations:
./RankingBanditsExperiment.py -q 1153 -m UBM -n 100 CascadeUCB1Algorithm --alpha 1.0 experiment_results

# Do the same with a Thompson like algorithm:
./RankingBanditsExperiment.py -q 1153 -m UBM -n 100 CascadeThompsonSamplerAlgorithm --alpha 1.0 --beta 1.0 experiment_results

# You can choose from 10 different queries and 6 different click models
# or you can experiment with an algorithm on all combinations of these
# by running the following:
./RankingBanditsExperiment.py -q all -m all -n 100 CascadeUCB1Algorithm --alpha 1.0 experiment_results

# To run an experimental algorithm, such as CopelandRakingAlgorithm, that
# uses a specific click-based lambdas algorithm, e.g. SkipClickLambdasAlgorithm,
# the command line would read:
./RankingBanditsExperiment.py -q 1153 -m UBM -n 100 CopelandRakingAlgorithm --feedback SkipClickLambdasAlgorithm --alpha 1.0 experiment_results