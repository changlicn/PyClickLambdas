#!/bin/bash

# List the algorithms for which you want to run the experiments.
ALGORITHMS=('CascadeUCB1' 'CascadeKL-UCB' 'RelativeCascadeUCB1' 'QuickRank' 'ShuffleAndSplit')

# Specify the output directory (WITHOUT TRAILING BACKSLASH) for the experiments.
OUTPUTDIR='experiments'

function shouldRun() { local i; for i in ${ALGORITHMS[@]}; do [[ "$i" == "$1" ]] && return 0; done; return 1; }


echo "Experiments started: `date`"
echo

# ============================================================================
# CascadeUCB1 algorithm
# ============================================================================
if shouldRun 'CascadeUCB1'
then
    echo "Running experiments for CascadeUCB1"
    echo "========================================"
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/CascadeUCB1Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} CascadeUCB1Algorithm -a 0.51 ${OUTPUTDIR}/CascadeUCB1Algorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# CascadeKL-UCB algorithm
# ============================================================================
if shouldRun 'CascadeKL-UCB'
then
    echo "Running experiments for CascadeKL-UCB"
    echo "========================================"
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} CascadeKLUCBAlgorithm ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# RelativeCascadeUCB1 algorithm
# ============================================================================
if shouldRun 'RelativeCascadeUCB1'
then
    echo "Running experiments for RelativeCascadeUCB1"
    echo "========================================"
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RelativeCascadeUCB1Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} RelativeCascadeUCB1Algorithm -a 0.51 ${OUTPUTDIR}/RelativeCascadeUCB1Algorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# QuickRank algorithm
# ============================================================================
if shouldRun 'QuickRank'
then
    echo "Running experiments for QuickRank"
    echo "========================================"
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/QuickRankAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} QuickRankAlgorithm ${OUTPUTDIR}/QuickRankAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# RankedUCB1Bandits algorithm
# ============================================================================
if shouldRun 'RankedUCB1Bandits'
then
    echo "Running experiments for RankedUCB1Bandits"
    echo "========================================"
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RankedUCB1BanditsAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} RankedUCB1BanditsAlgorithm -a 0.51 ${OUTPUTDIR}/RankedUCB1BanditsAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# ShuffleAndSplit algorithm
# ============================================================================
if shouldRun 'ShuffleAndSplit'
then
    echo "Running experiments for ShuffleAndSplit"
    echo "========================================"
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/ShuffleAndSplitAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n 10000000 -w 8 -c 5 -r -s ${i} ShuffleAndSplitAlgorithm ${OUTPUTDIR}/ShuffleAndSplitAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

echo "Experiments finished: `date`"
