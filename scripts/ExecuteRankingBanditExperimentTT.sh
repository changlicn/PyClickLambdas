#!/bin/bash

# List the algorithms for which you want to run the experiments.
ALGORITHMS=('CascadeUCB1' 'CascadeKL-UCB' 'RelativeCascadeUCB1' 'RankedBanditsUCB1' 'RankedBanditsExp3' 'QuickRank' 'MergeRank' 'ShuffleAndSplit')

# Specify the output directory (WITHOUT TRAILING BACKSLASH) for the experiments.
OUTPUTDIR='experiments'
NIMPRESSIONS=10000000

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
        python2.7 ./RankingBanditExperiment.py -q all -m all -n ${NIMPRESSIONS} -w 8 -c 5 -r -s ${i} CascadeUCB1Algorithm -a 0.51 ${OUTPUTDIR}/CascadeUCB1Algorithm/run${i}
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
        python2.7 ./RankingBanditExperiment.py -q all -m all -n ${NIMPRESSIONS} -w 8 -c 5 -r -s ${i} CascadeKLUCBAlgorithm ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
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
        python2.7 ./RankingBanditExperiment.py -q all -m all -n ${NIMPRESSIONS} -w 8 -c 5 -r -s ${i} RelativeCascadeUCB1Algorithm -a 0.51 ${OUTPUTDIR}/RelativeCascadeUCB1Algorithm/run${i}
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
        python2.7 ./RankingBanditExperiment.py -q all -m all -n ${NIMPRESSIONS} -w 8 -c 5 -r -s ${i} QuickRankAlgorithm ${OUTPUTDIR}/QuickRankAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# MergeRank algorithm
# ============================================================================
if shouldRun 'MergeRank'
then
    echo "Running experiments for MergeRank"
    echo "========================================"
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/MergeRankAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n ${NIMPRESSIONS} -w 8 -c 5 -r -s ${i} MergeRankAlgorithm ${OUTPUTDIR}/MergeRankAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# RankedBanditsUCB1 algorithm
# ============================================================================
if shouldRun 'RankedBanditsUCB1'
then
    echo "Running experiments for RankedBanditsUCB1"
    echo "========================================="
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RankedBanditsUCB1Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n ${NIMPRESSIONS} -w 8 -c 5 -r -s ${i} RankedBanditsUCB1Algorithm -a 0.51 ${OUTPUTDIR}/RankedBanditsUCB1Algorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================="
    echo
fi

# ============================================================================
# RankedBanditsExp3 algorithm
# ============================================================================
if shouldRun 'RankedBanditsExp3'
then
    echo "Running experiments for RankedBanditsExp3"
    echo "========================================="
    for i in `seq 1`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RankedBanditsExp3Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -q all -m all -n ${NIMPRESSIONS} -w 8 -c 5 -r -s ${i} RankedBanditsExp3Algorithm ${OUTPUTDIR}/RankedBanditsExp3Algorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================="
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
        python2.7 ./RankingBanditExperiment.py -q all -m all -n ${NIMPRESSIONS} -w 8 -c 5 -r -s ${i} ShuffleAndSplitAlgorithm ${OUTPUTDIR}/ShuffleAndSplitAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

echo "Experiments finished: `date`"
