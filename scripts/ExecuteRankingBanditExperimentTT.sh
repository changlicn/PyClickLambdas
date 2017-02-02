#!/bin/bash

# List the algorithms for which you want to run the experiments.
#ALGORITHMS=('CascadeUCB1' 'CascadeKL-UCB' 'RelativeCascadeUCB1' 'RankedBanditsUCB1' 'RankedBanditsExp3' 'QuickRank' 'MergeRank' 'MergeRankKL' 'ShuffleAndSplit')
ALGORITHMS=('PIEAlgorithm')

# Specify the input file with query/click models.
INPUT='data/60Q/model_query_collection.pkl'

# Specify the output directory (WITHOUT TRAILING BACKSLASH) for the experiments.
OUTPUTDIR='experiments/60Q/yandex'

# The upper time (number of impressions) limit.
NIMPRESSIONS=10000000

# The number of individual ranking experiment repetitions.
NRUNS=10

# The number of CPUs running the experiments in parallel
# NOTE: Works only for experiment with a single ranking algorithm!!!.
NCPUS=8

# The cut-off position.
CUTOFF=5

function shouldRun() { local i; for i in ${ALGORITHMS[@]}; do [[ "$i" == "$1" ]] && return 0; done; return 1; }


echo "Experiments started: `date`"
echo

# ============================================================================
# CascadeUCB1 algorithm
# ============================================================================
if shouldRun 'CascadeUCB1'
then
    echo "Running experiments for CascadeUCB1"
    echo "==================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/CascadeUCB1Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} CascadeUCB1Algorithm -a 0.51 -f 'ff' ${OUTPUTDIR}/CascadeUCB1Algorithm/run${i}
    done
    echo "Done: `date`"
    echo "==================================="
    echo
fi

# ============================================================================
# CascadeKL-UCB algorithm
# ============================================================================
if shouldRun 'CascadeKL-UCB'
then
    echo "Running experiments for CascadeKL-UCB"
    echo "====================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m PBM -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} CascadeKLUCBAlgorithm -f 'ff' ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "====================================="
    echo
fi

# ============================================================================
# RelativeCascadeUCB1 algorithm
# ============================================================================
if shouldRun 'RelativeCascadeUCB1'
then
    echo "Running experiments for RelativeCascadeUCB1"
    echo "==========================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RelativeCascadeUCB1Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} RelativeCascadeUCB1Algorithm -a 0.51 -f 'ff' ${OUTPUTDIR}/RelativeCascadeUCB1Algorithm/run${i}
    done
    echo "Done: `date`"
    echo "==========================================="
    echo
fi

# ============================================================================
# QuickRank algorithm
# ============================================================================
if shouldRun 'QuickRank'
then
    echo "Running experiments for QuickRank"
    echo "================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/QuickRankAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} QuickRankAlgorithm -f 'ff' ${OUTPUTDIR}/QuickRankAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "================================="
    echo
fi

# ============================================================================
# MergeRank algorithm
# ============================================================================
if shouldRun 'MergeRank'
then
    echo "Running experiments for MergeRank"
    echo "========================================"
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/MergeRankAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} MergeRankAlgorithm -f 'ff' ${OUTPUTDIR}/MergeRankAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# MergeRankKL algorithm
# ============================================================================
if shouldRun 'MergeRankKL'
then
    echo "Running experiments for MergeRankKL"
    echo "==================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/MergeRankKLAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} MergeRankAlgorithm -m 'kl' -f 'ff' ${OUTPUTDIR}/MergeRankKLAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "==================================="
    echo
fi


# ============================================================================
# RealMergeRankKL algorithm
# ============================================================================
if shouldRun 'RealMergeRankKL'
then
    echo "Running experiments for RealMergeRankKL"
    echo "======================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RealMergeRankKLAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} RealMergeRankAlgorithm -m 'kl' -c -f 'ff' ${OUTPUTDIR}/RealMergeRankKLAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "======================================="
    echo
fi


# ============================================================================
# MergeRankZeroKL algorithm
# ============================================================================
if shouldRun 'MergeRankZeroKL'
then
    echo "Running experiments for MergeRankZeroKL"
    echo "======================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/MergeRankZeroKLAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} MergeRankAlgorithm -m 'kl' -c -f 'ff' ${OUTPUTDIR}/MergeRankZeroKLAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "======================================="
    echo
fi

# ============================================================================
# RankedBanditsUCB1 algorithm
# ============================================================================
if shouldRun 'RankedBanditsUCB1'
then
    echo "Running experiments for RankedBanditsUCB1"
    echo "========================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RankedBanditsUCB1Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} RankedBanditsUCB1Algorithm -a 0.51 -f 'ff' ${OUTPUTDIR}/RankedBanditsUCB1Algorithm/run${i}
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
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RankedBanditsExp3Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} RankedBanditsExp3Algorithm -f 'ff' ${OUTPUTDIR}/RankedBanditsExp3Algorithm/run${i}
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
    echo "======================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/ShuffleAndSplitAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} ShuffleAndSplitAlgorithm -f 'ff' ${OUTPUTDIR}/ShuffleAndSplitAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "======================================="
    echo
fi

echo "Experiments finished: `date`"

# ============================================================================
# PIE(l) algorithm
# ============================================================================
if shouldRun 'PIEAlgorithm'
then
    echo "Running experiments for PIEAlgorithm"
    echo "===================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/PIEAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q all -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} PIEAlgorithm -l 0 -f 'fc' ${OUTPUTDIR}/PIEAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "===================================="
    echo
fi

