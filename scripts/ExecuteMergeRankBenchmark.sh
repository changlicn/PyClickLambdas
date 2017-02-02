#!/bin/bash

# List the algorithms for which you want to run the experiments.
ALGORITHMS=('CascadeKL-UCB-lc' 'RankedBanditsExp3' 'RankedBanditsKL-UCB' 'RealMergeRankKL' 'PIEAlgorithm')
# ALGORITHMS=('PIEAlgorithm')

# Specify the input file with query/click models.
INPUT='data/60Q/model_query_collection.pkl'

# Specify the output directory (WITHOUT TRAILING BACKSLASH) for the experiments.
OUTPUTDIR='experiments/mergerank_benchmark'

# The upper time (number of impressions) limit.
NIMPRESSIONS=10000
#10000000

# The number of individual ranking experiment repetitions.
NRUNS=5
#10

# The number of CPUs running the experiments in parallel
# NOTE: Works only for experiment with a single ranking algorithm!!!.
NCPUS=20

# The cut-off position.
CUTOFF=5

# Click model
CM="PBM CM"

function shouldRun() { local i; for i in ${ALGORITHMS[@]}; do [[ "$i" == "$1" ]] && return 0; done; return 1; }


echo "Experiments started: `date`"
echo

# ============================================================================
# CascadeKL-UCB-ff algorithm
# ============================================================================
if shouldRun 'CascadeKL-UCB-ff'
then
    echo "Running experiments for CascadeKL-UCB-ff"
    echo "========================================"
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} CascadeKLUCBAlgorithm -f 'ff' ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# CascadeKL-UCB-lc algorithm
# ============================================================================
if shouldRun 'CascadeKL-UCB-lc'
then
    echo "Running experiments for CascadeKL-UCB-lc"
    echo "========================================"
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} CascadeKLUCBAlgorithm -f 'lc' ${OUTPUTDIR}/CascadeKLUCBAlgorithm/run${i}
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
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RankedBanditsUCB1Algorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} RankedBanditsUCB1Algorithm -a 0.51 -f 'ff' ${OUTPUTDIR}/RankedBanditsUCB1Algorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================="
    echo
fi

# ============================================================================
# RankedBanditsKLUCB algorithm
# ============================================================================
if shouldRun 'RankedBanditsKL-UCB'
then
    echo "Running experiments for RankedBanditsKL-UCB"
    echo "==========================================="
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RankedBanditsKLUCBAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} RankedBanditsKLUCBAlgorithm -f 'ff' ${OUTPUTDIR}/RankedBanditsKLUCBAlgorithm/run${i}
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
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} RankedBanditsExp3Algorithm -f 'ff' ${OUTPUTDIR}/RankedBanditsExp3Algorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================="
    echo
fi

# ============================================================================
# RealMergeRankKL algorithm
# ============================================================================
if shouldRun 'RealMergeRankKL'
then
    echo "Running experiments for RealMergeRankKL"
    echo "========================================"
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/RealMergeRankKLAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} RealMergeRankAlgorithm -m 'kl' -c -f 'ff' ${OUTPUTDIR}/RealMergeRankKLAlgorithm/run${i}
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
    echo "========================================"
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/MergeRankKLAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} MergeRankAlgorithm -m 'kl' -f 'ff' ${OUTPUTDIR}/MergeRankKLAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

# ============================================================================
# MergeRankZeroKL algorithm
# ============================================================================
if shouldRun 'MergeRankZeroKL'
then
    echo "Running experiments for MergeRankZeroKL"
    echo "========================================"
    for i in `seq 1 ${NRUNS}`
    do
        echo "Run #${i} started: `date`"
        mkdir -p ${OUTPUTDIR}/MergeRankZeroKLAlgorithm/run${i}
        python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} MergeRankAlgorithm -m 'kl' -c -f 'ff' ${OUTPUTDIR}/MergeRankZeroKLAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "========================================"
    echo
fi

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
      python2.7 ./RankingBanditExperiment.py -i ${INPUT} -q 104183 11527 128292 46254 218954 89951 -m ${CM} -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} PIEAlgorithm -l 0 -f 'fc' ${OUTPUTDIR}/PIEAlgorithm/run${i}
    done
    echo "Done: `date`"
    echo "===================================="
    echo
fi

echo "Experiments finished: `date`"


# Code for running 
# python2.7 ./RankingBanditExperiment.py -i data/PBM+CM_model_query_collection.pkl -q 104183 11527 128292 46254 218954 89951 -m all -n ${NIMPRESSIONS} -w ${NCPUS} -c ${CUTOFF} -r -s ${i} PIEAlgorithm -l 0 -f 'fc' ${OUTPUTDIR}/PIEAlgorithm/run${i}
