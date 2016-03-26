#!/bin/bash

# Executes ```TrainClickModels.py <click model>``` with different click models.

SCRIPTSDIR='metacentrum'

mkdir -p ${SCRIPTSDIR}

for model in 'CM' 'PBM' 'DCM' 'DBN' 'CCM' 'UBM'
do
cat << 'END' > ${SCRIPTSDIR}/TrainClickModels_${model}.sh
#!/bin/bash
module add python27-modules-gcc
export PYTHONUSERBASE=/storage/brno3-cerit/home/tunystom/anaconda
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=/storage/brno3-cerit/home/tunystom/.local/pypy-4.0.1-linux_x86_64-portable/site-packages:$PYTHONUSERBASE/lib/python2.7/site-packages:$PYTHONPATH
END

cat << END >> ${SCRIPTSDIR}/TrainClickModels_${model}.sh
cd /storage/brno3-cerit/home/tunystom/PyClick/scripts
/storage/brno3-cerit/home/tunystom/.local/pypy-4.0.1-linux_x86_64-portable/bin/pypy TrainClickModels.py ${model} 1> ${SCRIPTSDIR}/TrainClickModels_${model}.o 2> ${SCRIPTSDIR}/TrainClickModels_${model}.e
END

qsub -l walltime=2d -l nodes=1:ppn=1:^cl_zebra:^cl_zewura,mem=20gb ${SCRIPTSDIR}/TrainClickModels_${model}.sh
sleep 1
done
