#!/bin/bash
group_size=(1)
model='resnet50'
date='0905'
epochs=10
gpus=8
for g_s in ${group_size[@]}
do
    log_path=./logfile_"${date}"/"${model}"_"${g_s}_WU"
    timeline_filename=./timeline_gs/time_"${date}"_"${model}"_"${g_s}".json
    echo $timeline_filename

    horovodrun -np $gpus --timeline-filename $timeline_filename  python distributed_train.py  --model $model --epochs $epochs --log-dir $log_path --group-size $g_s 
done

