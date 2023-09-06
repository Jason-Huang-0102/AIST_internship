#!/bin/bash
group_size=(1 10 100 1000 )
model='resnet50'
date='0830'
epochs=10
gpus=8
for g_s in ${group_size[@]}
do
    log_path=./logfile_"${date}"/baseline_"${model}"_"${g_s}"
    timeline_filename=./timeline_gs/baseline_time_"${date}"_"${model}"_"${g_s}".json
    echo $timeline_filename

    horovodrun -np $gpus --timeline-filename $timeline_filename  python baseline_train.py  --model $model --epochs $epochs --log-dir $log_path --group-size $g_s 
done

