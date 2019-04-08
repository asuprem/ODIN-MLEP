#!/bin/bash
# weight -- unweighted; performance
# updateSchedule -- 2592000000 (1 mo), 604800000 (1 week), 1210000000 (2 weeks)
# selectMethod -- historical, historical-new, historical-update, train, recent, recent-new, recent-update
# filterSelect -- nearest, top-k, no-filter
update=( "2592000000" "1210000000" )
weights=( "unweighted" "performance" )
select=( "train" "recent" "recent-new" "recent-updates" "historical-new" "historical-updates"  "historical" )
filter=( "no-filter" "top-k" "nearest" )
kval=( "5" )
for updatemethod in "${update[@]}"
do
    for weightsmethod in "${weights[@]}"
    do
        for selectmethod in "${select[@]}"
        do
            for filtermethod in "${filter[@]}"
            do
                for kvalmethod in "${kval[@]}"
                do
                    echo "python application.py expname --update ${updatemethod} --weights ${weightsmethod} --select ${selectmethod} --filter ${filtermethod} --kval ${kvalmethod} >> expLogs.log 2>&1" >> expLogs.log 2>&1
                    echo "python application.py expname --update ${updatemethod} --weights ${weightsmethod} --select ${selectmethod} --filter ${filtermethod} --kval ${kvalmethod} >> expLogs.log 2>&1" >> currentExp.log 2>&1
                    python application.py expname --update ${updatemethod} --weights ${weightsmethod} --select ${selectmethod} --filter ${filtermethod} --kval ${kvalmethod} >> expLogs.log 2>&1
                done
            done
        done
    done
done