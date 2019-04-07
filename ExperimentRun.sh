#!/bin/bash
# weight -- unweighted; performance
# updateSchedule -- 2592000000 (1 mo), 604800000 (1 week), 1210000000 (2 weeks)
# selectMethod -- historical, historical-new, historical-update, train, recent, recent-new, recent-update
# filterSelect -- nearest, top-k, no-filter
update=( "2592000000" "1210000000" )
weights=( "unweighted" "performance" )
select=( "train" "historical" "historical-new" "historical-updates" "recent" "recent-new" "recent-updates" )
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
                    python application.py expname --update ${updatemethod} --weights ${weightsmethod} --select ${selectmethod} --filter ${filtermethod} --kval ${kvalmethod} >> expLogs.log 2&>1
                done
            done
        done
    done
done