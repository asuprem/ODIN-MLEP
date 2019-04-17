#!/bin/bash
# weight -- unweighted; performance
# updateSchedule -- 2592000000 (1 mo), 604800000 (1 week), 1210000000 (2 weeks)
# selectMethod -- historical, historical-new, historical-update, train, recent, recent-new, recent-update
# filterSelect -- nearest, top-k, no-filter
update=( "M" "F" )
weights=( "U" "P" )
select=( "TT" "RR" "RN" "RU" "HN" "HU"  "HH" )
filter=( "F" "T" "N" )
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
                    #echo "python application.py expname --update ${updatemethod} --weights ${weightsmethod} --select ${selectmethod} --filter ${filtermethod} --kval ${kvalmethod} >> expLogs.log 2>&1" >> expLogs.log 2>&1
                    #echo "python application.py expname --update ${updatemethod} --weights ${weightsmethod} --select ${selectmethod} --filter ${filtermethod} --kval ${kvalmethod} >> expLogs.log 2>&1" >> currentExp.log 2>&1
                    #python application.py expname --update ${updatemethod} --weights ${weightsmethod} --select ${selectmethod} --filter ${filtermethod} --kval ${kvalmethod} >> expLogs.log 2>&1
                    echo ${updatemethod}-${weightsmethod}-${selectmethod}-${filtermethod}-${kvalmethod} >> expNames.log
                done
            done
        done
    done
done