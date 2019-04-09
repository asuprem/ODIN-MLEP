#!/bin/bash
# weight -- unweighted; performance
# updateSchedule -- 2592000000 (1 mo), 604800000 (1 week), 1210000000 (2 weeks)
# selectMethod -- historical, historical-new, historical-update, train, recent, recent-new, recent-update
# filterSelect -- nearest, top-k, no-filter
experimentOutputLogfile="./logfiles/experimentOutputLogfile.log"
experimentCurrentLogfile="./logfiles/experimentNames.log"
update=( "2592000000,M" "1210000000,F" )
weights=( "unweighted,U" "performance,P" )
select=( "train,TT" "recent,RR" "recent-new,RN" "recent-updates,RU" "historical-new,HN" "historical-updates,HU"  "historical,HH" )
filter=( "no-filter,F" "top-k,T" "nearest,N" )
kval=( "5,5" )
for updatemethod in "${update[@]}"
do
    IFS=',' read updateVal updateName <<< "${updatemethod}"
    for weightsmethod in "${weights[@]}"
    do
        IFS=',' read weightVal weightName <<< "${weightsmethod}"
        for selectmethod in "${select[@]}"
        do
            IFS=',' read selectVal selectName <<< "${selectmethod}"
            for filtermethod in "${filter[@]}"
            do
                IFS=',' read filterVal filterName <<< "${filtermethod}"
                for kvalmethod in "${kval[@]}"
                do
                    IFS=',' read kvalVal kvalName <<< "${kvalmethod}"
                    expName="${updateName}-${weightName}-${selectName}-${filterName}-${kvalName}"
                    echo "python application.py ${expname} --update ${updateVal} --weights ${weightVal} --select ${selectVal} --filter ${filterVal} --kval ${kvalVal} >> expLogs.log 2>&1" >> $experimentOutputLogfile 2>&1
                    echo "python application.py ${expname} --update ${updateVal} --weights ${weightVal} --select ${selectVal} --filter ${filterVal} --kval ${kvalVal} >> ${experimentOutputLogfile} 2>&1" >> $experimentCurrentLogfile 2>&1
                    #python application.py expname --update ${updatemethod} --weights ${weightsmethod} --select ${selectmethod} --filter ${filtermethod} --kval ${kvalmethod} >> expLogs.log 2>&1
                done
            done
        done
    done
done