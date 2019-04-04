# MLEP ( In progress)

# Execution
Use requirements.txt to create virtual environment

    $ (venv) python application.py

# Details

MLEP is an ML-Based Event Processing System. It is based on the system described in the ASSED paper (to be linked later).

MLEP performs filter generation, filter updates, and classification. General MLEP outline is provided below:

1. Launch MLEP Classifier Process
    - Get Unlabeled, Labeled from HDI 
    - getTopKFilters(Unlabeled/Labeled)
    - classifyUnlabeled(Unlabaled, [topK], "weighting")
        - Unweighted, weighted, model-weighted
        - driftDetectUpdate([topK])
2. Launch ScheduledGeneratorUpdate Process
    - scheduledFilterGenerate()
        - requestData(time)
        - createFilter()
    - scheduledFilterUpdate()
        - requestData(time)
        - data.forEach(getFilters(_topK) -> _topK.forEach(updateWithData(data)))
3. Launch DriftGenerateUpdate Process
    - FindFiltersThatHaveTooMuchDrift()
    - ForEachFilter -->
        - GetDataSinceLastUpdateThatWasMappedToThatFilter()
        - CopyFilter()
        - UpdateFilterCopy()

## Requirements

Here we focus on MLEP's abstract requirements. Technical requirements are provided in the next section.

MLEP operates on continuous data, and therefore performs **continuous learning**.

## Steps

1. First launch MLEP application. It runs as a process and maintains filters. Classifications are performed by accesses.

### Classification

    mlep = MLEP()
    mlep.connect(port=[])

    mlep.getGenerateMethod()
    mlep.getClasses()
    
    while not mlep.isTrained():
        if not mlep.inTraining():
            mlep.initialTraining(data)
        else:
            sleep(10)


    when receive new data:
        mlep.classify(data, k-Value, weightingMethod)
    

### Backend

    setUpEndPoints()

# Requirements

sqlite3
spatialite



    