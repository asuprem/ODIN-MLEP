# MLEP (In Progress)

# Quick start
## Creating a Python Virtual Environment
The main purpose of using a Python virtual environment is to create an isolated environment for
MLEP. This means that it can separately install its own dependencies, regardless of what package
versions are installed in your system.

Create a new Python virtual environment in directory `.env`:
```console
$ python3 -m venv .env
```

Now we need to activate the newly created virtual environment, setting up your shell to use it by
default:
```console
$ source .env/bin/activate
```

Later, you can deactivate it running:
```console
(.env) $ deactivate
```

## Installing Dependencies in the Python Virtual Environment
All dependencies for MLEP are declared in the file `requirements.txt`. To get started, install these
dependencies running:
```console
(.env) $ pip install -r requirements.txt
```

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



    
