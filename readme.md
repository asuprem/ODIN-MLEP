# MLEP (In Progress)
MLEP is an ML-Based Event Processing system inspired by the system described in the ASSED paper (to
be referenced later), operating on continuous data (continuous learning). It generates filters,
updates filters, and classify data following the process outlined below:

1. Launch MLEP Classifier Process
    - Get Unlabeled, Labeled from HDI (Heterogeneous Data Integration)
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

## Quick start
### Creating a Python Virtual Environment
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

Later, you can deactivate it by running:
```console
(.env) $ deactivate
```

### Installing Dependencies
All dependencies for MLEP are declared in the file `requirements.txt`. To get started, install these
dependencies running:
```console
(.env) $ pip install -r requirements.txt
```

### Launch MLEP Application
An MLEP application runs as a process and maintains filters. Classifications are performed by
accesses (each classifier is an API endpoint).

### Classification
```python
# TODO: Change.
mlep = MLEP()
# COPY THIS: mlep.connect(port=[ARGUMENT])

mlep.getGenerateMethod()
mlep.getClasses()

while not mlep.isTrained():
    if not mlep.inTraining():
        mlep.initialTraining(data)
    else:
        sleep(10)
```

when receive new data:
        mlep.classify(data, k-Value, weightingMethod)
```
    

### Backend

    setUpEndPoints()

# Requirements

sqlite3
spatialite



    
