# MLEP

MLEP is a Machine Learning Event Processor. It is a system to perform event detection in drifting data. We describe execution, implementation, and data exploration for MLEP.

## Requirements
The `requirements.txt` file includes all requirements for running MLEP with python-pip. Use pip install the requirements in an appropriate virtual environment (`venv`).

## Downloads
MLEP is designed to work out-of-the-box in most scenarios. However, there are some built-in files you will need to download, outlined below

### Bag of Words
You will need a text file to generate the Bag-of-Words model. We have provided a default file [here](https://drive.google.com/open?id=1xxnAGya_gYxgGuKk7FRZ1s07uACBXgeU). You will need to move this file (bow.txt) to `./config/RawSources`. We have also provided a Bag of Words encoder file [here](https://drive.google.com/open?id=1lKjFcgwtyMTEDCpAr-7Oc-zmfp-G7gWh). You do not need this (bow.model), as the Encoder will generate this if it is missing. You will need to move the bow.model file into `./config/Sources/`

### word2vec (Google News)
If you are planning to use the word2vec encoder using pretrained Google News vectors, you will need to download the zipped file [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). The downloaded file will be a `.bin.gz` file. You will need to extarct the `.bin` file and move it into `./config/Sources/`. The file should be named `GoogleNews-vectors-negative300.bin`.

### word2vec (Wikipedia)
MLEP contains built-in Encoders for word2vec trained on the wikipedia corpus (the `w2vGeneric` encoder class). You will need the Wikipedia titles list linked [here](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz). This is a `.gz` file. You will need to extract the titles file and move it into `./config/RawSources/`. The file should be named `enwiki-latest-all-titles-in-ns0`.

To speed up Encoder Generation, we have provided additional files [here](https://drive.google.com/drive/folders/1BuLbH6f_rjtdx9RwipTsmXi6xmxNFYH9?usp=sharing). These are `.wikipages` files with random pre-downloaded wikipedia articles (Random seed: "wikipedia"). Each file has 5K, 10K, or 20K articles, as per its name. You will need to move all `.wikipages` files into `./config/RawSources/`

To speed up Encoder access, we have also provided the actual Encoder model files [here](https://drive.google.com/open?id=1SL21I_FYOoDYuq77i4TRY1WDJhlZgy36). These are the same files you would get if you ran the Encoder without these files present, as the `w2vGeneric` Encoder trains the model files if they are not present. Each Encoder has three associated files: a `.bin` file, a `.bin.trainables.syn1neg.npy` file, and a `.bin.wv.vectors.npy` file. You will need all three for each Encoder. These files should be moved into `./config/Sources`

## Configutation
The default configuration file is located at `./config/configuration/MLEPServer.json`. Details are described in the `Configuration Files` section. You may add new pipelines to this file to test drift adaptivity. 

## Execution (script)

Once you have all relevant files, you can run:

    $ (venv) sh scripts/ExperimentRunG.sh

You may need to run `dos2unix` on the .sh file first to convert Windows line endings to UNIX line endings. Experiment outputs will be saved in `dataCollect.csv` in the following format:

    EXPERIMENT_NAME, P1, P2, P3, ..., PN

where EXPERIMENT_NAME is the name of the experiment (described below), P1 is performance on the first 100 samples, P2 is performance on the next 100 samples, and so on.

An `Experiment_Name` in `ExperimentRunG.sh` is defined as follows:

    expName="${updateName}-${weightName}-${selectName}-${filterName}-${kvalName}"

Each variable is described below:

 - **updateName** - Name of update mode. 
    - "M" : monthly updates 
    - "F" : fortnight updates (every two weeks)
 - **weightName** - Name of weight method. 
    - "U" : unweighted average 
    - "P" : performance weighted
 - **selectName** - Name of model selection scope.
    - "TT"  : Only select from initially trained models
    = "RR"  : Only select from models generated or updated in the prior update period
    - "RN"  : Only select from new models generated in prior update period
    - "RU"  : Only select from update models generated in prior update period
    - "HN"  : Only select from all newly-built models across all update periods
    - "HU"  : Only select from all update-type models across all update periods
    - "HH"  : Select from all models generated
 - **filterName** - Name of model selection filtering method. 
    - "F" : No filter. Select from all models. <span style="color:red">**WARNING:** extremely slow</span>
    - "T" : Top-k. Select the top-k highest performing models from **selectName**
    - "N" : Select the k-Nearest Neighbors to current data point from **selectName**
- **kval** - k-value for filterName
    - [INT] - value for **filterName**

# Execution (program)

Run:

    $ (venv) python application.py EXPERIMENT_NAME

    $ (venv) python application.py --help

# MLEP ( In progress) <span style="color:red">**out of date**</span>

# Execution
Use requirements.txt to create virtual environment

    $ (venv) python application.py

# Details ()

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



    