# MLEP
MLEP is a Machine Learning Event Processor. It is a system to perform event detection in drifting
data.

[RODRIGO: Suggestion -- WHAT IS DRIFTING DATA?]
[RODRIGO: Suggestion -- WHAT IS DRIFTING ADAPTIVITY?]
[RODRIGO: Suggestion -- WHAT IS A PIPELINE IN THIS CONTEXT?]

Here we describe execution, implementation, and data exploration for MLEP.

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

### Downloading Support Files
MLEP is designed to work out-of-the-box in most scenarios. However, there are some built-in files
you may need to download:
* Bag of Words: Text file to generate the Bag-of-Words model. We have provided a default
version [here](https://drive.google.com/open?id=1xxnAGya_gYxgGuKk7FRZ1s07uACBXgeU). You will need to
move this file `bow.txt` into `./config/RawSources`. We have also provided a pre-built encoder file
[here](https://drive.google.com/open?id=1lKjFcgwtyMTEDCpAr-7Oc-zmfp-G7gWh) (not required, as the
encoder will generate it if missing). If you opt to download it, you will also need to move this
file `bow.model` file into `./config/Sources/`.
* word2vec (Google News): File containing pre-trained vectors built from Google News to be used
by a word2vec encoder. We have provided a default version
[here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). You will need to extract
the file with extension `.bin` and move in into `./config/Sources/`. Also, rename it to
`GoogleNews-vectors-negative300.bin`.
* word2vec (Wikipedia): [RODRIGO: Lack of understanding -- NOT CLEAR]
You can download a file containing a list of Wikipedia titles to be used by a word2vec encoder from
[here](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz). You will need
to extract the archive and move this file into `./config/RawSources/`. Also, rename it to
`enwiki-latest-all-titles-in-ns0`. To speed up the encoder generation, we have provided additional
files [here](https://drive.google.com/drive/folders/1BuLbH6f_rjtdx9RwipTsmXi6xmxNFYH9?usp=sharing).
These are `.wikipages` files with random pre-downloaded Wikipedia articles (random seed:
"wikipedia"). Each file has 5k, 10k, or 20k articles, as per its name. You will need to move all
`.wikipages` files into `./config/RawSources/`. To speed up encoder access, we have provided the
actual encoder model files
[here](https://drive.google.com/open?id=1SL21I_FYOoDYuq77i4TRY1WDJhlZgy36). Each encoder has three
associated files: a `.bin` file, a `.bin.trainables.syn1neg.npy` file, and a `.bin.wv.vectors.npy`
file. These files should be moved into `./config/Sources`.

Furthermore, it is worth noting that MLEP contains a built-in encoder for word2vec trained on the
Wikipedia corpus (`w2vGeneric` encoder class).

### Configutation
The default configuration file is located at `./config/configuration/MLEPServer.json`. Details are
described in the `Configuration Files` section. You may add new pipelines to this file to test drift
adaptivity [RODRIGO: Lack of understanding -- WHAT ARE PIPELINES? WHAT IS DRIFT? WHAT IS DRIFT
ADAPTIVITY? HOW?]

### Execution (script)

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
