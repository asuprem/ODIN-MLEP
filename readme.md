# MLEP
MLEP is a Machine Learning Event Processor. It is a system to perform event detection in drifting
data.

Here we describe execution, implementation, and data exploration for MLEP.

## Division of work
Basically, Abhijit wrote the MLEP server (front-end interface) and drift-adaptive section on top of
the learning models developed by Rodrigo (back-end interface under `config/`). Right now, we are
both working on improving code modularity, documentation, and testing.

## Current release

This readme is the most up-to-date set of instructions. However, the most up-to-date stable code release is 0.4. You can download it from the Releases tab or from [here](https://github.com/asuprem/MLEP/releases).

Regarding support files, for release 0.4, you will only need `w2v-wiki-wikipedia-20000.bin*` (and associated .npy files) in ./config/Sources/. For details, please read the **Downloading Support Files** section in the **word2vec (Wikipedia)** from [here](https://drive.google.com/open?id=1SL21I_FYOoDYuq77i4TRY1WDJhlZgy36). Please read through the **Downloading Support Files** section nevertheless.

## Quick start
### Creating a Python Virtual Environment
The main purpose of using a Python virtual environment is to create an isolated environment for
MLEP. This means that it can separately install its own dependencies, regardless of what package
versions are installed in your system.

Create a new Python2.7 virtual environment in directory `.env`:
```console
$ python -m venv .env
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
Base dependencies for MLEP are declared in the file `minrequirements.txt`. To get started, install these
dependencies running:
```console
(.env) $ pip install -r minrequirements.txt
```

You can also use  `requirements.txt` if you are feeling adventurous. To get started, install these
dependencies running:
```console
(.env) $ pip install -r requirements.txt
```

### Testing
Unit tests are (still) being written and merged to the repo.
* `utils_tests.py`

### Downloading Support Files
MLEP is designed to work out-of-the-box in most scenarios. However, there are some built-in files
you may need to download:
* **Bag of Words**: Text file to generate the Bag-of-Words model. We have provided a default
version [here](https://drive.google.com/open?id=1xxnAGya_gYxgGuKk7FRZ1s07uACBXgeU). You will need to
move this file `bow.txt` into `./config/RawSources`. We have also provided a pre-built encoder file
[here](https://drive.google.com/open?id=1lKjFcgwtyMTEDCpAr-7Oc-zmfp-G7gWh) (not required, as the
encoder will generate it if missing). If you opt to download it, you will also need to move this
file `bow.model` file into `./config/Sources/`.
* **word2vec (Google News)**: File containing pre-trained vectors built from Google News to be used
by a word2vec encoder. We have provided a default version
[here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). You will need to extract
the file with extension `.bin` and move in into `./config/Sources/`. Also, rename it to
`GoogleNews-vectors-negative300.bin`.
* **word2vec (Wikipedia)**: You must download a file containing a list of Wikipedia titles to be used by a word2vec encoder from
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

### Configuration
The default configuration file is located at `./config/configuration/MLEPServer.json`. Details are
described in the `Configuration Files` section. You may add new pipelines to this file to test drift
adaptivity.

### Execution (script)

Once you have all relevant files, you can run:

    $ (venv) sh scripts/ExperimentRunG.sh

If you want to run this in the background (because all outputs are piped to logfiles anyways), you should use `nohup`:

    $ (venv) nohup sh scripts/ExperimentRunG.sh &

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

To get options, run

    $ (venv) python application.py --help

Example of execution with options:

    $ (venv) python application.py M-U-TT-T-5 --update 2592000000 --weights unweighted --select train --filter top-k --kval 5

You can give any name to the experiment. However, following the naming conventions in the previous section (*Execution (scripts)*) helps to keep track of the results.

# The Configuration File
We describe the MLEP configuration file (located in `./config/configuration/MLEPServer.json`). You may also see a `./config/configuration/ExperimentalServer.json`. This is generated during code execution and can be ignored. We focus on `./config/configuration/MLEPServer.json` here.

The **MLEPServer.json** file has four components: 
- **config** -- This describes overal functionality of an MLEP execution
- **models** -- This contains the list of models MLEP has access to.
- **encoders** -- This contains the list of encoders MLEP has access to.
- **pipelines** -- This is the list of pipelines MLEP will use for drift adaptation.

Unlike most ML approaches, we do not consider a single classifier as an event detector. Instead, our model for an event detector is a Pipeline, similar to an sklearn or Spark pipeline. A Pipeline is a Directed Acyclic Graph of transformations from an input to an output. In our case, we use simple Pipelines with just an encoder and a classifier, as we have not built the infrastructure for more complex, tree-like DAGs.

## Config

The following variables are supported in "config":

- `update_schedule` Time in milliseconds before MLEP performs a scheduled update. The default value is 30 days (2592000000 ms)

- `weight_method` How selected models are weighted for each classification task. There are three options
    - `"unweighted"` -- Unweighted average.
    - `"performance"` -- Weights are determined based on performance. 
- `select_method` -- Which models are considered for classifications for each sample
    - `"train"` -- Only models trained in the very first data window are considered.
    - `"recent"` -- Only models generated or updated in the prior data window are considered
    - `"recent-new"` -- Only generated models in the prior data window are considered
    - `"recent-updates"` -- Only updated models in the prior data window are considered. In the first data window, `recent-new` models are considered as `recent-updates`  because there are no update models in the first window
    - `"historical"` -- All models ever created are considered
    - `"historical-new"` -- All models that were newly generated in all windows are considered
    - `"historical-updates"` -- Only updated models across all windows are considered

- `filter_select` -- How models under consideration are selected for classification
    - `"no-filter"` -- All models are used in an ensemble. <span style="color:red"> Warning: This is very slow. </span>
    - `"top-k"` -- Only the best *k* performing models are selected
    - `"nearest"` -- For each data sample, we find the *k* training models whose training data is closest to the data sample. 

- `k-val` -- k-value for `filter_select`
    - A positive integer. This is used in `filter_select`

## models

Each model is a key-value pair, the value being the model description object. Each model in `MLEPServer.json` needs a corresponding model class file stored in `./config/LearningModel/`. The model class file must inherit the provided `LearningModel.py` abstraction and implement the functions provided within.

    "MODELNAME": {
        "name": "MODELNAME",
        "desc": "Description",
        "scriptName": "MODELSCRIPT"
    }

- `"MODELNAME"` -- The name of the model. Built in models include default "sgd" and "logreg"
    - `"name"` -- "MODELNAME" again. You will need to keep this consistent. Look at built-in examples
    - `"desc"` -- Text description. There is no sanitizing or checking, so it'd be great if special characters or nonstandard encodings weren't used.
    - `"scriptName"` -- Name of python file that represents this model. You do not need the extension. The python file MUST be in the `./config/LearningModel/` folder.

## encoders
An encoder is defined similar to a model. There is a key difference however. Each model defined in `models` is unique, because models are instantiated within pipelines as and when they are created anytime during classification. Since each model is trained on different data, there is no global model that can cover all model types.

However, an encoder is created only once, as it would be a waste of memory. So each encoder defined in `encoders` is unique. Built-in definitions include three versions of `w2vGeneric`: `w2v-generic-5000`, `w2v-generic-10000`, and `w2v-generic-20000`.

A sample encoder definition is given below.

    "bowDefault":
        {
            "name": "bowDefault",
            "desc": "Default bow Encoder using bow.model",
            "scriptName": "bowEncoder",
            "args":{
                "modelFileName":"bow.model"
            },
            "fail-args":{
                "rawFileName": "bow.txt",
                "modelFileName": "bow.model"
            }
        }

A generic encoder definition is given below.

    "ENCODERNAME":
        {
            "name": "ENCODERNAME",
            "desc": "Description",
            "scriptName": "ENCODERSCRIPT",
            "args":{
                "ARG1":"PARAM1"
            },
            "fail-args":{
                "ARG1": "PARAM1",
                "ARG2": "PARAM2"
            }
        }

- `"ENCODERNAME"` -- Name of the encoder. This needs to be unique
    - `"name"` -- "ENCODERNAME" again. This needs to be the same
    - `"desc"` -- Encoder description
    - `"scriptName"` -- Name of python file that represents this encoder class. You do not need the extension. The python file MUST be in the `./config/DataEncoder/`
    - `"args"` -- Arguments for the `setup()` function of the Encoder. See the `./config/DataEncoder/DataEncoder.py` interface for details.
    - `"fail-args"` -- Arguments for the `failCondition()` function of the Encoder. See the `./config/DataEncoder/DataEncoder.py` interface for details.

## pipelines

A pipeline is used for classification. A generic pipeline is given below.

    "PIPELINENAME":{
            "name": "PIPELINENAME",
            "sequence": ["ENCODERNAME", "MODELNAME"],
            "type":"binary",
            "encoder": "ENCODERNAME",
            "valid":true/false
        }

- `"PIPELINENAME"` -- Name of the pipeline
    - `"name"` -- "PIPELINENAME" again
    - `"sequence"` -- Our current implementation is a simple implementation. So the sequence is not a full DAG specification. It's, at this time, a list of two elements - the ENCODER and MODEL this pipeline specifies
    - `"type"` -- Currently, only "binary" is supported. We plan to add support for "regression" and "multiclass".
    - `"valid"` -- Whether the current pipeline is valid or not. This allows you to keep old pipelines by invalidating them without deleting them from the configuration file.


