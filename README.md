# Enhancing Project-Specific Code Completion by Inferring Intra-Project API Information

## Overview

In this paper, we follow the RAG paradigm but explore a practical approach to infer API information. Specifically, we propose a novel project-specific knowledge retrieval method that does not rely on import statements. Our method first expands the representation of intra-project API information, including usage examples and functional semantic information. Then, we use a code draft to guide the retrieval of API information required for code completion based on these two types of information. Additionally, we develop a project-specific code completion framework that not only considers similar code but also captures the intra-project API information on which the completion depends.

![famework](./doc/framework.jpg)

## Project Structure

This project contains the basic components of our approach. Here is an overview:

```shell
|-- build_j_func_base.py # build our api knowledge database (Java)
|-- build_py_ func_base.py # build our api knowledge database (Python)
|-- build_func_prompt.py # search and build relevant api inforamtion 
|-- run.py # run the code completion pipeline
|-- utils # utility functions
|-- evaluation # evaluation scripts
|-- datasets # the input data for the code completion task
|-- repos # the checkpoints of repositories used to build our benchmark. Please email us if you need the raw data.
|-- scripts # scripts for data processing
|-- prompts # prompts for code completion
|-- predictions # the output of code completion
|-- appendix.md # the online appendix of our paper
```

## Quick Start

### Install Requirements

```sh
$ pip install -r requirements.txt
```

### Run the Pipeline

**API Knowledge Instruction**

The `build_function_database` function in `run.py` shows the process if building the API knowledge information database.
```sh
$ sh scripts/build_database.sh
```

**Code Draft Generation**

Then we need to run the `build_code_draft_prompt` function in `run.py` get the code draft and the organize the results in the following format:

```json
{
    "prompt": "similar code snippets + unfinished code",
    "pred_res": "generated completion",
    "metadata": {...}
}
```

**Project Knowledge Retrieval & Target Code Generation**

After obtaining the code draft file, we run the `build_target_code_prompt` function in `run.py` to obtain the similar code snippets and the intra-project API information on which the completion depends.  It is worth noting that there is a parameter `mode` (full, -uer, -fsr) that can be changed to select specific components for our API Inference method.

```sh
$ sh scripts/infile_api.sh
```

**Evaluation**

Finally, we can evaluate the performance of the code completion with the `evaluation/eval.py` scripts