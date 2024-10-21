# Leesplank_Noot

This repository was created to develop the LLM Noot

It contains scripts for preprocessing the training data. There are 2 trainging datasets:

simplifictions:
    contains:
        - prompt: originional text
        - result: simplified output
    The processing script removes list like prompt, sorts the data on levenstein distance and saves the data on Huggingface
    https://huggingface.co/datasets/UWV/Leesplank_NL_wikipedia_simplifications_preprocessed 
veringewikkeldingen
    contains:
        - origional text
        - simplification
        - specific complex versions.
    The processing script reformats the dataset so that the simplification is the result and the complex versions are the prompt. 
    https://huggingface.co/datasets/UWV/veringewikkelderingen_preprocessed

In the server folder you find a markdown file containing some usefull commands to get the training started on the server and the documents needed on the server for training

https://www.youtube.com/watch?v=azLCUayJJoQ&t=591s