# Word and Constituent Boundaries in Hierarchical Multiscale Recurrent Neural Networks

### Corpus
* **Penn Treebank** partially available from [NLTK](http://www.nltk.org/nltk_data/)
```python
>>> import nltk
>>> nltk.download()
...
Identifier> treebank
```
* Generate groundtruth boundary labels from Penn Treebank under `treebank/`:
`python convert_boundary.py --path TARGET_PATH --threshold MIN_TOKENS`

### Next steps
1. F1 score of HM-RNN boundary detection:
    1. (*finished*) Convert parsing in PTB to 1s/0s boundary indicators, and use that as ground truth boundaries
    2. Run some distinctly tuned HM-LSTM models on PTB, and calculate F1 scores of HM-LSTM for some layer’s boundary indicators, (which is expected to be low…)
    3. Calculate BPC (LM evaluation metric) by these HM-LSTM on PTB
    4. Compare the correlation/trending of F1 and BPC

1. Statistically analyze with PCFG from PTB:
    1. Focus on improving syntactic meanings of HM-LSTM boundary indicators
    2. Compute PCFGs from PTB
    3. Find out if/what constituencies detected by HM-LSTM boundary coincide with PCFGs
    
1. QA on children book dataset
    1. Setup model
    2. ...