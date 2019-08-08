This is the code for the MSc dissertaion project of University of Edinburgh: Generating Wikipedia: Neural Abstractive Multi-document Summarisation with Graph Convolutional Networks.


In this repository we include a link to our WikiCatSum dataset and code for our ConvS2D model. Our code extends the code of [Generating Summaries with Topic Guidance and Structured Convolutional Decoders]() by Laura Perez-Beltrachini, Yang Liu and Mirella Lapata, as well as an earlier copy of [Facebook AI Research Sequence-to-Sequence Toolkit](https://github.com/pytorch/fairseq) with a sentence aware Structured Convolutional Decoder.

## Dependencies

Python 3.6.6
Torch 0.4.0

## WikiCatSum dataset

The WikiCatSum dataset is available in [this](https://datashare.is.ed.ac.uk/handle/10283/3353) repository.

Related scripts are available in the *wikicatsum/* directory.


## Training a New Model

### Build Graphs
Using the files in the downloaded datasets you can extract semantic relationships with the following command. You will need to define the variables as convenient.

```TEXT``` should be the directory where to find the source and target texts

For example, extract semantic relationships on the training set:
```
python preprocessing.py --data-dir TEXT --save-dir TEXT --split train
```

Extracted (s,p,o) triples and coreference resolution results will be in the following format:
```
{'subject': 'John Szpunar', 'subjectSpan': [149, 151], 'relation': 'has', 'relationSpan': [151, 152], 'object': 'book', 'objectSpan': [152, 153], 'paragraphIndex': 4}
{'representativeMention': 'this movie', 'representativeMentionSpan': [257, 259], 'mentioned': 'it', 'mentionedSpan': [249, 250]}
```

### Pre-process

Using the files in the downloaded datasets you can generate data and dictionaries with the following command. You will need to define the variables as convenient.  

```TEXT``` should be the directory where to find the source and target texts 

```SRC_L``` is the length at which you will truncate the input sequence of paragraphs  

Use argument ```--singleSeq``` to create source and target as a single long sequence:
```
cd fairseq
python my_preprocess.py --source-lang src --target-lang tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/$DSTDIR \
  --nwordstgt 50000 --nwordssrc 50000 \
  --singleSeq --L $SRC_L \
  1> data-bin/$DSTDIR/preprocess.log
```


### Train
After you preprocessed the files you can run the training procedures.

##### ConvS2S
```
cd fairseq
CUDA_VISIBLE_DEVICES=$GPUID python train.py data-bin/$DATADIR --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_wikicatsum --save-dir checkpoints/$MODELNAME  --skip-invalid-size-inputs-valid-test --no-progress-bar --task translation --max-target-positions $MAX_TGT_SENT_LEN --max-source-positions MAX_SRC_POSITIONS --outindices checkpoints/$IDXEXCLDIR/ignoredIndices.log --outindicesValid $OUTDIR$IDXEXCLDIR/valid_ignoredIndices.log 1> 'checkpoints/'$MODELNAME'/train.log'
```

```--outindices``` and ```--outindicesValid``` should point to files with list of excluded instances' indices. You should define the other variables as convenient.

##### ConvGCN
```
cd fairseq
CUDA_VISIBLE_DEVICES=$GPUID python train.py data-bin/$DATADIR --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_gcn_wikicatsum --save-dir checkpoints/$MODELNAME  --skip-invalid-size-inputs-valid-test --no-progress-bar --task gcn_wikicatsum --max-target-positions $MAX_TGT_SENT_LEN --max-source-positions MAX_SRC_POSITIONS --outindices checkpoints/$IDXEXCLDIR/ignoredIndices.log --outindicesValid $OUTDIR$IDXEXCLDIR/valid_ignoredIndices.log 1> 'checkpoints/'$MODELNAME'/train.log'
```

```--outindices``` and ```--outindicesValid``` should point to files with list of excluded instances' indices. You should define the other variables as convenient.


### Generate

Generating with obtained models.

##### ConvS2S

```
CUDA_VISIBLE_DEVICES=2 python my_generateSingle.py data-bin/$DATADIR --path checkpoints/$MODELNAME/checkpoint_best.pt --beam 5 --skip-invalid-size-inputs-valid-test --decode-dir $DECODEDIR --reference-dir $REFDIR --outindices $IDXEXCLDIR/valid_ignoredIndices.log --max-target-positions $MAX_TGT_SENT_LEN --quiet --gen-subset valid  1> $DECODEDIR/generate.log
```

You can also select best checkpoint based on ROUGE on valid:
```
export ARG_LIST="--beam 5 --skip-invalid-size-inputs-valid-test --reference-dir $REFDIR --outindices $IDXEXCLDIR/valid_ignoredIndices.log --max-target-positions $MAX_TGT_SENT_LEN --quiet "

CUDA_VISIBLE_DEVICES=$GPUID python run_dev_rouge.py \
--data-dir data-bin/$DATADIR \
--model-dir checkpoints/$MODELNAME \
--reference-dir $REFDIR \
--fconv
```

##### ConvGCN

Generating and Checkpoint selection based on ROUGE is similar to that of ConvS2S.

### Evaluation

#### ROUGE
Evaluation with ROUGE is based on the [pyrouge](https://github.com/bheinzerling/pyrouge) package. 

Rouge evaluation scripts are adapted from [here](https://github.com/ChenRocks/fast_abs_rl). ```my_generateSingle.py``` will save files for ROUGE evaluation. 

Install ```pyrouge``` ([pip install pyrouge](pypi.python.org/pypi/pyrouge)) and cloned it and configure ```ROUGE``` environment variable to the script within your
pyrouge directory. 

If you get a WordNet db error, proceed as explained [here](https://github.com/masters-info-nantes/ter-resume-auto/blob/master/README.md).

You can run the following to get ROUGE scores on the models' outputs:
```
export ROUGE=$HOME/pyrouge/tools/ROUGE-1.5.5/
export REFDIR=$REFDIR
python evaluation/eval_full_model.py --rouge --decode_dir $DECODEDIR
```

#### Content Metrics

To compute the additional *Abstract* and *Copy* metrics on models' outputs use the following command:
```
export DATADIR=$TEXTDATADIR
python evaluation/extra_metrics.py --decode-dir $DECODEDIR
```

```TEXTDATADIR``` is the directory that contains the text files of your dataset.


