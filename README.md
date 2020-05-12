# voice-search-board

This work is extended over the paper [Global-Locally Self-Attentive Dialogue State Tracker (GLAD)](https://arxiv.org/abs/1805.09655). 

If you do not want to build the Docker image, then run the following (you still need to have the CoreNLP server).

```
pip install -r requirements.txt
```

# Download and annotate data

This project uses Stanford CoreNLP to annotate the dataset.
In particular, we use the [Stanford NLP Stanza python interface](https://github.com/stanfordnlp/stanza).
To run the server, do

Install CoreNLP and then run the following to start the CoreNLP server
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

The first time you preprocess the data, we will [download word embeddings and character embeddings and put them into a SQLite database](https://github.com/vzhong/embeddings), which will be slow.
Subsequent runs will be much faster.

```
python preprocess_data.py
```
The data should be present in the directory structure `<code-directory>/data/dstc/ann/<json-files>`


# Train model

You can checkout the training options via `python train.py -h`.
By default, `train.py` will save checkpoints to `exp/glad/default`.

```
python train.py --gpu 0
```
To run with the cleaned DSTC2 dataset, run
```
python train.py --gpu 0  --dataset dstc 
```
To evaluate with ASR utterances during training, run
```
python train.py --gpu 0  --dataset dstc --infer_with_asr
```
To train and evaluate with confnet during training, run
```
python train.py --dataset dstc --gpu 0 --dexp <path-to-checkpoint> --local_dropout 0.2  --global_dropout 0.2 --emb_dropout 0.2 --word_dropout 0.0 --batch_size 50 --lr 0.001 --eps 0.00000001 --threshold 0.5 --selfattn_dropout 0 --train_using confnet --model glad_with_confnet --infer_with_confnet --epoch 70 --max_par_arc <confnet_par_arc_size>
```

To use different training settings, use argument `--train_using` with:
```
--train_using aug_confnet
--train_using confnet
--train_using asr
--train_using aug_asr
--train_using transcript
```
This saves models at `exp/glad_with_confnet/default`

To use similarity loss, use:
```
python train.py --dataset dstc --gpu 0 --dexp <path-to-checkpoint> --local_dropout 0.2  --global_dropout 0.2 --emb_dropout 0.2 --word_dropout 0.0 --batch_size 50 --lr 0.001 --eps 0.00000001 --threshold 0.5 --selfattn_dropout 0 --train_using confnet --model glad_with_confnet --infer_with_confnet --epoch 70 --max_par_arc <confnet_par_arc_size> --joint_training
```


# Evaluation

You can evaluate the model using

You can also dump a predictions file by specifying the `--fout` flag.
In this case, the output will be a list of lists.
Each `i`th sublist is the set of predicted slot-value pairs for the `i`th turn.
Please see `evaluate.py` to see how to match up the turn predictions with the dialogues.


```
python evaluate.py --gpu 0 --split test exp/glad_with_confnet/default
```

To infer with ASR utterances, run

```
python evaluate.py --gpu 0 --split test exp/glad_with_confnet/default --infer_with_asr
```
To infer with confnet, run
```
python evaluate.py <path-to-checkpoint> --split test --gpu 0 --dataset dstc --infer_with_confnet
```
To check inferene time for 1 forward pass, 
```
python evaluate.py <path-to-checkpoint> --split test --gpu 0 --dataset dstc --infer_with_confnet --forward_pass_time
```

Different experiment setting:
1) ver1: Use ![img](http://latex.codecogs.com/svg.latex?Embedding_%7BCN%7D%28C_t%29%26%3D%26%5Csum_i%5Cpi_t%5Eiq_t%5Ei) instead of ![img](http://latex.codecogs.com/svg.latex?Embedding_%7BCN%7D%28C_t%29%26%3D%26%5Csum_i%5Calpha_t%5Eiq_t%5Ei)

2) ver2: Change eq 4 to ![img](http://latex.codecogs.com/svg.latex?Embedding_%7BCN%7D%28C_t%29%26%3D%26%5Csum_ip_t%5Ei)

3) ver3: Change eq 4 to ![img](http://latex.codecogs.com/svg.latex?Embedding_%7BCN%7D%28C_t%29%3D%5Csum_i%5Cpi_t%5Ei%5Ctanh%28W_1Embedding%28w_t%5Ei%29%29)

4) Remove pi from eq 1.
