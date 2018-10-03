# SQuAD2.0 Baseline Test with BiDAF-No-Answer

This repository is the test of SQuAD2.0 Baseline with BiDAF-No-Answer. The source code for training and testing was copied from [https://bitbucket.org/omerlevy/bidaf_no_answer](https://bitbucket.org/omerlevy/bidaf_no_answer). The source code for evaluation and no answer probability extraction was copied from [https://worksheets.codalab.org/worksheets/0x9a15a170809f4e2cb7940e1f256dee55/](https://worksheets.codalab.org/worksheets/0x9a15a170809f4e2cb7940e1f256dee55/). Dockerfile for testing part and requirements.txt are also presented in the repository. It already contained a model for testing stored in the folder `./out` which we trained before.

## 0.Citation
If you use this code, please cite following papers:

Levy, Omer, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. "Zero-shot relation extraction via reading comprehension." arXiv preprint arXiv:1706.04115 (2017).[[pdf]](https://arxiv.org/pdf/1706.04115.pdf)[[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:1zV2iLT7iBkJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAW7MnqpcSipqFp9sJHLWFP_dWX1O0CKg2&scisf=4&ct=citation&cd=-1&hl=en)

Rajpurkar, Pranav, Robin Jia, and Percy Liang. "Know What You Don't Know: Unanswerable Questions for SQuAD." arXiv preprint arXiv:1806.03822 (2018).[[pdf]](https://arxiv.org/pdf/1806.03822.pdf)[[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:WsupEsFySccJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAW7Mo52h6VNN0kK0DjmM4gUpxK9oU_2Ez&scisf=4&ct=citation&cd=-1&hl=en)

## 1.Requirements
- python (version 3, issues reported with version 2)
- wget (for download.sh)
- python packages: nltk3.2.1, tqdm4.7.4, tensorflow0.12.1 and matplotlib

## 2.Running the Code
### 2.0 Install the dependencies:
- `wget` comes with most Linux distributions. If you use OS X, you can install wget with Homebrew:
```
brew install wget
``` 
- If you use the dockerfile provided, you **do not need to** install the python packages, otherwise you can install all python packages with `pip3 install package_name==version`

### 2.1 Download the dependencies: SQuAD2.0 dataset and glove vectors
```
chmod +x download.sh
./download.sh
``` 
For convenience of data preprocessing, training(if necessary) and testing, we change the filename from "\*2.0.json" to "\*1.1.json" when downloading.

Or you can download manually from [SQuAD train-v2.0](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json), [SQuAD dev-v2.0](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json), [glove.6B.100d.txt](https://worksheets.codalab.org/rest/bundles/0x15a09c8f74f94a20bec0b68a2e6703b3/contents/blob/). For the code to run properly, you also need to change file names as stated above.

### 2.2 Testing with pre-trained model
Use the dockerfile with following command:
```
docker build -t imagename .
docker run -it imagename cat /app/eval.json
``` 
After running the command, we should be able to see all the results printed on the screen. With the pre-trained model, the result should be: '{"exact": 32.87290491030068, "f1": 38.1342419703825, "total": 11873, "HasAns_exact": 65.82321187584346, "HasAns_f1": 76.36097417583525, "HasAns_total": 5928, "NoAns_exact": 0.01682085786375105, "NoAns_f1": 0.01682085786375105, "NoAns_total": 5945, "best_exact": 58.392992504000674, "best_exact_thresh": 0.027927149087190628, "best_f1": 61.062615045249075, "best_f1_thresh": 0.04179281368851662, "pr_exact_ap": 35.02907883753725, "pr_f1_ap": 44.91271869261081, "pr_oracle_ap": 69.62500467611224}'. 

Notice that '"best_f1": 61.062615045249075, "best_exact": 58.392992504000674,', which is similar to the the baseline mentioned in the paper: 'EM:59.2, F1:62.1'.

You could copy the evaluation file out with the command below:
```
docker cp containerID:/app/eval.json /host/path/target
``` 

### Extra Notes
If you want to run the whole BiDAF-No-Answer code including the training and testing without using docker, you could use following commands which are all copied from [https://worksheets.codalab.org/worksheets/0x9a15a170809f4e2cb7940e1f256dee55/](https://worksheets.codalab.org/worksheets/0x9a15a170809f4e2cb7940e1f256dee55/). Make sure your working directory is under the folder containing all the code in this directory.

#### Training
```
export CUDA_VISIBLE_DEVICES=0;
python3 -m squad.prepro --source_dir . --target_dir . --glove_dir .
python3 -m basic.cli --mode train --noload --batch_size 60 --sent_size_th 400 --num_steps 0 --num_epochs 15 --len_opt --cluster --num_gpus 1 --data_dir . --eval_period 2000

```

#### Testing
```
python3 -m squad.prepro --mode single --single_path data.json --target_dir inter --glove_dir .
python3 -m basic.cli --mode forward --batch_size 1 --len_opt --cluster --data_dir inter --eval_path inter/eval.pkl.gz --shared_path out/basic/00/shared.json --answer_path pred.json --device_type cpu
python3 extract_na_prob.py inter/eval.pkl.gz inter/data_single.json na_prob.json
python3 evaluate-v2.0.py data.json pred.json -o eval.json -n na_prob.json -p plots
```

Testing need to be run after training or pretrained model existing in the specified path.
