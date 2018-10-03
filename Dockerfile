# base image
#FROM tensorflow/tensorflow:0.12.0-gpu-py3
FROM python:3-onbuild

# copy data to working directory
WORKDIR /app
COPY . /app

# install dependencies
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt')"

# training
#RUN export PYTHONPATH="bidaf-no-answer:$PYTHONPATH"
#RUN export CUDA_VISIBLE_DEVICES=5
#RUN python3 -m squad.prepro --source_dir . --target_dir . --glove_dir .
#RUN python3 -m basic.cli --mode train --noload --batch_size 60 --sent_size_th 400 --num_steps 0 --num_epochs 15 --len_opt --cluster --num_gpus 1 --data_dir . --eval_period 2000

# testing
RUN  python3 -m squad.prepro --mode single --single_path dev1.1.json --target_dir inter --glove_dir .
RUN python3 -m basic.cli --mode forward --batch_size 1 --len_opt --cluster --data_dir inter --eval_path inter/eval.pkl.gz --shared_path out/basic/00/shared.json --answer_path pred.json --device_type cpu
RUN python3 extract_na_prob.py inter/eval.pkl.gz inter/data_single.json na_prob.json
RUN python3 evaluate-v2.0.py dev1.1.json pred.json -o eval.json -n na_prob.json -p plots 
RUN cat eval.json

# THE RESULT IS IN eval.json


