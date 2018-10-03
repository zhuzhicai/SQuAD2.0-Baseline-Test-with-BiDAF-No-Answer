python -m zeroshot.zeroshot2squad ~/data/$1/train ~/data/${1}/train-v1.1.json
python -m zeroshot.zeroshot2squad ~/data/$1/test ~/data/${1}/dev-v1.1.json
python -m squad.prepro -s ~/data/${1} -t data/${1}
