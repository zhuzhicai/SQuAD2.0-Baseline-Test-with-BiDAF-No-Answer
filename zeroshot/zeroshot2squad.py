import argparse
import csv
import json
import os
from random import Random
import hashlib
import codecs


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")

    parser.add_argument("levy_path")
    parser.add_argument("squad_path")
    parser.add_argument("--num", "-n", type=int, default=-1)
    parser.add_argument("--singlequestion", "-s", action='store_true')
    parser.add_argument("--anonquestions", "-a", action='store_true')
    parser.add_argument("--noquestions", "-q", action='store_true')
    parser.add_argument("--withmasks", "-m", action='store_true')

    return parser.parse_args()


def levy2squad(args):
    levy_path = args.levy_path
    squad_path = args.squad_path
    num = args.num
    rnd = Random(17)
    if args.singlequestion:
        r2q = dict()

    squad = {'data': [{'paragraphs': []}]}
    squad['version'] = '0.1'
    paras = squad['data'][0]['paragraphs']

    with open(levy_path, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for i, each in enumerate(reader):
            rel, ques_temp, ques_arg, sent = each[:4]
            if args.singlequestion:
                if rel in r2q:
                    ques_temp = r2q[rel]
                else:
                    r2q[rel] = ques_temp
            ques = ques_temp.replace('XXX', ques_arg)
            if args.noquestions:
                ques = rel
            elif args.anonquestions:
                ques = hashlib.sha1(codecs.encode(rel)).hexdigest()

            qa = {'question': ques, 'answers': []}
            qa['id'] = str(i)
            if len(each) > 4:
                ans_list = each[4:]
                indices = [(sent.index(ans), sent.index(ans) + len(ans)) for ans in ans_list]
                starts, ends = zip(*indices)
                ans_start = min(starts)
                ans_end = max(ends)
                ans = sent[ans_start:ans_end]
                qa['answers'].append({'text': ans, 'answer_start': ans_start})
            paras.append({'context': sent, 'qas': [qa]})
            
            if args.withmasks and len(each) > 4:
                qa = {'question': ques, 'answers': []}
                qa['id'] = str(i) + 'MASK'
                masked_ans = mask_answer(ans, rnd)
                masked_sent = sent[:ans_start] + masked_ans + sent[ans_end:]
                qa['answers'].append({'text': masked_ans, 'answer_start': ans_start})
                paras.append({'context': masked_sent, 'qas': [qa]})
            
            if args.num >= 0 and i + 1 == num:
                break
    
    with open(squad_path, 'w') as fp:
        json.dump(squad, fp)


def mask_answer(answer, rnd):
    return ''.join([rnd.choice('abcdefghijklmnopqrstuvwxyz') if c in ALPHANUM else c for c in answer])


ALPHANUM = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')


def main():
    args = get_args()
    levy2squad(args)


if __name__ == "__main__":
    main()
