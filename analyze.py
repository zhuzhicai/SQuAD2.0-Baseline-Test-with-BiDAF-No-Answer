import codecs
import re
from itertools import groupby
import string
import ujson as json
import sys
import numpy as np
from docopt import docopt


def main():
    args = docopt("""
    Usage:
        analyze.py <test_set> <answer_file>
    """)
    q_aprf = read_results(args['<test_set>'], args['<answer_file>'])
    print(pretify(q_aprf))


def read_results(test_set, answer_file):
    with codecs.open(test_set, 'r', 'utf-8') as fin:
        data = [line.strip().split('\t') for line in fin]
    metadata = [x[:4] for x in data]
    gold = [set(x[4:]) for x in data]
    with codecs.open(answer_file, 'r', 'utf-8') as fin:
        results = [json.loads(line) for line in fin]
    answers = [a for i, a in sorted([(int(i), a) for i, a in results[0].items() if i != 'scores' and i != 'na'])]
    have_answers = parse_no_answers(results)

    telemetry = []
    for m, g, a, has in zip(metadata, gold, answers, have_answers):
        if not has:
            a = ''
        stats = score(g, a)
        telemetry.append([m[0], m[1], str(len(g) > 0), stats])
    return aprf(telemetry)


def parse_no_answers(results):
    p_answer = [a for i, a in sorted([(int(i), a) for i, a in results[0]['scores'].items()])]
    p_no_answer = [a for i, a in sorted([(int(i), a) for i, a in results[0]['na'].items()])]

    import numpy as np
    return [answer > no_answer for answer, no_answer in zip(p_answer, p_no_answer)]


def gb(collection, keyfunc):
    return [(k, list(g)) for k, g in groupby(sorted(collection, key=keyfunc), keyfunc)]


def aprf(g):
    tp, tn, sys_pos, real_pos = sum(map(lambda x: x[-1], g))
    total = len(g)
    # a = float(tp + tn) / total
    # nr = tn / float(total - real_pos)
    # npr = tn / float(total - sys_pos)
    if tp == 0:
        p = r = f = 0.0
    else:
        p = tp / float(sys_pos)
        r = tp / float(real_pos)
        f = 2 * p * r / (p + r)
    # return np.array((a, p, r, f, npr, nr))
    return np.array((p, r, f))


def score(gold, answer):
    if len(gold) > 0:
        gold = set.union(*[simplify(g) for g in gold])
    answer = simplify(answer)
    result = np.zeros(4)
    if answer == gold:
        if len(gold) > 0:
            result[0] += 1
        else:
            result[1] += 1
    if len(answer) > 0:
        result[2] += 1
    if len(gold) > 0:
        result[3] += 1
    return result


def simplify(answer):
    return set(''.join(c for c in t if c not in PUNCTUATION) for t in answer.strip().lower().split()) - {'the', 'a', 'an', 'and', ''}


def pretify(results):
    return ' \t '.join([': '.join((k, v)) for k, v in zip(['Precision', 'Recall', 'F1'], map(lambda r: '{0:.2f}%'.format(r*100), results))])


PUNCTUATION = set(string.punctuation)


if __name__ == "__main__":
    main()
