import math
from collections import Counter

import nltk
import numpy as np
import pandas as pd


# jaccard and cosine scoring functions courtesy of https://gist.github.com/gaulinmp/da5825de975ed0ea6a24186434c24fe4
def jaccard_distance(a, b):
    """Calculate the jaccard distance between sets A and B"""
    a = set(a)
    b = set(b)

    return 1.0 * len(a&b)/len(a|b)

def cosine_similarity(a, b):
    vec1 = Counter(a)
    vec2 = Counter(b)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0

    return float(numerator) / denominator

def ngSim(s1, s2):

    largestWordLen = max(map(len, (s1 + ' '+ s2).split(' ')))
    weights = np.arange(2, largestWordLen)
    scores = np.zeros(weights.size)

    for i in np.arange(weights.size):
        n = weights[i]
        s1grams = nltk.ngrams(s1, n)
        s2grams = nltk.ngrams(s2, n)
        scores[i] = cosine_similarity(s1grams, s2grams)

    return ((weights * scores).sum()/weights.sum())

def getClosest(i, scores):
    xscores = np.copy(scores.iloc[0].values)

    while True:
        maxx = xscores.argmax()

        if xscores[maxx] <= 0 :
            return -1
        maxy = scores.iloc[:, maxx].argmax()

        if maxy != i:
            xscores[maxx] = -1
        else:
            print('score : ', xscores[maxx])

            return int(maxx)


def normalizeStr(s):
    s = s.lower()
    s = ''.join(list(filter(lambda x: x.isalpha() or x == ' ', s)))

    return s

def stringMatch(rl1, rl2):
    l1 = pd.Series(rl1).map(normalizeStr).values
    l2 = pd.Series(rl2).map(normalizeStr).values

    lenl1 = l1.shape[0]
    lenl2 = l2.shape[0]

    scores = pd.DataFrame(np.zeros((lenl1, lenl2)))
    matches = np.zeros((lenl1), dtype=np.int64)

    print('computing score matrix..')
    # compute scores exhaustively (slowest part)

    for i in np.arange(lenl1):
        v = l1[i]
        scores.iloc[i] = pd.Series(l2).map(lambda x: ngSim(x, v)).values
        # scores.iloc[i] = np.array(list(map(lambda x: ngSim(x, v), l2)))

    print('Finding matches..')

    for i in np.arange(lenl1):
        m = getClosest(i, scores)
        matches[i] = m

        if m < 0:
            print('no match found for %s' % rl1[i])
        else:
            print('%s -> %s' % (rl1[i], rl2[m]))

    results = pd.DataFrame(columns=['String', 'Match'])
    results.loc[:, 'String'] = rl1
    vm = np.where(matches > 0)[0]
    results.loc[vm, 'Match'] = rl2[matches[vm]]
    results = results.fillna('no match')

    return results

def genRandSentences(maxsl=4, n=20):
    l = np.array([])
    wf = open('/etc/dictionaries-common/words', 'r')
    words = pd.Series(wf.read().split('\n'))

    for i in np.arange(n):
        sl = np.random.randint(1, maxsl)
        s = ' '.join(words.sample(sl).values)
        l = np.append(l, s)

    return l

def getDataset():
    f = open('paperTitles.txt', 'r') # paperTitles.txt generated from the titles of papers in the arxiv Dataset (https://www.kaggle.com/neelshah18/arxivdataset)
    titles = pd.Series(f.read().split('\n'))
    l1 = titles[titles.index % 2 != 0].values
    l2 = titles[titles.index % 2 == 0].values

    return l1, l2

l1, l2 = getDataset()
res = stringMatch(l1[:200], l2[:200])
