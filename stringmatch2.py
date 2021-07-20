import nltk
import numpy as np
import pandas as pd


def jaccard_distance(a, b):
    """Calculate the jaccard distance between sets A and B"""
    a = set(a)
    b = set(b)

    return 1.0 * len(a&b)/len(a|b)

def ngSim(s1, s2):

    largestWordLen = max(map(len, (s1 + ' '+ s2).split(' ')))
    weights = np.arange(2, largestWordLen)
    scores = np.zeros(weights.size)

    for i in np.arange(weights.size):
        n = weights[i]
        s1grams = nltk.ngrams(s1, n)
        s2grams = nltk.ngrams(s2, n)
        scores[i] = jaccard_distance(s1grams, s2grams)

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
    l1 = titles[titles.index % 2 != 0]
    l2 = titles[titles.index % 2 == 0]

    return l1, l2

l1, l2 = getDataset()
res = stringMatch(l1[:1000], l2[:1000])
