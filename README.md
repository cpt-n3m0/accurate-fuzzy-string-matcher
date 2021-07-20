* Scoring approach
  + Calculate the weighted average of the cosine similarity of the strings' n-grams for values of n ranging from 2 to the length of the largest words, and where the weigths are the ngram length to give preference to larger shared substrings and reduce false positives
The approach consists of generating a score matrix
* String matching between 2 lists of strings
  +  Improves on the greedy approach of matching with the string with the highest score (can also be thought of as shortest distance) by adding the additional constraint that the matched string not only has to be the closest one but also the farthest from all other strings, i.e. no closer to any other string, and so I think it should be optimal assuming the score (distance) is accurate
