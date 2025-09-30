---
title: "To ken izers"
date: 2025-29-09
draft: false
toc: false
math: true
tags:
  - NLP, LLM
---
# An overview of different commonly used tokenizers

Langauge models have taken the world by storm, thanks to transformers ([Vaswani et al.](https://arxiv.org/abs/1706.03762)). A fundamental question when modeling language is how to convert words into numerical representations that computers can understand. One obvious way is to think of each unique word as a one-hot encoded vector of dimension equal to the size of our vocabulary. This is a bit of a problem since there are over 150,000 words in the english language! Surely we can do better than assigning a unique vector to each word in our vocabulary. Behold, tokenizers. Tokenizers are basically algorithms that help prepare a vocabulary for language models (hopefully one that is smaller than the set of unique words in the corpus). Let's jump right into different tokenizers that are commonly used today.

## Byte-Pair Encoding (BPE)

[Byte-Pair Encoding](https://arxiv.org/abs/1508.07909) is an iterative algorithm that builds a vocabulary by identifying the most frequently occuring pairs of symbols in our "fragmented corpus" and merging them to produce a new (sub)word. We start with a base vocabulary consisting of the unique characters appearing in our corpus, after applying some pre-tokenization (lowercasing all words, removing punctuation, etc.). Let's say our corpus is 
```python
"Hello there, how are you?"
```
Then, our base vocabulary $V_0$ would be `['h','e','l','o','t','h','r','w','a','y','u']` and fragmented corpus $C_0$ would be`[('h', 'e', 'l', 'l', 'o'), ('t', 'h', 'e', 'r', 'e'), ('h', 'o', 'w'), ('a', 'r', 'e'), ('y', 'o', 'u')]`. In the first iteration of the algorithm, the most frequently appearing pair of characters in the corpus would be merged and added to the vocabulary. The fragmented corpus would then be modified by performing the same merge across the words as well. In our case, the most frequently occurring pair is `'h', 'e'`. We merge it to create a new subword `'he'` and add it to the vocabulary. Our new vocabulary $V_1$ is `['h','e','l','o','t','h','r','w','a','y','u','he']` and the fragmented corpus $C_1$ is `[('he', 'l', 'l', 'o'), ('t', 'he', 'r', 'e'), ('h', 'o', 'w'), ('a', 'r', 'e'), ('y', 'o', 'u')]`. Repeating this for one more step, we get $V_2$ is `['h','e','l','o','t','h','r','w','a','y','u','he','re']` and $C_2$ is `[('he', 'l', 'l', 'o'), ('t', 'he', 're'), ('h', 'o', 'w'), ('a', 're'), ('y', 'o', 'u')]`. We stop when we've exhausted the fragmented corpus (all single words) or we've reached the required vocabulary size (hyperparameter chosen by us). Usually, an end-of-sentence `<EOS>` and catch-all unknown `<UNK>` tokens are included in the vocabulary. GPT-1 uses BPE with a vocabulary size of 40,000.

The order of merges is important since it decides how a new sentence is tokenized.  Note that we have arbitrarily broken ties for the most frequent pair here. Given a new sentence, we tokenize it by applying merge rules in the same order they were added to the vocabulary. For example. the word 'Cherry' would be tokenized as
```python
'cherry' -> '<UNK>' 'h' 'e' 'r' 'r' 'y' -> '<UNK>' 'he' 'r' 'r' 'y'
```
Since the character `'c'` was not in the vocabulary, it was tokenized with the catch-all token `<UNK>`. Here's a naive implementation of [BPE in python](https://github.com/shankram/LLMs-from-scratch/blob/main/Tokenizers/Tokenizers.ipynb).

## WordPiece Encoding

WordPiece is also an iterative algorithm similar to BPE except for three differences:
* All characters and subwords in the fragmented corpus (except prefixes) are encoded with `##` before the word. For example, the word `'how'` in $C_0$ is `('h', '##o', '##w')`. `'##o'` and `'##w'` would be merged as `'##ow'`.
* The pair to be merged is chosen using a score 
$$ \text{score} = \text{freq of pair} / (\text{freq of first element}\times \text{freq of second element})$$ The score is designed to prioritize merging pairs that occur together frequently but don't occur by themselves in other words. For example, consider the pairs `('ad', '##vantage')` and `('un', '##able')`. `'un'` and `'able'` occur very frequently as parts of other words (undo, capable, etc.) but `'ad'` and `'vantage'` probably occur less frequently so adding `'advantage'` to the vocabulary before `unable` makes sense (assuming the two words fairly frequently in the corpus to begin with).
* The merge rules are not stored, only the final vocabulary.

A new word is tokenized by recursively finding the longest prefix of the word from the vocabulary. If any part of the remaining word can't be tokenized with the vocabulary, we tokenize the entire word as `'<UNK>'`. Here are two examples to illustrate. We don't go into the corpus or the vocabulary to sidestep tedious details.
```python
'heresay' -> 'he' '##resay' -> 'he' '##re' '##say'  
```
```python
'totem' -> 'to' '##tem' -> 'to' '##te' '##m' -> 'to' '##te' '<UNK>' -> '<UNK>' 
```

## Unigram Encoding