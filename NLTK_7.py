# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:00:26 2020

@author: Administrator
"""
import nltk
import random
#引入姓名库
from nltk.corpus import names
# 定义特征提取函数，返回Word 中的最后一个字母
#**************Part one : Document classification using NLTK *************
##文档分类##
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

#文档分类特征提取器
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.most_common()[:2000]
def document_features(document):
    document_words = set(document)
    features = {}
    for (word,freq) in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
print (document_features(movie_reviews.words('pos/cv957_8737.txt')))

#构造分类器
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))
print (classifier.show_most_informative_features(5))

##词性标注##
from nltk.corpus import brown
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
common_suffixes = suffix_fdist.most_common()[:100]
print (common_suffixes)
#定义特征提取器
def pos_features(word):
    features = {}
    for (suffix,freq) in common_suffixes:
        features['endswith(%s)' % suffix] = word.lower().endswith(suffix)
    return features
    
#训练分类器
tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[:1000], featuresets[2000:3000]
classifier = nltk.DecisionTreeClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))
print (classifier.classify(pos_features('cats')))

#决策树输出
print (classifier.pseudocode(depth=4))

#根据上下文构造特征提取器
def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features
    
pos_features(brown.sents()[0], 8)

tagged_sents = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append((pos_features(untagged_sent, i), tag) )
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)

#%%*************************Part two : Sequence classification *******
##序列分类##
#定义特征提取器
def pos_features(sentence, i, history): 
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
        features["prev-tag"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
        features["prev-tag"] = history[i-1]
    return features

#构建序列分类器    
class ConsecutivePosTagger(nltk.TaggerI): 
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = pos_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = pos_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)
        
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
tagger = ConsecutivePosTagger(train_sents)
print (tagger.evaluate(test_sents))

#%%*******************Part three : Sentence segmentation*********************
##句子分割##
#获取已分割的句子数据
sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for sent in nltk.corpus.treebank_raw.sents():
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset-1)

#定义特征提取器
def punct_features(tokens, i):
    return {'next-word-capitalized': tokens[i+1][0].isupper(),
    'prevword': tokens[i-1].lower(),
    'punct': tokens[i],
    'prev-word-is-one-char': len(tokens[i-1]) == 1}

#定义标注    
featuresets = [(punct_features(tokens, i), (i in boundaries))
    for i in range(1, len(tokens)-1)
    if tokens[i] in '.?!']
        
#构建分类器
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)

#基于分类的断句器
def segment_sentences(words):
    start = 0
    sents = []
    for i, word in words:
        if word in '.?!' and classifier.classify(words, i) == True:
            sents.append(words[start:i+1])
            start = i+1
    if start < len(words):
        sents.append(words[start:])
        
##识别对话行为类型##
posts = nltk.corpus.nps_chat.xml_posts()[:10000]

#定义特征提取器
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains(%s)' % word.lower()] = True
    return features

#训练分类器
featuresets = [(dialogue_act_features(post.text), post.get('class'))
                for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))


#%%*********************** Part Four : Results Assessment ********************
####评估####
#创建训练集与测试集
import random
from nltk.corpus import brown
tagged_sents = list(brown.tagged_sents(categories='news'))
random.shuffle(tagged_sents)
size = int(len(tagged_sents) * 0.1)
train_set, test_set = tagged_sents[size:], tagged_sents[:size]

#使用同类型文件
file_ids = brown.fileids(categories='news')
size = int(len(file_ids) * 0.1)
train_set = brown.tagged_sents(file_ids[size:])
test_set = brown.tagged_sents(file_ids[:size])

#使用不同类型文件
train_set = brown.tagged_sents(categories='news')
test_set = brown.tagged_sents(categories='fiction')

##准确度##
names_set = ([(name, 'male') for name in names.words('male.txt')] +
        [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names_set)

featuresets = [(gender_features(n), g) for (n,g) in names_set]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print ('Accuracy: %4.2f' % nltk.classify.accuracy(classifier, test_set))

##精准度与召回率##
from sklearn.metrics import classification_report
test_set_fea=[features for (features,gender) in test_set]
test_set_gen=[gender for (features,gender) in test_set]
pre=classifier.classify_many(test_set_fea)
print(classification_report( test_set_gen,pre))

##混淆矩阵##
cm = nltk.ConfusionMatrix(test_set_gen,pre)
print (cm)

####决策树####
#熵和信息增益
import math
def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in nltk.FreqDist(labels)]
    return -sum([p * math.log(p,2) for p in probs])
print (entropy(['male', 'male', 'male', 'male']))

print (entropy(['male', 'female', 'male', 'male']))

print (entropy(['female', 'male', 'female', 'male']))

print (entropy(['female', 'female', 'male', 'female']))

print (entropy(['female', 'female', 'female', 'female']))



    