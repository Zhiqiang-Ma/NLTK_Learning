# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:56:18 2020

@author: Administrator
"""
# -*- coding: utf-8 -*-
####分类####
##性别鉴定##
#构造特征提取器
import nltk
import random
#引入姓名库
from nltk.corpus import names
# 定义特征提取函数，返回Word 中的最后一个字母
def gender_features(word):
    return {'last_letter': word[-1]}
gender_features('Shrek')
names_set = ([(name, 'male') for name in names.words('male.txt')] +
        [(name, 'female') for name in names.words('female.txt')])
#print (names_set[:10])
#洗牌
random.shuffle(names_set)
#print (names_set[:10])
#构造特征
featuresets = [(gender_features(n), g) for (n,g) in names_set]
#分割数据集
train_set, test_set = featuresets[500:], featuresets[:500]
#模型训练
classifier = nltk.NaiveBayesClassifier.train(train_set)
#测试
classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))
#分类正确率判断
print (nltk.classify.accuracy(classifier, test_set))
#最有效的特征
classifier.show_most_informative_features(5)

#%%
#大型数据时的数据集划分
from nltk.classify import apply_features
train_set = apply_features(gender_features, names_set[500:])
test_set = apply_features(gender_features, names_set[:500])
##手动计算贝叶斯分类器##
#计算P(特征|类别)
def f_c(data,fea,cla):
    cfd=nltk.ConditionalFreqDist((classes,features) for (features,classes) in data)
    return cfd[cla].freq(fea)

#计算P(特征)  
def p_feature(data,fea):
    fd=nltk.FreqDist(fea for (fea,cla) in data)
    return fd.freq(fea)
   
#计算P(类别)
def p_class(data,cla):
    fd=nltk.FreqDist(cla for (fea,cla) in data)
    return fd.freq(cla)
    
#计算P(类别│特征)
def res(data,fea,cla):
    return f_c(data,fea,cla)*p_class(data,cla)/p_feature(data,fea)
    
#构造输入数据集
data=([(name[-1], 'male') for name in names.words('male.txt')] +
        [(name[-1], 'female') for name in names.words('female.txt')])
random.shuffle(data)
train,test=data[500:],data[:500]

#计算Neo的为男性的概率
res(train,'k','male')
res(train,'a','female')
res(train,'n','female')

##选择正确的特征##
#过度拟合
def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features
    
gender_features2('John')

featuresets = [(gender_features2(n), g) for (n,g) in names_set]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))

#%%
#数据划分为训练集、开发测试集、测试集
train_names = names_set[1500:]
devtest_names = names_set[500:1500]
test_names = names_set[:500]

#重新训练模型
train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
test_set = [(gender_features(n), g) for (n,g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set) 
print (nltk.classify.accuracy(classifier, devtest_set))

#打印错误列表
errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append( (tag, guess, name) )
for (tag, guess, name) in sorted(errors): # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    print ('correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name))
    
#重新构建特征
def gender_features(word):
    return {'suffix1': word[-1:],
            'suffix2': word[-2:]}    
 
#重新训练模型           
train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, devtest_set))