# -*- coding: utf-8 -*-


from __future__ import unicode_literals
import sys
import jieba
import jieba.posseg
import jieba.analyse
sys.path.append("../")
#分词
seg_list = jieba.cut("我来到北京清华大学",cut_all=True)
print ("Full Mode:", "/ ".join(seg_list)) #全模式

seg_list = jieba.cut("我来到北京清华大学",cut_all=False)
print ("Default Mode:", "/ ".join(seg_list)) #精确模式

seg_list = jieba.cut("他来到了网易杭研大厦") #默认是精确模式
print (", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造") #搜索引擎模式
print (", ".join(seg_list))

#添加词典
jieba.load_userdict("d:/data/dict.txt")  
seg_list = jieba.cut("他是创新办主任，也是云计算方面的专家") #默认是精确模式
print( ", ".join(seg_list))

#调整词典，.join 胶水连接函数
print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
#建议中将分开
jieba.suggest_freq(('中', '将'), True)

print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))

print('/'.join(jieba.cut('台中正确应该不会被切开', HMM=False)))

jieba.suggest_freq('应该不会', True)

print('/'.join(jieba.cut('台中正确应该不会被切开', HMM=False)))

#关键词提取

s = '''
此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册
资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。
目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。
'''
for x, w in jieba.analyse.extract_tags(s, topK=20, withWeight=True):
    print('%s %s' % (x, w))

#textrank
for x, w in jieba.analyse.textrank(s, withWeight=True):
    print('%s %s' % (x, w))

#词性标注
words = jieba.posseg.cut("我爱北京天安门")
for word, flag in words:
    print('%s %s' % (word, flag))
    
#Tokenize
result = jieba.tokenize(u'永和服装饰品有限公司')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
    
result = jieba.tokenize(u'永和服装饰品有限公司', mode='search')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))


   
