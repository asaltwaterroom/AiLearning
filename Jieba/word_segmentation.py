'''
自然语言处理 - 2.分词
分词(Word Segmentation): 将连续的自然语言文本，切分成具有语义合理性和完整性的词汇序列
'''
#encoding=utf-8
import jieba #中文分词组件
#四种分词模式
#1.精确模式（默认）：试图将句子最精确地切开，适合文本分析
seg_list = jieba.cut("他来到了网易杭研大厦")
print(",".join(seg_list))

seg_list = jieba.cut("我来到华南理工大学",cut_all=False)
print("Default Mode:" + "/".join(seg_list))

#2.全模式:把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义
seg_list = jieba.cut("我来到华南理工大学",cut_all=True)
print("Full Mode:" + "/".join(seg_list))

#3.搜索引擎模式: 在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print("Search mode:" + "/".join(seg_list))

#4.paddle模式: 利用Paddle深度学习框架，训练序列标注（双向GRU）网络模型实现分词，同时支持词性标注
jieba.enable_paddle() #启动paddle模式，0.40版本后支持
strs = ["我来到北京清华大学", "乒乓球拍卖完了", "中国科学技术大学"]
for str in strs:
    seg_list = jieba.cut(str,use_paddle=True)
    print("Paddle mode:" + "/".join(list(seg_list)))

