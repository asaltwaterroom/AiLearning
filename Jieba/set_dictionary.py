'''
自然语言处理 - 2.分词
添加自定义词典
'''
'''
jieba 采用延迟加载，import jieba 和 jieba.Tokenizer() 不会立即触发词典的加载，一旦有必要才开始加载词典构建前缀字典。
jieba.initialize()  # 手动初始化（可选）
在 0.28 之前的版本是不能指定主词典的路径的，有了延迟加载机制后，你可以改变主词典的路径:
jieba.set_dictionary('data/dict.txt.big')
'''
#encoding=utf-8
import sys
sys.path.append("../../")
import jieba

#案例
def cuttest(test_sent):
    result = jieba.cut(test_sent)
    print("/".join(result))

def testcase():
    cuttest("这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。")
    cuttest("我不喜欢日本和服。")
    cuttest("雷猴回归人间。")
    cuttest("工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作")
    cuttest("我需要廉租房")
    cuttest("永和服装饰品有限公司")
    cuttest("我爱北京天安门")
    cuttest("abc")
    cuttest("隐马尔可夫")
    cuttest("雷猴是个好网站")
    cuttest("欢迎来到互联网创新办")
    cuttest("凱特琳是我女朋友")
    cuttest("台中台北有云计算吗")

if __name__ == "__main__":
    testcase()
    jieba.set_dictionary("testdic.txt")
    print("================================================")
    testcase()

'''
开发者可以指定自己自定义的词典，以便包含 jieba 词库里没有的词
用法: jieba.load_userdict(file_name) # file_name 为文件类对象或自定义词典的路径
1.词典格式和testdict.txt 一样，一个词占一行
2.每一行分三部分: 词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒
3.file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码
4.词频省略时使用自动计算的能保证分出该词的词频
'''
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
jieba.load_userdict("userdic.txt")
testcase()

'''
调整词典的词
使用 add_word(word, freq=None, tag=None) 和 del_word(word) 可在程序中动态修改词典。
使用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来。
注意: 自动计算的词频在使用 HMM 新词发现功能时可能无效
'''
print("****************************************************************")
print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
jieba.suggest_freq(('中', '将'), True)
print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))

print('/'.join(jieba.cut('「大护法」正确应该不会被切开', HMM=False)))
jieba.suggest_freq('大护法', True)
print('/'.join(jieba.cut('「大护法」正确应该不会被切开', HMM=False)))

'''
关键词提取
基于TF-IDF算法的关键词提取
该框架TF-IDF效果一般：有词才能计算IDF的值，框架要先提供IDF值才能计算最终的TF-IDF值
TF（term frequency），衡量一个词在一个文件中出现的频率，通常是一个词出现的次数除以文档的总长度。
IDF（inverse document frequency）逆向文件频率，衡量一个词的重要性/区分度。计算词频TF的时候，所有的词语都被当做一样重要的，
但是某些词，比如”is”, “of”, “that”很可能出现很多很多次，但是可能根本并不重要，因此需要减轻在多个文档中都频繁出现的词的权重。
TF(t) = (词t在某个文档中出现的总次数) / (某个文档的词总数)
IDF = log_e(总文档数/词t出现的文档数)
TF-IDF = TF*IDF
注意：非常短的文本很可能影响 tf-idf 值
'''
import jieba.analyse
"""
tags = jieba.analyse.extract_tags(sentence,topK=20,withWeight=False,allowPOS=())
* sentence 为待提取的文本
* topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
* withWeight 为是否一并返回关键词权重值，默认值为 False
* allowPOS 仅包括指定词性的词，默认值为空，即不筛选
* jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件
    # idf是jieba通过语料库统计得到的
    # idf的值时通过语料库统计得到的，所以，实际使用时，可能需要依据使用环境，替换为使用对应的语料库统计得到的idf值。
"""
jieba.analyse.set_stop_words("stop_word.txt")
jieba.analyse.set_idf_path("idf.txt.big")
content = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
print("-------------------------------------------------------------------------------")
for word,weight in jieba.analyse.extract_tags(content,withWeight=True):
    print("%s %s" %(word,weight))

'''
基于TextRank算法的关键词抽取
基本思想:
1.将待抽取关键词的文本进行分词
2.以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
3.计算图中节点的PageRank，注意是无向带权图
'''
print("-------------------------------------------------------------------------------")
print("%s" %(jieba.analyse.textrank(content)))

'''
词性标注
jieba.posseg.POSTokenizer(tokenizer=None) 新建自定义分词器, tokenizer 参数可指定内部使用的
jieba.Tokenizer 分词器, jieba.posseg.dt 为默认词性标注分词器
标注句子分词后每个词的词性，采用和 ictclas 兼容的标记法
除了jieba默认分词模式，提供paddle模式下的词性标注功能。paddle模式采用延迟加载方式，通过 enable_paddle() 安装 paddlepaddle-tiny，并且import相关代码
'''

'''
Tokenize: 返回词语在原文的起止位置
注意，输入参数只接受 unicode
'''
print('**************************************************************************')
result = jieba.tokenize(u'今天也是可爱的一只猫咪呢')
for tk in result:
    print("word:%s\t\t\t start:%d\t end:%d" %(tk[0],tk[1],tk[2]))


'''
ChineseAnalyzer for Whoosh 搜索引擎
引用: from jieba.analyse import ChineseAnalyzer
pip install whoosh
Whoosh是一个用来索引文本并能根据索引搜索的的包含类和方法的类库。它允许你开发一个针对自己内容的搜索引擎。
例如，如果你想创建一个博客软件，你可以使用Whoosh添加一个允许用户搜索博客类目的搜索功能。

'''