import json
from .config import config
import jieba
from opencc import OpenCC 

def toJson(_object):
    return json.dumps(_object, ensure_ascii=False)

def writeJson(path, json_object):
    f = open(config.data_path + path+'.json', 'w', encoding='utf-8')
    f.write(json.dumps(json_object, ensure_ascii=False))  #, sort_keys=True, indent=1
    f.close()
    return

def loadJson(path):
    f = open(path, 'r', encoding='utf-8')
    _object = json.loads(f.read())
    f.close()
    return _object

def writeCsv(path, arr):
    f = open(config.data_path + path+'.csv', 'w', encoding='utf-8')
    f.write('\n'.join([','.join(item) for item in arr]))
    f.close()

def loadCsv(path):
    f = open(path, 'r', encoding='utf-8')
    rows  = f.read().strip('\n').split('\n')
    f.close()
    return [row.split(',') for row in rows]

stop_words = set(list(r'。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼ '))
def filter(words):
    return [word for word in words if word not in stop_words]

new_words = open(config.new_word_path, "r",encoding='utf-8').readlines()
for new_word in new_words:
    jieba.add_word(new_word)
    
def seg(sentence):
    return list(jieba.cut(sentence))

def segWithFilter(sentence):
    return filter(list(jieba.cut(sentence)))

def segAllWithFilter(sentence):
    return filter(list(sentence))

def segCutAllWithFilter(sentence):
    return filter(list(jieba.cut(sentence, cut_all=True)))

def segUesdInCorups(sentence):
    return segWithFilter(sentence)+segAllWithFilter(sentence)

# 似乎这么写会影响性能呀
def genOrCondition(key, arr):
    if len(arr) == 0:
        return '({}!={})'.format(key, key)

    sql_str = '(' + ' OR '.join(['({}=\'{}\')'.format(key, item) for item in arr]) + ')'
    # print(sql_str)
    return sql_str

cc1 = OpenCC('s2t')
def s2t(string):
	return cc1.convert(string)

cc2 = OpenCC('t2s')
def t2s(string):
	return cc2.convert(string)