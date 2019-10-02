import json
from django.http import HttpResponse
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sqlite3
import  re
import random
from .data_process.config import config
from .data_process.commonFunction import writeJson, loadJson, genOrCondition, s2t, t2s, toJson, segWithFilter, seg, segUesdInCorups,loadCsv
import urllib.request
from urllib.parse import quote
from gensim import corpora, models, similarities
import traceback
import string
import gensim

print('开始加载模型')
word2vec_model = gensim.models.Word2Vec.load(config.word_model_path)
sentence_model = Doc2Vec.load(config.sentence_model128_path)
print('模型加载完毕')
# potery

potery_ids = []
potery_num = 0

imp_author_ids = []
sentence_ids = []
def init():
    global potery_ids, potery_num, imp_author_ids, sentence_ids

    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()
    rows = db.execute('SELECT id from potery ORDER BY rank ASC')

    potery_ids = [row[0] for row in rows]
    potery_num = len(potery_ids)

    rows = db.execute('SELECT id from author  WHERE sims_tfidf_self IS NOT NULL')
    rows = list(rows)
    imp_author_ids = set([row[0] for row in rows])

    rows = list(db.execute('SELECT id FROM sentence'))
    sentence_ids = [row[0] for row in rows]
    conn.close()

init()

def jsonHttp(json_object):
    return HttpResponse(json.dumps(json_object))

def textHttp(text):
    return  HttpResponse(text)


def getSomePoteries(request):
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    data = []
    for time in range(10):
        random_index = random.randint(0, potery_num-1)
        random_potery_id = potery_ids[random_index]
        # print(random_potery_id)
        rows = db.execute('SELECT sim_potery_ids FROM potery WHERE id = ?', (random_potery_id, ))
        rows = list(rows)
        sims = json.loads(rows[0][0])
        
        sim_potery_ids = [sim[0] for sim in sims][:10]
        # print(sim_potery_ids)
        rows = db.execute('SELECT  id,content FROM potery WHERE {}'.format(genOrCondition('id',sim_potery_ids)))
        rows = list(rows)
        data.append(rows)
    conn.close()
    return jsonHttp(data)


def getSomeAuthors(request):
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    data = []
    random_index = random.randint(0, len(imp_author_ids)-1)
    random_author_id = list(imp_author_ids)[random_index]  #'苏轼' #

    # print(author_cat_num, random_cat)
    rows1 = db.execute('SELECT id, sims_tfidf_self FROM author WHERE id=?', (random_author_id,))
    rows1 = list(rows1)
    sims = json.loads(rows1[0][1])
    author_ids = [sim[0] for sim in sims]

    sql2 = 'SELECT id, sims_tfidf_self FROM author WHERE {} AND sims_tfidf_self IS NOT NULL'.format(genOrCondition('id', author_ids))
    # print(sql2)
    rows2 = db.execute(sql2)
    rows2 = list(rows2)

    # print(len(rows))
    links = set()
    for row in rows2 + rows1:
        id = row[0]
        sims = row[1]
        sims = json.loads(sims)
        if id != random_author_id:
            sims = sims[0:3]
        for sim in sims:
            # print(sim)
            sim_id =sim[0]
            sim = sim[1]
            if sim_id in imp_author_ids and id in imp_author_ids and sim_id==id and sim>0.8:
                continue

            if id<sim_id:
                link = '{},{},{}'.format(id, sim_id, sim)
            else:
                link = '{},{},{}'.format(sim_id, id, sim)
            links.add(link)
    
    links = [link.split(',') for link in links]
    # , float(link[2])
    links = [[link[0], link[1], float(link[2])] for link in links]
    

    author_ids = set()
    for link in links:
        author_ids.add(link[0])
        author_ids.add(link[1])
    rows = db.execute('SELECT id, rank FROM author WHERE {}'.format(genOrCondition('id', author_ids)))
    author2rank = {row[0]:row[1] for row in rows}
    conn.close()
    return jsonHttp({
        'potery_reltions': links, 
        'social_relations': getRelationsBetween(author_ids), 
        'author2rank': author2rank
    })

# ignore_relations = ["文风效法Y", "为Y之建筑物题咏记命名", ]
relations = loadCsv(config.author_relations_path)
def getRelationsBetween(author_names):
    author_names = set(author_names)

    results = []
    for relation in relations:
        p1, p2, r_type, is_direct = relation
        if p1 in author_names and p2 in author_names:
            results.append(relation)
    return results

def getPoteryInfo(request):
    p_id  = request.GET.get('potery_id')
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    rows = db.execute('SELECT * FROM potery WHERE id=?', (p_id, ))
    rows = list(rows)
    if len(rows)==0:
        return jsonHttp({'msg': 'no data'})
    row = rows[0]
    sentence_ids = json.loads(row[3])
    potery_info = {
        'id': row[0],
        'poet_id': row[2],
        'sentences': [],
    }
    for s_id in sentence_ids:
        rows = db.execute('SELECT * FROM sentence WHERE id=?', (s_id, ))
        rows = list(rows)
        row = rows[0]
        sentence = {
            'id': s_id,
            'content': row[1],
            'index': row[2],
            'comments': row[3],
            'words': json.loads(row[4]),
            'sims': json.loads(row[5])
        }
        potery_info['sentences'].append(sentence)
    conn.close()
    return jsonHttp(potery_info)


def getAuthorInfo(request):
    a_id  = request.GET.get('author_id')
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    
    rows = db.execute('SELECT info FROM author WHERE id=?', (a_id, ))
    rows = list(rows)
    # print(rows)
    if len(rows)==0:
        return jsonHttp({'msg': 'no data'})
    info =  rows[0][0]
    # print(info)
    potery_rows = db.execute('SELECT id, content FROM potery WHERE poet_id=?', (a_id, ))
    potery_rows = list(potery_rows)
    potery_rows = [[row[0], row[1]]  for row in potery_rows]
    if info is not None:
        # info = json.loads(info)
        return textHttp(info)
    else:
        try:
            t_name = s2t(a_id)
            url = 'https://cbdb.fas.harvard.edu/cbdbapi/person.php?name=' + t_name + '&o=json'
            print('下载',url)
            url = quote(url,safe=string.printable)
            response = urllib.request.urlopen(url, timeout=30)
            response = response.read().decode('utf-8')
            response = json.loads(response)
            response['poteries'] = potery_rows
            response = toJson(response)
            response = t2s(response)
            db.execute('UPDATE author SET info=? WHERE id=?',(response, a_id))
            conn.commit()
            return textHttp(response)
        except:
            # traceback.print_exc()
            return jsonHttp({'msg': 'no data in cbdb', 'poteries': potery_rows})
        finally:
            conn.close()
            # return jsonHttp({'msg': 'error', })

def getSomeWords(request):
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    rows = db.execute('SELECT word, count FROM word WHERE is_common IS NULL AND word_len>1 ORDER BY count DESC LIMIT 200')
    rows = list(rows)
    conn.close()
    # print(rows)
    return jsonHttp(rows)

def getRelatedWords(request):
    word = request.GET.get('word')

    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    limit_word_len = 1 if len(word)==1 else 2
    sims = word2vec_model.wv.most_similar(positive=[word], topn=5000)
    sims = [[sim[0], sim[1]] for sim in sims if len(sim[0])>=limit_word_len][:200]

    conn.close()
    return jsonHttp(sims)

sentiments = ['喜', '怒', '哀', '乐', '思']
def analyzeSentiment(word):
    if word not in word2vec_model.wv:
        return ['', 0]
    
    best_prob_sentiment = sentiments[0]
    best_prob = word2vec_model.similarity(word, best_prob_sentiment)
    for sentiment in sentiments:
        prob = word2vec_model.similarity(word, sentiment)
        if prob > best_prob:
            best_prob = prob
            best_prob_sentiment = sentiment
    return [best_prob_sentiment, float(best_prob)]


# sentence_tfidf_model = models.TfidfModel.load(config.sentence_tfidf_model)
# sentence_dictionary = corpora.Dictionary.load(config.sentence_tfidf_dict_path)# 生成字典和向量语料
# # sentence_similarity = similarities.Similarity(sim_path, corpus_tfidf , num_features=len(sentence_dictionary))
# sentence_similarity = similarities.Similarity.load(config.sentence_tfidf_sim_index)
# sentence_similarity.num_best = 15
# 和上面的重复了
def analyzePotery(request):
    print('1')
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    pid = request.GET.get('pid')
    rows = list(db.execute('SELECT poet_id, content FROM potery WHERE id = ?', (pid, )))
    if len(rows)==0:
        return jsonHttp({'msg': 'no data'})
    peot_id = rows[0][0]
    content = rows[0][1]
    words = segWithFilter(content)
    print('2')
    word2sentiment = {word: analyzeSentiment(word) for word in words}
    print('3')
    rows = list(db.execute('SELECT id, content, comments, words, sim FROM sentence WHERE potery_id = ? ORDER BY sentence."index" ASC', (pid, )))
    rows = [list(row) for row in rows]
    sentence_ids = [row[0] for row in rows]
    print('4')
    # seg_contents = [segUesdInCorups(row[1]) for row in rows]
    # sentences = []
    for index, row in enumerate(rows):
        id = row[0]
        row[3] = json.loads(row[3])
        print('8')
        if row[4] is None:
            sims = sentence_model.docvecs.most_similar([id], topn=10)
            sims = [[sim[0], sim[1]] for sim in sims]
            # db.execute('UPDATE sentence SET sim=? WHERE id=?', (toJson(sims), id))
            # conn.commit()
        else:
            sims = json.loads(row[4])
        # doc_bow =  sentence_dictionary.doc2bow(seg_content)
        # vec_tfidf = sentence_tfidf_model[doc_bow]
        # sims = sentence_similarity[vec_tfidf]
        # sims.sort(key=lambda item: -item[1])
        # print(sims)
        # sims = [sentence_ids[int(sim[0])] for sim in sims]
        print('5')
        sims = [sim[0] for sim in sims]
        sim_sentences = []
        for sim_id in sims:
            s_rows = db.execute('SELECT id, potery_id, content FROM sentence WHERE id = ?', (sim_id, ))
            s_row = [list(elm) for elm in s_rows][0]
            sim_sentences.append(s_row)
        rows[index].append(sim_sentences)
        print('6')
    print('7')
    data = {
        'potery_id': pid,
        'poet': peot_id,
        'sentences': rows,
        'word2sentiment': word2sentiment
    }
    conn.close()
    return jsonHttp(data)
    
def analyzeWritePotery(request):
    content = request.GET.get('content')
    # print(content)
    words = segWithFilter(content)
    # print(words)
    word2info = {}
    for word in words:
        word_info = { 'sim': [], 'sentiment': None}
        if word in word2vec_model.wv:
            word_len = len(word)
            sims = word2vec_model.wv.most_similar(positive=[word], topn=1000)
            # print(word, sims)
            sims = [sim[0] for sim in sims if len(sim[0])==word_len and sim[1]>0.3][:20]
            sims = [sim for sim in sims if sim!=word]
            word_info['sim'] = sims
            word_info['sentiment'] = analyzeSentiment(word)
        word2info[word] = word_info
    
    predicts = word2vec_model.wv.most_similar(positive=[word for word in words if word in word2vec_model.wv][-10:], topn=10)
    return jsonHttp({
        'words': word2info,
        'predict': [elm[0] for elm in predicts],
    })

def search(request):
    content = request.GET.get('content')
    key_words = content.split(' ')
