import json
from django.http import HttpResponse
from .data_process.dataStore import poteryManager, sentenceManager, authorManager
from .data_process.commonFunction import writeJson, loadJson, seg, writeCsv, toJson, segAllWithFilter, segWithFilter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from .data_process.config import config
from .data_process.dbManager import createDbConnect
import sqlite3
import  re
from multiprocessing import cpu_count
from gensim.models import AuthorTopicModel
from gensim.similarities import MatrixSimilarity
import networkx as nx 
import numpy as np
from sklearn.cluster import KMeans

def processData(request):
    return HttpResponse('success')

def writeAuthorsList():
    authors = authorManager.authors
    author_list = [[author.name] for author in authors]
    writeCsv('author_list', author_list)
    print('写入完成')

def sentence2vec():
    documents = []
    db = createDbConnect()
    sentences = db.execute('SELECT id, content from sentence')
    for sentence in sentences:
        # print(sentence[0], sentence[1])
        documents.append(TaggedDocument(segWithFilter(sentence[1]) + segAllWithFilter(sentence[1]), [sentence[0]]))
    
    print('开始训练')
    # model_path = config.sentence_model128_path
    # model = Doc2Vec(documents, vector_size=128, window=5, min_count=10, workers=cpu_count())
    # model.save(model_path)
    # print('保存成功')
    model = Doc2Vec.load(model_path)
    for index in range(20):
        print('训练', index, '次')
        model.train(documents, total_examples=len(documents), epochs=20)
        model.save(model_path)

def calAuthorSim():
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    model = AuthorTopicModel.load(config.author_model128_path)
    poets = list(model.id2author.values())
    print(len(poets))
    # vec = model.get_author_topics('苏轼')
    index = MatrixSimilarity(model[list(model.id2author.values())], num_best=30)
    index.save(config.author_simMatrix_path)
    # index = MatrixSimilarity.load(config.author_simMatrix_path)

    for name in poets:
        # print(name)
        sims = index[model[name]]
        sims = sorted(sims, key=lambda item: -item[1])
        sims = [ [poets[sim[0]] , sim[1]] for sim in sims]
        # print(sims)
        # sql_comment  = "UPDATE author SET sims=? WHERE id=?"
        # db.execute(sql_comment, (toJson(sims), name))

        sql_comment  = "UPDATE author SET sims=\'{}\' WHERE id=\'{}\'".format(toJson(sims), name)
        db.execute(sql_comment)
        # print(sql_comment)
    # print(len(poets))
    conn.commit()
# calAuthorSim()

def calcualteAuthorRank():
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    # rows = db.execute('SELECT id,poet_id,sim_potery_ids FROM potery')
    # potery2poet = {}

    # G = nx.Graph()
    # rows = [row for row in rows]
    # for row in rows:
    #     potery_id = row[0]
    #     poet_id = row[1]
    #     potery2poet[potery_id] = poet_id

    # for row in rows:
    #     potery_id = row[0]
    #     poet_id = row[1]
    #     sim_potery_ids = json.loads(row[2])
    #     # print(len(sim_potery_ids))
    #     for sim in sim_potery_ids:
    #         sim_id = sim[0]
    #         sim = sim[1]
    #         G.add_weighted_edges_from([(poet_id, potery2poet[sim_id], sim)])
    # print('开始计算', len(G.nodes))
    # poet_rank = {}
    # pr=nx.pagerank(G, weight='weight', max_iter=1000)
    # ranks = np.array([pr[id] for id in pr])
    # min = np.min(ranks)
    # max = np.max(ranks)
    # for id in pr:
    #     rank = pr[id]
    #     rank = (rank-min)/(max-min)
    #     poet_rank[id] = rank
    # writeJson('poet_rank', poet_rank)

    poet_rank = loadJson(config.data_path+'poet_rank.json')
    count = 0
    rows = db.execute('SELECT id from sentence')
    for name in poet_rank:
        count += 1
        if count%10000==0:
            # print(count)
            conn.commit()
        sql_comment  = "UPDATE author SET rank=? WHERE id=?"
        db.execute(sql_comment, (poet_rank[name], name))
    conn.commit()
# calcualteAuthorRank()

# test()
# bug 没用
# def author2vec():
#     corpus = []
#     author2doc = {}

#     for index, potery in enumerate(poteryManager.poteries):
#         # print(index)
#         if index>100:
#             break
#         author = potery.author.name
#         content = segAllWithFilter(potery.getContents())
#         if author not in author2doc:
#             author2doc[author] = []
#         author2doc[author].append(index)
#         corpus.append(content)

#     model = AuthorTopicModel(corpus, author2doc=author2doc, num_topics=128, serialized=True, serialization_path=config.author_model128_path)

# author2vec()


# 计算诗词的数量
def calPoetPoteryIdsAndNum():
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    rows = db.execute('SELECT poet_id, id FROM potery')
    rows = [row for row in rows]
    poet2potery = {}
    for row in rows:
        poet_id = row[0]
        potery_id = row[1]
        if poet_id not in poet2potery:
            poet2potery[poet_id] = []
        poet2potery[poet_id].append(potery_id)
    for name in poet2potery:
        item = poet2potery[name]
        db.execute('UPDATE author SET potery_ids=?, potery_num=? WHERE id=?', (toJson(item), len(item), name))
    conn.commit()
# calPoetPoteryIdsAndNum()


def writeAllThings2db():
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    count = 0
    for potery in poteryManager.poteries:
        count += 1
        if count%10000==0:
            print('potery',count, len(poteryManager.poteries))
            conn.commit()
        vec3 = toJson(potery.vec3)
        id = potery.id
        poet_id = potery.author.name
        sentence_ids = toJson([sentence.id for sentence in potery.sentences])
        sim_potery_ids = toJson(potery.getSimPoteriesWithSim(30))
        rank = potery.rank
        sql_comment  = "INSERT INTO potery VALUES (?,?,?,?,?,?)"
        db.execute(sql_comment, (id, vec3, poet_id, sentence_ids, sim_potery_ids, rank))
    conn.commit()

    count = 0
    for sentence in sentenceManager.sentences:
        count += 1
        if count%10000==0:
            print('sentence',count, len(sentenceManager.sentences))
            conn.commit()

        id = sentence.id
        potery_id = sentence.potery.id
        index = sentence.index
        content = sentence.content

        comments = toJson(sentence.comments)
        re.escape(comments)
        words = toJson(seg(content))
        sql_comment  = "INSERT INTO sentence VALUES (?,?,?,?,?,?)"
        db.execute(sql_comment, (id, potery_id, index, content, comments, words))
    conn.commit()

    count = 0
    for author in authorManager.authors:
        count += 1
        if count%10000==0:
            print('author',count, len(authorManager.authors))
            conn.commit()

        sql_comment  = "INSERT INTO author VALUES (?, ?, null, null, null, null, null, null)"
        db.execute(sql_comment, (author.name, author.name))
    conn.commit()

    count = 0
    poet_info = loadJson(config.data_path+'poet_info.json')
    print(len(poet_info.keys()))
    def countCertainty(info):
        certainty = 0
        for key in info:
            elm = info[key]
            if elm is not None:
                certainty =+ 1
            if elm==9999 or elm==-9999:
                certainty -= 1
        return certainty

    for name in poet_info:
        count += 1
        if count%10000==0:
            print('author_info',count, len(authorManager.authors))
            conn.commit()
            
        info = poet_info[name][0]
        max_certainty = 0
        for elm in poet_info[name]:
            certainty = countCertainty(elm)
            if certainty>max_certainty:
                info = elm
                max_certainty = certainty

        gender = '男' if info['female']=='0' else '女'
        alt_name = toJson(info['alt_name'])
        birth = info['birth_year'] if info['birth_year']!=-9999 else None
        death = info['death_year'] if info['death_year']!=9999 else None
        time_range = info['time_range']
        start = time_range[0] if time_range[0]!=-9999 else None
        end = time_range[1] if time_range[1]!=9999 else None

        sql_comment  = "UPDATE author SET name=?, gender=?, info=?, birth=?, death=?, start=?, end=?, dynasty=?, alt_name=? WHERE id=?"
        db.execute(sql_comment, (info['name'], gender, toJson(poet_info[name]), birth, death, start, end, info['dy'], alt_name, info['name']))
    conn.commit()

    count = 0
    # 这里似乎有错
    model128 = Doc2Vec.load(config.sentence_model128_path)
    rows = db.execute('SELECT id from sentence')
    # print(len([row for row in rows]))
    for row in rows:
        count += 1
        if count%10==0:
            print('sim_info',count)
            conn.commit()
        id = row[0]
        sims = model128.docvecs.most_similar([id], topn=30)
        sims = [ [sim[0], sim[1]] for sim in sims]
        sql_comment  = "UPDATE sentence SET sim=? WHERE id=?"
        sql_comment = "UPDATE sentence SET sim=\'{}\' WHERE id=\'{}\'".format(toJson(sims), id)
        print(sql_comment)
        # db.execute(sql_comment, (toJson(sims), id))  #这里会崩
        db.execute(sql_comment)

    conn.commit()
    conn.close()
    return
# writeAllThings2db()

def calPoteryContent():
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    count = 0
    for potery in poteryManager.poteries:
        count+=1
        if count%1000==0:
            print(count)
            conn.commit()
        id = potery.id
        content = potery.getContents()
        db.execute('UPDATE potery SET content=? WHERE id=?', (content, id))
    conn.commit()
    conn.close()
    # rows = db.execute

def calPoteryCat():
    model = Doc2Vec.load(config.potery_model128_path)
    conn = sqlite3.connect(config.db_path)
    db = conn.cursor()

    rows = db.execute('SELECT id FROM potery')
    ids = [row[0] for row in rows][0:8000]
    vecs = [model.docvecs[id.replace('p_', '')] for id in ids]

    id2label = {}
    label2id = {}
    labels = KMeans(n_clusters=1000, max_iter=2000, n_jobs=1).fit_predict(vecs)
    for index, label in enumerate(labels):
        label = str(label)
        id = ids[index]
        if label not in label2id:
            label2id[label] = []
        label2id[label].append(id)
        id2label[id] = label
    
    for label in label2id:
        print(label)
        sub_ids = label2id[label]
        sub_vecs = [model.docvecs[id.replace('p_', '')] for id in sub_ids]
        cluster_num = 80 if len(sub_ids)>80 else len(sub_ids)
        sub_labels = KMeans(n_clusters=cluster_num, max_iter=2000, n_jobs=1).fit_predict(sub_vecs)
        for index, label in enumerate(sub_labels):
            label = str(label)
            id = sub_ids[index]

            id2label[id] += '-' + label 

    for id in id2label:
        label = id2label[id]
        db.execute('UPDATE potery SET cat=? WHERE id=?', (label, id))
    conn.commit()
    conn.close()
calPoteryCat()
# calPoteryContent()

