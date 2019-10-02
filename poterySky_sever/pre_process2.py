import json
import sqlite3
import os
from gensim.models import AuthorTopicModel
from gensim import corpora, models, similarities
import numpy as np
from sklearn.cluster import KMeans
from data_process.config import config
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from data_process.commonFunction import segWithFilter, segAllWithFilter, toJson, loadJson, segCutAllWithFilter

db_path = './data_process/data/db/potery_sky.db'
def calPoteryCat():
    model = Doc2Vec.load('./data_process/model/potery2vec/128d/potery2vec_128d.model')
    conn = sqlite3.connect(db_path)
    db = conn.cursor()

    rows = db.execute('SELECT id FROM potery')
    ids = [row[0] for row in rows]
    vecs = [model.docvecs[id.replace('p_', '')] for id in ids]

    print('开始计算')
    id2label = {}
    label2id = {}
    labels = KMeans(n_clusters=100, max_iter=100, n_jobs=-1).fit_predict(vecs)
    for index, label in enumerate(labels):
        label = str(label)
        id = ids[index]
        if label not in label2id:
            label2id[label] = []
        label2id[label].append(id)
        id2label[id] = label
    
    print('开始计算第二层')
    for label in label2id:
        print(label)
        sub_ids = label2id[label]
        sub_vecs = [model.docvecs[id.replace('p_', '')] for id in sub_ids]
        cluster_num = 100 if len(sub_ids)>100 else len(sub_ids)
        sub_labels = KMeans(n_clusters=cluster_num, max_iter=1000, n_jobs=-1).fit_predict(sub_vecs)
        for index, label in enumerate(sub_labels):
            label = str(label)
            id = sub_ids[index]

            id2label[id] += '-' + label 

    for id in id2label:
        label = id2label[id]
        db.execute('UPDATE potery SET cat=? WHERE id=?', (label, id))
    conn.commit()
    conn.close()
# calPoteryCat()AQ

def calAuthorCat():
    model = AuthorTopicModel.load('./data_process/model/author2vec/author2vec.model')
    conn = sqlite3.connect(db_path)
    db = conn.cursor()

    rows = db.execute('SELECT id FROM author WHERE potery_num>=5')
    ids = set(model.id2author.values())

    ids = [row[0] for row in rows if row[0] in ids]
    print(len(ids))

    def getSpraceVec(id):
        topics = model.get_author_topics(id)
        vec = np.zeros(128)
        for topic in topics:
            vec[topic[0]] = topic[1]
        return vec
    vecs = [getSpraceVec(id) for id in ids]

    print('开始计算')
    id2label = {}
    label2id = {}
    labels = KMeans(n_clusters=50, max_iter=1000, n_jobs=-1).fit_predict(vecs)
    for index, label in enumerate(labels):
        label = str(label)
        id = ids[index]
        if label not in label2id:
            label2id[label] = []
        label2id[label].append(id)
        id2label[id] = label

    # print('开始计算第二层')
    # for label in label2id:
    #     print(label)
    #     sub_ids = label2id[label]
    #     sub_vecs = [getSpraceVec(id) for id in sub_ids]
    #     cluster_num = 10 if len(sub_ids)>10 else len(sub_ids)
    #     sub_labels = KMeans(n_clusters=cluster_num, max_iter=1000, n_jobs=-1).fit_predict(sub_vecs)
    #     for index, label in enumerate(sub_labels):
    #         label = str(label)
    #         id = sub_ids[index]

    #         id2label[id] += '-' + label 

    for id in id2label:
        label = id2label[id]
        db.execute('UPDATE author SET cat=? WHERE id=?', (label, id))
    conn.commit()
    conn.close()



def calPoteriesSimByTfIDF():
    print('calPoteriesSimByTfIDF')
    def seg(content):
        return segWithFilter(content)+segAllWithFilter(content)
    conn = sqlite3.connect(db_path)
    db = conn.cursor()

    rows = db.execute('SELECT id, poet_id, content FROM potery')
    rows = list(rows) #[:100]

    ids = [row[0] for row in rows]

    
    # print('开始改正')
    # temp_rows = db.execute('SELECT id, sim_potery_ids_tfidf FROM potery WHERE sim_potery_ids_tfidf IS NOT NULL')
    # temp_rows = list(temp_rows) #[:100]
    # for row in temp_rows:
    #     # print(row, row[1])
    #     sims = json.loads(row[1])
    #     sims= [ [ids[int(sim[0])], sim[1]] for sim in sims]
    #     db.execute('UPDATE potery SET sim_potery_ids_tfidf=? WHERE id=?', (toJson(sims), row[0]))
    # conn.commit()
    # print('改正完成')

    re_poets = [row[1] for row in rows]
    seg_poteries = [seg(row[2]) for row in rows]

    dictionary = corpora.Dictionary(seg_poteries)  # 生成字典和向量语料
    corpus = [dictionary.doc2bow(seg_potery) for seg_potery in seg_poteries]

    tfidf_model = models.TfidfModel(corpus)
    model_path = './data_process/model/potery2vec/tfidf/tf-idf.model'
    tfidf_model.save(model_path)
    corpus_tfidf = tfidf_model[corpus]
    sim_path = './data_process/model/potery2vec/tfidf/Similarity-tfidf-index'
    similarity = similarities.Similarity(sim_path, corpus_tfidf , num_features=len(dictionary))
    # similarity = similarities.Similarity.load(mode_path)
    print('保存')

    rows = db.execute('SELECT id FROM potery WHERE sim_potery_ids_tfidf IS NOT NULL')
    rows = list(rows) #[:100]
    fin_ids = set([row[0] for row in rows])

    rows = db.execute('SELECT id FROM author WHERE potery_num>=10')
    should_pro_authors =  set([row[0] for row in rows])

    should_pro_ids = [id for index, id in enumerate(ids) if re_poets[index] in should_pro_authors and id not in fin_ids]
    should_pro_ids = set(should_pro_ids)
    similarity.num_best = 30
    for index, p_id in enumerate(ids):
        if p_id not in should_pro_ids:
            continue
        if index%1000==0:
            print(index, len(should_pro_ids))
            conn.commit()
        doc_bow =  corpus[index] #dictionary.doc2bow(seg_poteries[index])
        vec_tfidf = tfidf_model[doc_bow]
        sims = similarity[vec_tfidf]
        sims.sort(key=lambda item: -item[1])
        # print(sims)
        sims = [[ids[int(sim[0])], float(sim[1])] for sim in sims]
        db.execute('UPDATE potery SET sim_potery_ids_tfidf = ? WHERE id=?',(toJson(sims), p_id))
    conn.commit()
    conn.close()
# calPoteriesSimByTfIDF()


def calAuthorSimByTFIDFSelf():
    print('calAuthorSimBySelf')
    def seg(content):
        return segWithFilter(content)+segAllWithFilter(content)
    conn = sqlite3.connect(db_path)
    db = conn.cursor()

    rows = db.execute('SELECT id FROM author WHERE potery_num>=10')
    should_pro_authors =  set([row[0] for row in rows])

    rows = db.execute('SELECT poet_id, content FROM potery')
    rows = list(rows)  #[:100]

    author2content = {}
    for poet_id, content in rows:
        # if poet_id in should_pro_authors:
        if poet_id not in author2content:
            author2content[poet_id] = ''
        author2content[poet_id] += content
    poet_ids = [poet_id for poet_id in author2content]
    seg_poteries  =[seg(author2content[poet_id]) for poet_id in author2content]

    dictionary = corpora.Dictionary(seg_poteries)  # 生成字典和向量语料
    corpus = [dictionary.doc2bow(seg_potery) for seg_potery in seg_poteries]

    print('开始训练')
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    model_path = './data_process/model/author2vec/tfidf/tf-idf.model'
    tfidf_model.save(model_path)
    sim_path = './data_process/model/author2vec/tfidf/Similarity-tfidf-index'
    similarity = similarities.Similarity(sim_path, corpus_tfidf , num_features=len(dictionary))
    similarity.num_best = 30
    print('训练结束')

    for index, poet_id in enumerate(poet_ids):
        if poet_id not in should_pro_authors:
            continue
        if index%1000==0:
            print(index, len(should_pro_authors))
            conn.commit()
        doc_bow = corpus[index] #dictionary.doc2bow(seg_poteries[index])
        vec_tfidf = tfidf_model[doc_bow]
        sims = similarity[vec_tfidf]
        sims.sort(key=lambda item: -item[1])
        # print(sims)
        sims = [[poet_ids[int(sim[0])], float(sim[1])] for sim in sims]
        db.execute('UPDATE author SET sims_tfidf_self = ? WHERE id=?',(toJson(sims), poet_id))
    conn.commit()
    conn.close()

# calAuthorSimByTFIDFSelf()

def calAuthorSimByPotery():
    print('calAuthorSimByPotery')
    conn = sqlite3.connect(db_path)
    db = conn.cursor()

    rows = db.execute('SELECT id, poet_id FROM potery')
    potery2author = {row[0]:row[1] for row in rows}

    author2sim_num = {}
    rows = db.execute('SELECT poet_id, sim_potery_ids_tfidf FROM potery WHERE sim_potery_ids_tfidf IS NOT NULL')
    rows = list(rows) #[:100]
    for row in rows:
        id = row[0]
        sims = json.loads(row[1])
        for sim_id, sim in sims:
            sim_id = potery2author[sim_id]
            if id not in author2sim_num:
                author2sim_num[id] = {}
            if sim_id not in author2sim_num:
                author2sim_num[sim_id] = {}
            if sim_id not in author2sim_num[id]:
                author2sim_num[id][sim_id] = 0
            if id not in author2sim_num[sim_id]:
                author2sim_num[sim_id][id] = 0
            author2sim_num[id][sim_id] += 1
            author2sim_num[sim_id][id] += 1
    for a_id in author2sim_num:
        sims = author2sim_num[a_id]
        sims = [[sim_id, sims[sim_id]]  for sim_id in sims] 
        sims.sort(key=lambda item: -item[1])
        sims = sims[0:30]
        db.execute('UPDATE author SET sims_tfidf_potery=? WHERE id=?', (toJson(sims), a_id))
    conn.commit()
    conn.close()
# calAuthorSimByPotery()

def calSentenceTfIdfModel():
    print('calSentenceTfIdfModel')
    def seg(content):
        return segWithFilter(content)+segAllWithFilter(content)
    model_path = './data_process/model/sentence2vec/tfidf/tf-idf.model'
    sim_path = './data_process/model/sentence2vec/tfidf/Similarity-tfidf-index'
    dict_path = './data_process/model/sentence2vec/tfidf/dict'
    conn = sqlite3.connect(db_path)
    db = conn.cursor()

    rows = list(db.execute('SELECT id FROM sentence'))
    ids = [row[0] for row in rows]

    # seg_sentences = [seg(row[1]) for row in rows]
    # print('分词完成')
    # dictionary = corpora.Dictionary(seg_sentences)  # 生成字典和向量语料
    # dictionary.save(dict_path)
    # corpus = [dictionary.doc2bow(seg_sentence) for seg_sentence in seg_sentences]

    # tfidf_model = models.TfidfModel(corpus)
    # tfidf_model.save(model_path)
    # corpus_tfidf = tfidf_model[corpus]
    # similarity = similarities.Similarity(sim_path, corpus_tfidf , num_features=len(dictionary))
    # similarity.save()

    tfidf_model = models.TfidfModel.load(model_path)
    similarity = similarities.Similarity.load(sim_path)
    dictionary = corpora.Dictionary.load(dict_path)

    need_process_ids = list(db.execute('SELECT id, content FROM sentence WHERE sim IS NULL'))
    similarity.num_best = 6
    for index, elm in enumerate(need_process_ids):
        s_id, s_content = elm
        if index%10==0:
            print(index, len(need_process_ids))
            conn.commit()
        words = seg(s_content)
        if len(words)<5:
            sims = []
        else:
            doc_bow = dictionary.doc2bow(words)
            vec_tfidf = tfidf_model[doc_bow]
            sims = similarity[vec_tfidf]
            sims.sort(key=lambda item: -item[1])
            # print(sims)
            sims = [[ids[int(sim[0])], float(sim[1])] for sim in sims]
        db.execute('UPDATE sentence SET sim = ? WHERE id=?',(toJson(sims), s_id))
        
    print('保存')
    conn.close()
# calSentenceTfIdfModel()

# open('./data_process/model/sentence2vec/tfidf\\Similarity-tfidf-index.0','r', encoding='utf-8')
def saveWord2db():
    conn = sqlite3.connect(db_path)
    db = conn.cursor()

    word2count = loadJson('./data_process/data/word2count.json')
    for index, word in enumerate(word2count):
        if index%1000==0:
            conn.commit()
        sql_comment  = "INSERT INTO word VALUES (?,?,null,null, ?, null)"
        db.execute(sql_comment,(word, word2count[word], len(word)))
    conn.commit()
    
    word_dir_path = './data_process/data/dict'
    word2info = {}
    files = os.listdir(word_dir_path) #列出文件夹下所有的目录与文件
    for file in files:
        path = os.path.join(word_dir_path,file)
        if os.path.isfile(path) and '.json' in file:
            file = loadJson(path)
            for word in file:
                word2info[word] = file[word]
    
    for index, word in enumerate(word2info):
        if index%1000==0:
            conn.commit()
        info = word2info[word]
        sql_comment = 'UPDATE word SET info = ? WHERE word=?'
        db.execute(sql_comment, (toJson(info), word))

    common_words = open('./data_process/data/停用词表.csv', 'r', encoding='utf-8').read().strip('\n').split('\n')
    for word in common_words:
        # print(word)
        sql_comment = 'UPDATE word SET is_common = 1 WHERE word=?'
        db.execute(sql_comment, (word, ))

    conn.commit()
    conn.close()

# saveWord2db()