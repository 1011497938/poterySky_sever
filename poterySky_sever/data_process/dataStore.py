import json
import os
import copy
from .commonFunction import writeJson, loadJson
from .config import config
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

p_id2rank = loadJson(config.potery_rank_path)

class Author:
    def __init__(self, author_name):
        self.name = author_name

class Sentence:
    def  __init__(self, json_object, potery, index):
        if 'Content' in json_object:
            self.content = json_object['Content']
        else:
            self.content = ""

        if 'Comments' in json_object:
            self.comments = json_object['Comments']
        else: 
            self.comments = ''

        self.potery = potery
        self.index = index
        # BreakAfter 是什么

    def getSimpDict(self):
        return {
            'content': self.content,
            'comments': self.comments,
        }

    def getSimSentenceWithSim(self, num = 20):
        sims = sentenceManager.s_model128.docvecs.most_similar([self.identchars()], topn=num)
        return [ ['p_'+sim[0], sim[1]] for sim in sims]

class Potery:
    def __init__(self, json_object):
        self.id = json_object["id"]
        shidata = json_object["ShiData"][0]

        if 'Author' in shidata:
            self.author = shidata['Author']
            if self.author == '无名氏':
                self.author = ''
        else:
            self.author = ''
        self.author = authorManager.createAuthor(self.author)
        self.sentences = [sentenceManager.createSentence(item, self, index) for index, item in enumerate(shidata['Clauses'])]
        self.rank = -9999
        self.vec3 = [0,0,0]
        self.sims = []

    def getSimPoteries(self, num = 20):
        return [poteryManager.get(pid) for pid in self.sims][0: num]
        # sims = poteryManager.p_model128.docvecs.most_similar([self.getFormerId()], topn=num)
        # sims = sorted(sims, key=lambda sim: -sim[1])
        # return [poteryManager.get('p_'+sim[0]) for sim in sims]
    def getSimPoteriesWithSim(self, num=20):
        sims = poteryManager.p_model128.docvecs.most_similar([self.getFormerId()], topn=num)
        # sims = sorted(sims, key=lambda sim: -sim[1])
        return [ ['p_'+sim[0], sim[1]] for sim in sims]

    def getVec3(self):
        # temp_id = self.getFormerId()
        # return poteryManager.p_model3.docvecs[temp_id]
        return self.vec3

    def getFormerId(self):
        return self.id.replace('p_', '') 

    def getVec128(self):
        temp_id = self.getFormerId()
        return poteryManager.p_model128.docvecs[temp_id]

    def getRank(self):
        return self.rank

    def getSimpDict(self):
        return {
            'id': self.id,
            'author': self.author.name,
            'sentence': [sentence.content  for sentence in self.sentences]
        }

    def getContents(self):
        return ''.join([sentence.content for sentence in self.sentences])

class SentenceManager: 
    def __init__(self):
        self.sentences = set()
        self.auto_id = 0

    def createSentence(self, json_object, potery, index):
        new_sentence = Sentence(json_object, potery, index)
        new_sentence.id = 's_' + str(self.auto_id)
        self.auto_id += 1
        self.sentences.add(new_sentence)
        return new_sentence

    def loadModel(self):
        self.s_model128 = Doc2Vec.load(config.sentence_model128_path)

class AuthorManager:
    def __init__(self):
        self.name2author = {}
        self.authors = set()
    def createAuthor(self, name):
        if name in self.name2author:
            return self.name2author[name]
        else:
            new_author = Author(name)
            self.name2author[name] = new_author
            self.authors.add(new_author)
            return new_author

class PoteryManager:
    def __init__(self):
        self.id2potery = {}
        self.poteries = []
        # self.loads()

    def getAllPoteries(self):
        return list(self.poteries)

    def getAllIds(self):
        return self.id2potery.keys()

    def getImpPotery(self, num):
        return self.poteries[0:num]

    def loads(self):
        files = os.listdir(config.potery_path) #列出文件夹下所有的目录与文件
        # files = os.listdir(config.potery200000_path) #列出文件夹下所有的目录与文件
        for file in files:
            path = os.path.join(config.potery_path,file)
            if os.path.isfile(path):
                file = loadJson(path)
                for potery_id in file:
                    file[potery_id]['id'] =  'p_' + potery_id
                    self.createPotery(file[potery_id])
        # self.loadPoteryRank()
        # self.loadPoteryVec()
        # self.loadPoteryModel()
        # self.loadPotery2Sim()
        print('数据加载完成')

    def loadPoteryRank(self):
        potery_rank = loadJson(config.potery_rank_path)
        for pid in potery_rank:
            potery = self.get('p_' + pid) #暂时先这么写着
            potery.rank = potery_rank[pid]
        self.poteries = sorted(self.poteries, key=lambda potery: -potery.rank)
        # print([potery.rank for potery in self.poteries])
        
    def loadPoteryVec(self):
        pid2vec = loadJson(config.potery2vec_path)
        for pid in pid2vec:
            potery = self.get('p_' + pid) #暂时先这么写着
            potery.vec3 = pid2vec[pid]

        # p_model3 = Doc2Vec.load(config.potery_model3_path)
        # for potery in self.poteries:
        #     temp_id = potery.getFormerId()
        #     potery.vec3 =  p_model3.docvecs[temp_id].tolist()

    def loadPotery2Sim(self):
        pid2sim = loadJson(config.potery2sim_path)
        for pid in pid2sim:
            potery = self.get(pid) #暂时先这么写着
            potery.sims = pid2sim[pid]

    def loadPoteryModel(self):
        self.p_model128 = Doc2Vec.load(config.potery_model128_path)

    def get(self, pid):
        if pid not in self.id2potery:
            # print()
            return None
            # print(self.poteries, len(self.poteries))
            # return self.poteries[0]
        else:
            return self.id2potery[pid]

    def createPotery(self, json_object): 
        new_potery = Potery(json_object)
        self.id2potery[json_object['id']] = new_potery
        self.poteries.append(new_potery)
        return new_potery

    def getId2simpPotery(self):
        return {potery.id: potery.getSimpDict()  for potery in self.poteries}

    def getId2PoteryName(self):
        return {potery.id: potery.sentences[0].content for potery in self.poteries}

    def getId2vec3(self):
        return {potery.id: potery.vec3 for potery in self.poteries}

authorManager = AuthorManager()
sentenceManager = SentenceManager()
poteryManager = PoteryManager()


