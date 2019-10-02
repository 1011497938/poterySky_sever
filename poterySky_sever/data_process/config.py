
class Config:
    def __init__(self):
        self.root_path = 'poterySky_sever/data_process/'
        self.data_path = self.root_path + 'data/'
        self.model_path = self.root_path + 'model/'

        self.potery_model128_path = self.model_path + 'potery2vec/128d/potery2vec_128d.model'
        self.potery_model3_path = self.model_path + 'potery2vec/3d/potery2vec_3d.model'
        self.potery_path = self.data_path + 'poem/'
        self.potery200000_path = self.data_path + 'poem200000/'
        self.potery_rank_path = self.data_path + 'potery_rank.json'
        self.potery2vec_path = self.data_path + 'potery2vec/7_30.json'
        self.potery2sim_path = self.data_path + 'pid2sim.json'

        self.sentence_model128_path = self.model_path + 'sentence2vec/128d/sentence2vec_128d.model'
        self.author_model128_path = self.model_path + 'author2vec/author2vec.model'
        self.author_simMatrix_path = self.model_path + 'author2vec/sim.index'
        self.sentence_tfidf_model = self.model_path + 'sentence2vec/tfidf/tf-idf.model'
        self.sentence_tfidf_sim_index = self.model_path + 'sentence2vec/tfidf/Similarity-tfidf-index'
        self.sentence_tfidf_dict_path = self.model_path + 'sentence2vec/tfidf/dict'

        self.word_model_path =self.model_path + 'word2vec/128d/word2vec.model'
        self.word_model_path_r = './data_process/model/word2vec/128d/word2vec.model'

        self.db_path = self.data_path + 'db/potery_sky.db'
        self.new_word_path = self.data_path + 'new_words.csv'
        self.new_word_path_r = './data_process/data/new_words.csv'

        self.temp_data = self.root_path + 'pre_process/temp_data/'

        self.author_relations_path = self.data_path + '诗人关系.csv'
config = Config()