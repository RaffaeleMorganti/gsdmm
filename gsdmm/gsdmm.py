from warnings import warn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

_GSDMM_IMPORT_MISSING = {"WORDCLOUD": False}

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    _GSDMM_IMPORT_MISSING["WORDCLOUD"] = True


class GSDMM:
    def __init__(self, clust=10, alpha=0.1, beta=0.1, n_iters=30, seed=None, verbose=False, **kwargs):
        '''
        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param clust: int
            Upper bound on the number of possible clusters. Typically many fewer
        :param alpha: float between 0 and 1
            Alpha controls the probability that a document will be clustered in a currently
            empty one. When alpha is 0, no one will join an empty cluster.
        :param beta: float between 0 and 1
            Beta controls the documents' affinity for similar documents. A low beta means
            documents desire to be clustered with similar ones. A high beta means they are less
            concerned with affinity and are more influenced by the popularity of a cluster
        :param n_iters: int
            Training iterations. Usually low is enough
        :param seed: int
            Random seed for reproducibility purpose
        :param verbose: bool
            show training progress
        :param **kwargs: CountVectorizer() parameters
            see sklearn.feature_extraction.text.CountVectorizer
        '''
        self.__K = clust
        self.__alpha = alpha
        self.__beta = beta
        self.__n_iters = n_iters
        self.__seed = seed
        self.__rng = np.random.default_rng(seed)
        self.__verbose = verbose
        self.__vectorizer = CountVectorizer(**kwargs)

        # slots for computed variables
        self.__tokenizer = None
        self.__number_docs = None
        self.__vocab = None
        self.__voc_len = None
        self.__clust_doc_count = None
        self.__clust_word_count = None
        self.__vocab_indexer = None
        self.__train_matrix = None

    def get_params(self):
        '''
        Get setup parameters
        '''
        return {
            'clust': self.__K,
            'alpha': self.__alpha,
            'beta':  self.__beta,
            'iters': self.__n_iters,
            'seed':  self.__seed,
            'token': self.__vectorizer.get_params()
        }

    def __vocab_index(self, token, add=False):
        '''
        return token position in vocab [store in dict and do in O(1)]
        '''
        if token not in self.__vocab_indexer:
            if not add:
                return -1
            self.__vocab_indexer[token] = int(np.where(self.__vocab == token)[0])
        return self.__vocab_indexer[token]

    def __tokenize(self, doc, add=False):
        '''
        tokenize document using CountVectorizer and return index position
        '''
        doc = np.array(self.__tokenizer(doc))
        return np.array([self.__vocab_index(t, add) for t in doc[np.isin(doc, self.__vocab)]])

    def __clust_add(self, k, d, s):
        self.__clust_doc_count[k] += 1
        self.__clust_word_count[k] += s
        self.__train_matrix[k, :] += d

    def __clust_del(self, k, d, s):
        self.__clust_doc_count[k] -= 1
        self.__clust_word_count[k] -= s
        self.__train_matrix[k, :] -= d

    def fit(self, docs):
        '''
        Cluster the input documents and return fitted clusters
        :param docs: list of documents
        :return: list of length len(doc)
            cluster label for each document
        '''

        # setup variables
        self.__clust_doc_count = np.zeros(self.__K, int)
        self.__clust_word_count = np.zeros(self.__K, int)

        # setup tokenizer
        td_matrix = self.__vectorizer.fit_transform(docs).toarray()
        self.__tokenizer = self.__vectorizer.build_analyzer()
        self.__vocab = np.array(self.__vectorizer.get_feature_names())
        self.__voc_len = len(self.__vocab)
        self.__train_matrix = np.zeros((self.__K, self.__voc_len))
        self.__vocab_indexer = {}

        self.__number_docs = len(docs)
        doc_t_count = td_matrix.sum(1)
        doc_t_list = [0] * self.__number_docs

        # choose a random  initial cluster for the docs
        dz = self.__rng.integers(self.__K, size=self.__number_docs)
        # initialize the clusters

        for k in range(self.__number_docs):
            doc_t_list[k] = self.__tokenize(docs[k], True)
            self.__clust_add(dz[k], td_matrix[k, :], doc_t_count[k])

        if self.__verbose:
            print("Model initialized")
        for i in range(self.__n_iters):
            trans = 0
            for k in range(self.__number_docs):
                # remove the doc from it's current cluster
                z_old = dz[k]
                self.__clust_del(z_old, td_matrix[k, :], doc_t_count[k])

                # draw sample from distribution to find new cluster
                z_new = self.__rng.multinomial(1, self.__score(doc_t_count[k], doc_t_list[k])).argmax()

                dz[k] = z_new
                self.__clust_add(z_new, td_matrix[k, :], doc_t_count[k])

                # transfer doc to the new cluster
                if z_new != z_old:
                    trans += 1

            if self.__verbose:
                print("Step %d: moved %d documents (%d clusters populated)" %
                      (i + 1, trans, sum(self.__clust_doc_count > 0)))

        return dz

    def __score(self, ntok, pos):
        '''
        Score a document (Implements formula (3) of Yin and Wang 2014)
        :param ntok: int: number of tokens
        :param pos: list[int]: index of tokens in vocab
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''

        size_range = np.arange(ntok).reshape((ntok, 1))

        # We break the formula into the following pieces
        # p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        # lN1 = log(mz[z] + alpha)
        # lN2 = log(D - 1 + K*alpha)
        # lN2 = log(product(nzw[w] + beta)) = sum(log(nzw[w] + beta))
        # lD2 = log(product(nz[d] + V*beta + i -1)) = sum(log(nz[d] + V*beta + i -1))
        lN1 = np.log(self.__clust_doc_count + self.__alpha)
        lD1 = np.log(self.__number_docs - 1 + self.__K * self.__alpha)
        lD2 = np.log(self.__clust_word_count + self.__voc_len * self.__beta + size_range).sum(0)
        if len(pos) != 0:
            lN2 = np.log(self.__train_matrix[:, pos] + self.__beta).sum(1)
        else:
            lN2 = np.zeros((self.__K,))

        p = np.exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        return p / max(sum(p), 1e-300)

    def predict_proba(self, docs):
        '''
        Score documents (Implements formula (3) of Yin and Wang 2014)
        :param docs: list[str]: list of documents
        :return: list[list[float]]:
            A list with length n docs, each as K probability vector where each component
            represents the probability of the document appearing in a particular cluster
        '''
        doc_t_count = self.__vectorizer.transform(docs).toarray().sum(1)
        doc_t_list = [self.__tokenize(docs[k]) for k in range(len(docs))]
        return np.array([self.__score(doc_t_count[k], doc_t_list[k]) for k in range(len(docs))])

    def predict(self, docs):
        '''
        Choose the highest probability label for the input document
        :param docs: list[str]: list of documents
        :return: list of best cluster
        '''
        return self.predict_proba(docs).argmax(1)

    def get_importances(self):
        '''
        Word importance for each cluster (Implements formula (9) of Yin and Wang 2014)
        :return: list of dict
          for each cluster dict: word importances
        '''
        n = self.__train_matrix + self.__beta
        n[n == self.__beta] = 0
        d = self.__train_matrix.sum(1) + self.__voc_len * self.__beta
        r = n / d.reshape((self.__K, 1))

        phi = np.array([{} for _ in range(self.__K)])
        for z in range(self.__K):
            pos = r[z, :] > 0
            phi[z] = dict(sorted(zip(self.__vocab[pos], r[z, pos]), key=lambda x: x[1], reverse=True))
        return phi

    def get_avg_importances(self):
        '''
        NOTE: use importances to get importances as by Yin and Wang 2014
        Relative word importance for each cluster
        avg_imp = imp[clust==k] - avg(imp[clust!=k])
        :return: list of dict
          for each cluster dict: word importances
        '''
        n = self.__train_matrix + self.__beta
        n[n == self.__beta] = 0
        d = self.__train_matrix.sum(1) + self.__voc_len * self.__beta
        o = n / d.reshape((self.__K, 1))

        o += (o - o.sum(0)) / (self.__K - 1)

        phi = np.array([{} for _ in range(self.__K)])
        for z in range(self.__K):
            pos = o[z, :] > 0
            phi[z] = dict(sorted(zip(self.__vocab[pos], o[z, :][pos]), key=lambda x: x[1], reverse=True))
        return phi

    def get_clust_info(self):
        '''
        Trained clusters sizes information
        :return: list of tuple
          for each cluster (int,int):
            docs into cluster ratio
            words into cluster ratio
        '''
        docs = self.__clust_doc_count / self.__number_docs
        words = self.__clust_word_count / sum(self.__clust_word_count)

        return np.c_[docs, words]

    def get_wordclouds(self, imp, ncol=3, cloud={"background_color": "white"}, plot={}):
        '''
        Return matplotlib figure with wordclouds for each cluster
        :param imp:
            list returned by get_importances() or get_avg_importances()
        :param ncol: (default 3)
            number of clouds in one line
        :param cloud: (optional)
            dict of WordCloud args (see wordcloud.WordCloud)
        :param plot: (optional)
            dict of figure args (see matplotlib.pyplot.figure)
        :return: matplotlib.pyplot.figure
        '''
        if _GSDMM_IMPORT_MISSING["WORDCLOUD"]:
            warn("\nNOTE: The gsdmm.wordcloud() method requires extra libraries to be installed."
                 "\nYou must install wordcloud, matplotlib (you can use 'pip install gsdmm[plot]')")
            return None

        wc = WordCloud(**cloud)
        fig, axs = plt.subplots(int(np.ceil(self.__K / ncol)), ncol, **plot)
        axs = np.array(axs).reshape(-1)
        [ax.axis("off") for ax in axs]

        for i in range(self.__K):
            axs[i].set_title("Cluster #%d" % i)
            if len(imp[i]) > 0:
                axs[i].imshow(wc.generate_from_frequencies(imp[i]), interpolation='bilinear')

        fig.suptitle("Word importances")
        fig.tight_layout()
        return fig
