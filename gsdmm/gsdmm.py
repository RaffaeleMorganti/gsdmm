import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    print("""NOTE: The gsdmm.wordcloud() method requires extra libraries to be installed: wordcloud.
    You can install these using pip install gsdmm[plot]. All other methods are still available.""")


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
        self.__cluster_doc_count = None
        self.__cluster_word_count = None
        self.__cluster_word_distribution = None

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

    def __tokenize(self, doc):
        '''
        tokenize document using CountVectorizer
        '''
        doc = np.array(self.__tokenizer(doc))
        return doc[np.isin(doc, self.__vocab)]

    def fit(self, docs):
        '''
        Cluster the input documents and return fitted clusters
        :param docs: list of documents
        :return: list of length len(doc)
            cluster label for each document
        '''

        # setup variables
        self.__cluster_doc_count = np.zeros(self.__K, int)
        self.__cluster_word_count = np.zeros(self.__K, int)
        self.__cluster_word_distribution = np.array([{} for _ in range(self.__K)])

        # unpack to easy var names
        mz, nz, nzw = self.__cluster_doc_count, self.__cluster_word_count, self.__cluster_word_distribution

        # setup tokenizer
        self.__vectorizer.fit(docs)
        self.__tokenizer = self.__vectorizer.build_analyzer()
        self.__vocab = self.__vectorizer.get_feature_names()

        docs = [self.__tokenize(doc) for doc in docs]

        self.__number_docs = len(docs)

        # initialize the clusters
        # choose a random  initial cluster for the docs
        dz = self.__rng.integers(self.__K, size=self.__number_docs)

        for i, doc in enumerate(docs):
            z = dz[i]
            mz[z] += 1
            nz[z] += len(doc)

            for word in doc:
                if word not in nzw[z]:
                    nzw[z][word] = 0
                nzw[z][word] += 1

        if self.__verbose:
            print("Model initialized")

        for _iter in range(self.__n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # remove the doc from it's current cluster
                z_old = dz[i]

                mz[z_old] -= 1
                nz[z_old] -= len(doc)

                for word in doc:
                    nzw[z_old][word] -= 1

                    # compact dictionary to save space
                    if nzw[z_old][word] == 0:
                        del nzw[z_old][word]

                # draw sample from distribution to find new cluster
                z_new = self.__rng.multinomial(1, self.__score(doc)).argmax()

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                dz[i] = z_new
                mz[z_new] += 1
                nz[z_new] += len(doc)

                for word in doc:
                    if word not in nzw[z_new]:
                        nzw[z_new][word] = 0
                    nzw[z_new][word] += 1

            if self.__verbose:
                print("Stage %d: transferred %d documents (%d clusters populated)" %
                      (_iter + 1, total_transfers, sum(mz > 0)))

        self.__cluster_word_distribution = nzw
        return dz

    def __score(self, doc):
        '''
        Score a document (Implements formula (3) of Yin and Wang 2014)
        :param docs: list[str]: list of tokens
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''
        mz, nz, nzw = self.__cluster_doc_count, self.__cluster_word_count, self.__cluster_word_distribution

        p = np.zeros(self.__K)
        size_range = np.arange(len(doc))
        v = len(self.__vocab)

        # We break the formula into the following pieces
        # p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        # lN1 = log(mz[z] + alpha)
        # lN2 = log(D - 1 + K*alpha)
        # lN2 = log(product(nzw[w] + beta)) = sum(log(nzw[w] + beta))
        # lD2 = log(product(nz[d] + V*beta + i -1)) = sum(log(nz[d] + V*beta + i -1))

        lN1 = np.log(mz + self.__alpha)
        lD1 = np.log(self.__number_docs - 1 + self.__K * self.__alpha)
        lN2 = np.zeros(self.__K)
        lD2 = np.zeros(self.__K)

        for i in range(self.__K):
            lD2[i] = np.log(nz[i] + v * self.__beta + size_range).sum()
            lN2[i] = np.sum([np.log(nzw[i].get(word, 0) + self.__beta) for word in doc])

        p = np.exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        return np.nan_to_num(p / sum(p))

    def predict_proba(self, docs):
        '''
        Score documents (Implements formula (3) of Yin and Wang 2014)
        :param docs: list[str]: list of documents
        :return: list[list[float]]:
            A list with length n docs, each as K probability vector where each component
            represents the probability of the document appearing in a particular cluster
        '''
        docs = [self.__tokenize(doc) for doc in docs]
        return np.array([self.__score(doc) for doc in docs])

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
        nzw = self.__cluster_word_distribution
        phi = np.array([{} for _ in range(self.__K)])

        v = len(self.__vocab)
        for z in range(self.__K):
            d = sum(nzw[z].values()) + v * self.__beta
            n = np.fromiter(nzw[z].values(), float) + self.__beta
            phi[z] = dict(sorted(zip(nzw[z].keys(), n / d), key=lambda x: x[1], reverse=True))

        return phi

    def get_avg_importances(self):
        '''
        NOTE: use importances to get importances as by Yin and Wang 2014
        Relative word importance for each cluster
        avg_imp = imp[clust==k] - avg(imp[clust!=k])
        :return: list of dict
          for each cluster dict: word importances
        '''
        s = np.array(self.__vocab)
        v = len(self.__vocab)
        r = np.array([{} for _ in range(self.__K)])

        imp = self.get_importances()

        o = np.zeros((v, self.__K))
        for i in range(self.__K):
            o[[self.__vocab.index(k) for k in imp[i].keys()], i] = list(imp[i].values())

        m = o.sum(1).reshape((v, 1))

        o += (o - m) / (self.__K - 1)

        for i in range(self.__K):
            pos = o[:, i] > 0
            r[i] = dict(sorted(zip(np.array(s)[pos], o[:, i][pos]), key=lambda x: x[1], reverse=True))

        return r

    def get_clust_info(self):
        '''
        Trained clusters sizes information
        :return: list of tuple
          for each cluster (int,int):
            docs into cluster ratio
            words into cluster ratio
        '''
        docs = self.__cluster_doc_count / self.__number_docs
        words = self.__cluster_word_count / sum(self.__cluster_word_count)

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
