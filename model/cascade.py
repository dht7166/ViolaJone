from ViolaJone import ViolaJone
import pickle



class Cascade:
    def __init__(self, feature_nums):
        self.feature_nums = feature_nums # numbers of features in each cascade layer
        self.str_clfs = [] # list to store classfier in each layer
        
    def train(self, X, Y):
        # lists of positive and negative examples
        P = []
        N = []
        
        for i in range(len(Y)):
            if Y[i] == 1:
                P.append([X[i], Y[i]])
            else:
                N.append([X[i], Y[i]])
                
        # for now we only focus on false positive rate
        # todo: incorporate detection rate
        for num_feature in self.feature_nums:
            if len(N) == 0:
                print('Number of false positive is 0. Stop training.')
                break
            
            # todo: currently num features can only initialized for AdaBoost not ViolaJone
            str_clf = ViolaJone(num_feature=num_feature)
            joined_data = P + N
            X = [data[0] for data in joined_data]
            Y = [data[1] for data in joined_data]
            str_clf.fit(X, Y)
            self.str_clfs.append(str_clf)

            # for subsequent layer, let N be the false positives
            false_P = []
            for ex in N:
                if self.classify(ex[0]) == 1:
                    false_P.append(ex)
            N = false_P

        def classify(self, X):
            for str_clf in self.str_clfs:
                if str_clf.classify(X) == 0:
                    return 0
            return 1