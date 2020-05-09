import numpy as np
import cv2
from utils import *
import glob
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold, SelectKBest
import pickle

class AdaBoost:
    def __init__(self,num_feature = None):
        self.T = num_feature
        self.weak_clf = []

    def weak_learner(self,X,Y,W):
        """
        :param X: The computed values for some features
        :param Y: The true label
        :param W: Weight
        :return: Error, parity and threshold (E,P,and theta)
        """
        # X is a vector of images for some feature
        # This function is to train the weak classifier
        # By learning the threshold and parity (which is for the sign)
        E = 999999999
        P = 1
        thres = None
        # Start by sorting
        ordering = np.argsort(X)
        X = X[ordering]
        Y = Y[ordering]
        W = W[ordering]
        # Calculate Tp and Tn
        Tp = 0
        Tn = 0
        for i in range(X.shape[0]):
            if Y[i]==0:
                Tn+=W[i]
            else:
                Tp+=W[i]

        # One pass for calculating Sp and Sn
        Sp = 0
        Sn = 0
        for i in range(X.shape[0]):
            below_as_neg = Sp + (Tn-Sn)
            below_as_pos = Sn + (Tp-Sp)
            if E>below_as_pos:
                E = below_as_pos
                P = 1
                thres = X[i]
            if E > below_as_neg:
                E = below_as_neg
                P = -1
                thres = X[i]
            if Y[i] == 0:
                Sn += W[i]
            else:
                Sp += W[i]

        return E,P,thres

    def fit(self,X,Y,haar_feature):
        """
        :param X: the images x feature matrix
        :param Y: the true label
        :return: None, all the weak classifier will be in self.chosen
        """

        # Initialize W
        num_pos = np.count_nonzero(Y==1)
        num_neg = np.count_nonzero(Y==0)
        W = np.zeros(Y.shape)
        for i in range(Y.shape[0]):
            if Y[i] ==0:
                W[i] = 1/(2*num_neg)
            else:
                W[i] = 1/(2*num_pos)

        for num_classifier in range(self.T): # Choose only T weak classifier
            print("CHOOSING THE {}/{} WEAK CLASSIFIER".format(num_classifier+1,self.T),flush=True)
            # normalize weight
            W = W/np.sum(W)

            # select the best weak classifier
            best_E = 999999999
            best_weak_classifier = None
            for i in range(len(haar_feature)):
                E,P,thres = self.weak_learner(X[:,i],Y,W)
                if best_E>E:
                    best_E = E
                    best_weak_classifier = (i,haar_feature[i],P,thres)

            # Now we have best_weak_classifier
            Beta = best_E/(1-best_E)
            alpha = np.log(1/Beta)
            i,feature, P, thres = best_weak_classifier
            self.weak_clf.append((alpha,feature,P,thres))
            print("CHOSE WEAK CLASSIFIER {} with best error {:.3f} and weight {:.3f}".format(i,best_E,alpha))
            print("ADJUSTING WEIGHT")
            # now we adjust the W
            for i in range(Y.shape[0]):
                j,feature,P,thres = best_weak_classifier
                pred = 1 if P*X[i,j]<P*thres else 0
                e = 0 if pred == Y[i] else 1
                W[i] = W[i]*Beta**(1-e)


    def save(self,name):
        with open(name,'wb') as f:
            pickle.dump(self.weak_clf,f)

    def load(self,name):
        with open(name,'rb') as f:
            self.weak_clf = pickle.load(f)
        self.T = len(self.weak_clf)

    def predict(self,X):
        # of course X is a integral image
        sum_pred = 0
        baseline = 0
        for alpha,feature,P,thres in self.weak_clf:
            sum_pred+=alpha*(1 if P*compute_feature_using_integral(X,feature)<P*thres else 0)
            baseline+=alpha/2
        return 1 if sum_pred>baseline else 0


class ViolaJone:
    def __init__(self,size):
        self.size = size
        self.create_haar_feature()
        self.layer = [] # list of layers, each is a AdaBoosted classifier

    def create_haar_feature(self):
        # create two rectangle feature
        self.feature = []
        list_coord = [(x,y) for x in range(self.size) for y in range(self.size)]
        for x,y in list_coord:
            for h in range(1,self.size):
                for w in range(1,self.size):
                    if x+h >=self.size or y+w>=self.size:
                        continue
                    current = ((x,y),(x+h,y+w))
                    if x+2*h<self.size:
                        down = ((x+h,y),(x+2*h,y+w))
                        self.feature.append(([current],[down]))
                        self.feature.append(([down],[current]))
                    if y+2*w<self.size:
                        right = ((x,y+w),(x+h,y+2*w))
                        self.feature.append(([current],[right]))
                        self.feature.append(([right],[current]))

                    if x+3*h<self.size:
                        downdown = ((x+2*h,y),(x+3*h,y+w))
                        self.feature.append(([current,downdown],[down]))
                        self.feature.append(([down],[current,downdown]))
                    if y+3*w<self.size:
                        rightright = ((x,y+2*w),(x+h,y+3*w))
                        self.feature.append(([current, rightright], [right]))
                        self.feature.append(([right], [current, rightright]))
                    if x+2*h<self.size and y+2*w<self.size:
                        downright = ((x+h,y+w),(x+2*h,y+2*w))
                        self.feature.append(([current,downright],[right,down] ))
                        self.feature.append(([right,down],[current,downright]))

    def compute_feature(self,X):
        ret = np.zeros((len(self.feature),X.shape[0]))
        for i in tqdm(range(len(self.feature))):
            feature = self.feature[i]
            for j in range(X.shape[0]):
                integral_img = X[j]
                ret[i,j] =compute_feature_using_integral(integral_img,feature)
        return ret.transpose() # return a image x features matrix

    def prelim_feature_selection_sklearn(self,X,Y,k = 500):
        # f_sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        f_sel = SelectKBest(k = k)
        f_sel.fit(X,Y)
        return f_sel.get_support(),f_sel.transform(X)

    def choose_feature_w_mask(self,feature_mask):
        features = []
        for i in range(feature_mask.shape[0]):
            if feature_mask[i]:
                features.append(self.feature[i])
        self.feature = features

    def fit(self,X,Y):
        # We are assuming that compute feature is called by the user
        # For now, add only one layer
        ada = AdaBoost(50)
        ada.fit(X,Y)
        ada.save('adaboost/save_{}.pkl'.format(len(self.layer)))
        self.layer.append(ada)


    def load(self):
        list_ada = glob.glob('adaboost/save_*.pkl')
        self.layer = []
        for saved in list_ada:
            ada = AdaBoost()
            ada.load(saved)
            self.layer.append(ada)

    def save(self):
        for i,ada in enumerate(self.layer):
            ada.save('adaboost/save_{}.pkl'.format(i))

    def predict(self,X): # X has to be a proper resized image
        X = compute_integral_image(X)
        for ada in self.layer:
            if ada.predict(X)==0:
                return 0
        return 1

    def get_sliding_window(self,img):
        sliding_window = []
        img = cv2.resize(img,(200,200))
        for window_size in [200,150,120,100,80,50,40,20]:
            for x in range(self.size):
                for y in range(self.size):
                    if x+window_size<200 and y+window_size<200:
                        sliding_window.append( (np.copy(img[x:(x+window_size),y:(y+window_size)]),
                                        x,y,window_size) )
        return sliding_window

    def detect(self,img):
        pass # This feature will not work at this moment
        coord = []
        sliding_window = self.get_sliding_window(img)
        for patch,x,y,window_size in sliding_window:
            pred = self.predict(cv2.resize(patch,(self.size,self.size)))
            if pred >0:
                coord.append((pred,(x,y) ,(x+window_size,y+window_size) ))
        coord.sort(key=lambda x: x[0])
        coord.reverse()
        ret = [(x[1],x[2]) for x in coord[:5]]
        return ret





if __name__ == '__main__':

    img_size = 17


    # Reading the dataset
    print("READING THE DATASET")
    list_file = glob.glob('FDDB-folds/*-ellipseList.txt')
    face = []
    non_face = []
    for fold in list_file[:6]:
        print(fold)
        ll = read_file(fold)
        a, b = create_face_dataset(ll,img_size)
        face.extend(a)
        non_face.extend(b)
    X = np.zeros((len(face)+len(non_face),img_size,img_size))
    Y = np.zeros((len(face)+len(non_face)))
    print("PREPARING THE INTEGRAL IMAGES")
    for i in range(len(face)):
        X[i] = compute_integral_image(face[i])
        Y[i] = 1
    for i in range(len(non_face)):
        X[i+len(face)] = compute_integral_image(non_face[i])
        Y[i+len(face)] = 0


    print('Raw data',X.shape,Y.shape)
    print('Face {} vs Non-face {}'.format(np.count_nonzero(Y==1),np.count_nonzero(Y==0)))


    # Init the model

    vl = ViolaJone(img_size)
    # Prepare the data for training
    try:
        feature_matrix = np.load('feature_matrix.npy')
    except:
        feature_matrix = vl.compute_feature(X)
        np.save('feature_matrix.npy', feature_matrix)
    X = feature_matrix
    print('feature matrix raw',X.shape)

    # Implement feature selection
    try:
        feature_mask = np.load('prelim_feature_mask.npy')
        X = X[:,feature_mask]
    except:
        feature_mask, X = vl.prelim_feature_selection_sklearn(X, Y,5000)
        np.save('prelim_feature_mask.npy', feature_mask)

    print('prelim feature selection',X.shape)
    vl.choose_feature_w_mask(feature_mask)
    print("Training the viola jones model")
    vl.fit(X,Y)
    vl.save()