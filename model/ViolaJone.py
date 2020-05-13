import numpy as np
import cv2
from utils import *
import glob
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.metrics import confusion_matrix
import pickle
import os

class AdaBoost:
    def __init__(self,num_feature = None):
        self.T = num_feature
        self.weak_clf = []
        self.threshold = None

    def weak_learner(self, X, Y, W):
        """
        :param X: The computed values for some features
        :param Y: The true label
        :param W: Weight
        :return: Error, parity and threshold (E,P,and theta)
        """
        # X is a vector of images for some feature
        # This function is to train the weak classifier
        # by learning the threshold and parity (which is for the sign)
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
            if E > below_as_pos:
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
        :param haar_feature: The haar feature given by viola jones
        :return: None, all the weak classifier will be in self.chosen
        """

        # Initialize W
        num_pos = np.count_nonzero(Y == 1)
        num_neg = np.count_nonzero(Y == 0)
        W = np.zeros(Y.shape)
        for i in range(Y.shape[0]):
            if Y[i] ==0:
                W[i] = 1/(2*num_neg)
            else:
                W[i] = 1/(2*num_pos)

        for num_classifier in range(self.T):  # Choose only T weak classifier
            print("CHOOSING THE {}/{} WEAK CLASSIFIER".format(num_classifier+1,self.T),flush=True)
            # normalize weight
            W = W/np.sum(W)

            # select the best weak classifier
            best_E = 999999999
            best_weak_classifier = None
            for i in tqdm(range(len(haar_feature))):
                E,P,thres = self.weak_learner(X[:,i],Y,W)
                if best_E>E:
                    best_E = E
                    best_weak_classifier = (i,haar_feature[i],P,thres)

            # Now we have best_weak_classifier
            Beta = best_E/(1-best_E)
            alpha = np.log(1/Beta)
            j, feature, P, thres = best_weak_classifier
            self.weak_clf.append((alpha, feature, P, thres))
            print("CHOSE WEAK CLASSIFIER {} with best error {:.3f} and weight {:.3f}".
                  format(j, best_E, alpha))
            # now we adjust the W
            for i in range(Y.shape[0]):
                pred = 1 if P*X[i,j]<P*thres else 0
                e = 0 if pred == Y[i] else 1
                W[i] = W[i]*Beta**(1-e)

        self.threshold = sum([x[0] for x in self.weak_clf])/2

    def decrease_threshold(self):
        self.threshold-=0.01

    def predict(self, X):
        # of course X is a integral image
        sum_pred = 0
        for alpha, feature, P, thres in self.weak_clf:
            sum_pred += alpha*(1 if P*compute_feature_using_integral(X,feature)<P*thres else 0)
        return sum_pred-self.threshold

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump((self.weak_clf,self.threshold), f)

    def load(self, name):
        with open(name, 'rb') as f:
            self.weak_clf,self.threshold = pickle.load(f)
        self.T = len(self.weak_clf)

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
                    if x+2*h+1<self.size:
                        down = ((x+h+1,y),(x+2*h+1,y+w))
                        self.feature.append(([current],[down]))
                        self.feature.append(([down],[current]))
                    if y+2*w+1<self.size:
                        right = ((x,y+w+1),(x+h,y+2*w+1))
                        self.feature.append(([current],[right]))
                        self.feature.append(([right],[current]))

                    if x+3*h+2<self.size:
                        downdown = ((x+2*h+2,y),(x+3*h+2,y+w))
                        self.feature.append(([current,downdown],[down]))
                        self.feature.append(([down],[current,downdown]))
                    if y+3*w+2<self.size:
                        rightright = ((x,y+2*w+2),(x+h,y+3*w+2))
                        self.feature.append(([current, rightright], [right]))
                        self.feature.append(([right], [current, rightright]))
                    if x+2*h+1<self.size and y+2*w+1<self.size:
                        downright = ((x+h+1,y+w+1),(x+2*h+1,y+2*w+1))
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

    def fit(self,X,Y,non_faces,valid_X,valid_Y):
        """
        :param X: The feature matrix of img x feature
        :param Y: The true label
        :param non_faces: The integral non_faces images in X
        :param valid_X: The validation set, numpy array of integral images
        :param valid_Y: The label for valid_X
        :return:
        """
        # We are assuming that compute feature is called by the user
        max_fp_rate = 0.4
        min_detect_rate = 0.98
        F_target = 0.02
        P = X[Y==1]
        N = X[Y==0]
        F = [1.0]
        D = [1.0]
        i = 0
        while F[i]>F_target:
            print(i,P.shape,N.shape)
            i=i+1
            n = 5
            F.append(F[i-1])
            D.append(D[i-1])
            while F[i]>max_fp_rate*F[i-1]:
                if len(self.layer)>i:
                    self.layer.pop()
                n+=5
                print(F[i], D[i])
                print(i, n,end = "*****\n")
                ada = AdaBoost(n)
                train = np.vstack((P,N))
                label = np.append(np.ones(P.shape[0]),np.zeros(N.shape[0]))
                ada.fit(train,label,self.feature)
                self.layer.append(ada)
                # Eval on test
                prediction = [self.predict(img) > 0 for img in valid_X]
                tn, fp, fn, tp = confusion_matrix(valid_Y, prediction).ravel()
                F[i] = fp / (tn + fp)
                D[i] = tp / (tp + fn)
                # adjust threshold
                while D[i]<min_detect_rate*D[i-1]:
                    self.layer[-1].decrease_threshold()
                    prediction = [self.predict(img)>0 for img in valid_X]
                    tn, fp, fn, tp = confusion_matrix(valid_Y,prediction).ravel()
                    F[i] = fp/(tn+fp)
                    D[i] = tp/(tp+fn)
            if F[i]>F_target:
                # Time to add only false images to N
                pred = np.array([self.predict(x) for x in non_faces])
                N = X[Y==0]
                N = N[pred==1]





    def load(self,name):
        list_ada = glob.glob(name+'/save_*.pkl')
        self.layer = []
        for saved in list_ada:
            ada = AdaBoost()
            ada.load(saved)
            self.layer.append(ada)

    def save(self,name):
        os.makedirs(name,exist_ok=True)
        for i,ada in enumerate(self.layer):
            ada.save(name+'/save_{}.pkl'.format(i))

    def predict(self,X):
        """
        :param X: An integral image
        :return: The prediction if face or not (1 or 0)
        """
        for ada in self.layer:
            pred = ada.predict(X)
            if pred<=0:
                return 0
        return 1

    def get_sliding_window(self,img):
        sliding_window = []
        img = cv2.resize(img,(200,200))
        for window_size in [80,50,40]:
            for x in range(int(400/window_size)):
                for y in range(int(400/window_size)):
                    xx = x*int(window_size/2)
                    yy = y*int(window_size/2)
                    if xx+window_size<200 and yy+window_size<200:
                        sliding_window.append( (np.copy(img[xx:(xx+window_size),
                                                        yy:(yy+window_size)]),
                                        xx,yy,window_size) )
        return sliding_window

    def detect(self,img):
        coord = []
        sliding_window = self.get_sliding_window(img)
        for patch,x,y,window_size in tqdm(sliding_window):
            patch = cv2.resize(patch,(self.size,self.size))
            patch = compute_integral_image(patch)
            pred = self.predict(patch)
            if pred >0:
                coord.append((pred,(x,y) ,(x+window_size,y+window_size) ))
        coord.sort(key=lambda x: x[0])
        ret = [(x[1],x[2]) for x in coord[-2:]]
        return ret





if __name__ == '__main__':

    img_size = 17

    # Reading the dataset
    print("READING THE DATASET")
    list_file = glob.glob('FDDB-folds/*-ellipseList.txt')
    face = []
    non_face = []
    for fold in list_file[:5]:
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
    non_face_training = X[Y == 0]
    print('Raw data',X.shape,Y.shape)
    print('Face {} vs Non-face {}'.format(np.count_nonzero(Y==1),np.count_nonzero(Y==0)))


    print("PREPARING THE VALIDATION MATRIX")
    face = []
    non_face = []
    for fold in list_file[5:]:
        print(fold)
        ll = read_file(fold)
        a, b = create_face_dataset(ll, img_size)
        face.extend(a)
        non_face.extend(b)
    valid_X = np.zeros((len(face) + len(non_face), img_size, img_size))
    valid_Y = np.zeros((len(face) + len(non_face)))
    for i in range(len(face)):
        valid_X[i] = compute_integral_image(face[i])
        valid_Y[i] = 1
    for i in range(len(non_face)):
        valid_X[i + len(face)] = compute_integral_image(non_face[i])
        valid_Y[i + len(face)] = 0
    print('VALIDATION Face {} vs Non-face {}'.format(np.count_nonzero(valid_Y==1),np.count_nonzero(valid_Y==0)))

    # Init the model
    vl = ViolaJone(img_size)
    # Prepare the data for training
    try:
        feature_matrix = np.load('../old_feature_matrix.npy')
    except:
        feature_matrix = vl.compute_feature(X)
        np.save('../old_feature_matrix.npy', feature_matrix)
    X = feature_matrix
    print('feature matrix raw', X.shape)

    # Implement feature selection
    try:
        feature_mask = np.load('../prelim_feature_mask.npy')
        X = X[:,feature_mask]
    except:
        feature_mask, X = vl.prelim_feature_selection_sklearn(X, Y,5000)
        np.save('../prelim_feature_mask.npy', feature_mask)

    print('prelim feature selection',X.shape)
    vl.choose_feature_w_mask(feature_mask)


    print("Training the viola jones model")
    vl.fit(X,Y,non_face_training,valid_X,valid_Y)
    vl.save('../trained_cascade')