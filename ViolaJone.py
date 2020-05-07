import numpy as np
import cv2
from utils import *
import glob
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold, SelectKBest
import pickle

class ViolaJone:
    def __init__(self,size,final_feature_size):
        self.T = final_feature_size
        self.size = size
        self.create_haar_feature()
        self.chosen = [] # list of chosen features, including the regions, the parity and threshold

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


    def fit(self,X,Y):
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
            for i in range(len(self.feature)):
                E,P,thres = self.weak_learner(X[:,i],Y,W)
                if best_E>E:
                    best_E = E
                    best_weak_classifier = (i,self.feature[i],P,thres)

            # Now we have best_weak_classifier
            Beta = best_E/(1-best_E)
            alpha = np.log(1/Beta)
            i,feature,P,thres = best_weak_classifier
            self.chosen.append((alpha,feature,P,thres))
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
            pickle.dump(self.chosen,f)

    def load(self,name):
        with open(name,'rb') as f:
            self.chosen = pickle.load(f)
        self.T = len(self.chosen)

    def predict(self,X):
        # of course X is a small 15x15 image
        X = compute_integral_image(X)
        sum_pred = 0
        baseline = 0
        for alpha,feature,P,thres in self.chosen:
            sum_pred+=alpha*(1 if P*compute_feature_using_integral(X,feature)<P*thres else 0)
            baseline+=alpha/2
        return 1 if sum_pred>baseline else 0




if __name__ == '__main__':

    # Reading the dataset
    print("READING THE DATASET")
    list_file = glob.glob('FDDB-folds/*-ellipseList.txt')
    face = []
    non_face = []
    for fold in list_file[:5]:
        print(fold)
        ll = read_file(fold)
        a, b = create_face_dataset(ll)
        face.extend(a)
        non_face.extend(b)
    X = np.zeros((len(face)+len(non_face),15,15))
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
    vl = ViolaJone(15,50)
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
        feature_mask, X = vl.prelim_feature_selection_sklearn(X, Y,1000)
        np.save('prelim_feature_mask.npy', feature_mask)

    print('prelim feature selection',X.shape)
    vl.choose_feature_w_mask(feature_mask)
    print("Training the viola jones model")
    vl.fit(X,Y)
    vl.save('violajones5fold.pkl')