import numpy as np
import cv2
import glob
import math
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.metrics import confusion_matrix
import pickle
import os
import json


def read(name): # default to read gray scale image
    return cv2.imread(name,cv2.IMREAD_GRAYSCALE)

def display(img,name = 'Image'):
    cv2.imshow(name,img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

def read_elip_coord(line):
    """
    Parse the string in line to elip coord
    :param line: a string, values are space seperated
    :return:
    """
    val = [int(float(x)) for x in line.split()]
    major_axis_radius = val[0]
    minor_axis_radius = val[1]
    angle             = val[2]
    center_x          = val[3]
    center_y          = val[4]
    return (center_x,center_y),(minor_axis_radius,major_axis_radius),angle # cv2 have a reversed axes

def convert_ellipse_to_rect(center,axes,angle):
    # Some advanced math
    angle = angle/180*math.pi
    cos = math.cos(angle)
    sin = math.sin(angle)
    a,b = axes
    height = int(math.sqrt(a**2*cos**2+b**2*sin**2))
    width = int(math.sqrt(a**2*sin**2 + b**2*cos**2))
    tl = (max(0,center[0]-height),max(0,center[1]-width))
    br = (max(0,center[0]+height),max(0,center[1]+width))
    return tl,br # top-left and bottom-right


def resize(img,elip_coord,W,H):
    oH,oW = img.shape
    center,axes,angle = elip_coord
    axes = (int(axes[0] * W /oW), int(axes[1] * H /oH))
    center = (int(center[0] * W /oW), int(center[1] * H /oH))
    img =cv2.resize(img,(W,H))
    return img,(axes,center,angle)

def compute_integral_image(img):
    S = img
    for i in range(S.ndim):
        S = np.cumsum(S,axis = i)
    return S

def compute_feature_using_integral(S,feature):
    """
    :param S: An integral image
    :param feature: a tuple, containing the positive and negative regions
    :return:
    """
    def calc(S,region):
        S = S.astype(int)
        t,l = region[0]
        b,r = region[1]
        if t > 0 and l > 0:
            return S[b,r]-S[t-1,r]-S[b,l-1]+S[t-1,l-1]
        if t == 0 and l > 0:
            return S[b,r]-S[b,l-1]
        if t > 0 and l == 0:
            return S[b,r] - S[t-1,r]
        if t == 0 and l == 0:
            return S[b][r]
    pos,neg = feature
    rpos = sum([calc(S,region) for region in pos])
    rneg = sum([calc(S,region) for region in neg])
    return rpos-rneg

def read_file(name):
    all_coord = []
    with open (name,'r') as f:
        while True:
            try:
                img_name = next(f).rstrip()
                num_faces = int(next(f).rstrip())
                coord = [read_elip_coord(next(f).rstrip()) for i in range(num_faces)]
                all_coord.append((img_name,coord))
            except:
                break
    return all_coord

def create_face_dataset(all_coord,img_size):
    X = [] # face
    Y = [] # non-face
    for img_name,list_coord in all_coord:
        img = read('originalPics/'+img_name+'.jpg')
        for coord in list_coord:  # Get face pic
            axes,center,angle = coord
            tl,br = convert_ellipse_to_rect(axes,center,angle)
            crop =np.copy(img[tl[1]:br[1],tl[0]:br[0]])
            crop = cv2.resize(crop,(img_size,img_size))
            X.append(crop)
        for coord in list_coord:  # Remove the faces
            axes,center,angle = coord
            tl,br = convert_ellipse_to_rect(axes,center,angle)
            img[tl[1]:br[1], tl[0]:br[0]] = 0

        img = cv2.resize(img,(10*img_size,10*img_size))
        for i in range(6):
            start = i*img_size
            end = start+img_size
            Y.append(img[start:end,start:end])
    return X,Y

def generate_json():
    json_list = [] #each element is a dictionary, {"iname": "1.jpg", "bbox": [1, 2, 3 ,5]}
    list_img = glob.glob('test_images/*.jpg')

    # initialize model
    model = ViolaJone(17)
    model.load('trained_10')

    for image in list_img:
        iname = os.path.basename(image)
        img = read(image)
        coord = model.detect(img)
        for c, r, c_w, r_w in coord:
            element = {"iname": iname, "bbox": [c, r, c_w, r_w]}
            json_list.append(element)

    #the result json file name
    output_json = "results.json"

    #dump json_list to result.json
    with open(output_json, 'w') as f:
        json.dump(json_list, f)


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

    def prelim_feature_selection_sklearn(self,X,Y,k = 5000):
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
            if pred > 0:
                coord.append((x,y,x+window_size,y+window_size))
        return coord


def main():
    


if __name__ == '__main__':
    main()