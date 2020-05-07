from utils import *
from ViolaJone import ViolaJone
import glob
from sklearn.metrics import f1_score

if __name__=='__main__':
    list_file = glob.glob('FDDB-folds/*-ellipseList.txt')
    face = []
    non_face = []
    for fold in list_file[6:]:
        print(fold)
        ll = read_file(fold)
        a,b = create_face_dataset(ll)
        face.extend(a)
        non_face.extend(b)
    X = np.zeros((len(face) + len(non_face), 15, 15))
    Y = np.zeros((len(face) + len(non_face)))
    for i in range(len(face)):
        X[i] = face[i]
        Y[i] = 1
    for i in range(len(non_face)):
        X[i + len(face)] = non_face[i]
        Y[i + len(face)] = 0
    print('Face {} vs Non-face {}'.format(np.count_nonzero(Y==1),np.count_nonzero(Y==0)))
    model = ViolaJone(15,10)
    model.load('violajones5fold.pkl')
    pred = np.zeros(Y.shape)
    for i in range(X.shape[0]):
        pred[i] = model.predict(X[i])

    print(f1_score(Y,pred))
    print(np.sum(pred==Y)/X.shape[0])
