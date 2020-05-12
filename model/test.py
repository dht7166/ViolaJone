from utils import *
from ViolaJone import ViolaJone
import glob
from sklearn.metrics import f1_score , confusion_matrix


def test_overall_acc(img_size):
    list_file = glob.glob('FDDB-folds/*-ellipseList.txt')
    face = []
    non_face = []
    for fold in list_file[6:]:
        print(fold)
        ll = read_file(fold)
        a, b = create_face_dataset(ll,img_size)
        face.extend(a)
        non_face.extend(b)
    X = np.zeros((len(face) + len(non_face), img_size, img_size))
    Y = np.zeros((len(face) + len(non_face)))
    for i in range(len(face)):
        X[i] = face[i]
        Y[i] = 1
    for i in range(len(non_face)):
        X[i + len(face)] = non_face[i]
        Y[i + len(face)] = 0
    print('Face {} vs Non-face {}'.format(np.count_nonzero(Y == 1), np.count_nonzero(Y == 0)))
    model = ViolaJone(img_size)
    model.load('trained_20')
    pred = np.zeros(Y.shape)
    for i in range(X.shape[0]):
        pred[i] =1 if  model.predict(X[i])>0 else 0

    print(f1_score(Y, pred))
    print(np.sum(pred == Y) / X.shape[0])
    tn, fp, fn, tp = confusion_matrix(Y,pred).ravel()
    print(tn,fp,fn,tp)

def demo_predict(img_size):
    list_file = glob.glob('FDDB-folds/*-ellipseList.txt')
    model = ViolaJone(img_size)
    model.load()
    for fold in list_file[6:]:
        all_coord = read_file(fold)
        for img_name, list_coord in all_coord:
            img = read('originalPics/' + img_name + '.jpg')
            for coord in list_coord:  # Get face pic
                axes, center, angle = coord
                tl, br = convert_ellipse_to_rect(axes, center, angle)
                crop = np.copy(img[tl[1]:br[1], tl[0]:br[0]])
                display(crop)
                print(model.predict(cv2.resize(crop, (img_size, img_size ))))

def demo_detect():
    list_img = glob.glob('test_images/*.jpg')
    model = ViolaJone(17)
    model.load('trained_10')
    for feature in model.layer[0].weak_clf:
        print(feature)
    for image in list_img:
        img = read(image)
        coord = model.detect(img)
        color = cv2.imread(image)
        color = cv2.resize(color,(200,200))
        for tl,br in coord:
            cv2.rectangle(color,tl,br,(0, 0, 255))
        display(color)


if __name__=='__main__':
    test_overall_acc(17)
    # demo_predict(17)
    # demo_detect()
