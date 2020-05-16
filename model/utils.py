import cv2
import math
import numpy as np

def read(name): # default to read gray scale image
    return cv2.imread(name,cv2.IMREAD_GRAYSCALE)

def display(img,name = 'SOME_RANDOM_NAME'):
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
    # Some advanced math shit
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
        if t>0 and l >0:
            return S[b,r]-S[t-1,r]-S[b,l-1]+S[t-1,l-1]
        if t == 0 and l >0:
            return S[b,r]-S[b,l-1]
        if t>0 and l == 0:
            return S[b,r] - S[t-1,r]
        if t==0 and l ==0:
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


def normalize(img):
    return (img-img.mean())/(img.std()+1e-7)

def create_face_dataset(all_coord,img_size):
    X = [] # face
    Y = [] # non-face
    for img_name,list_coord in all_coord:
        # print('originalPics/'+img_name+'.jpg')
        img = read('originalPics/'+img_name+'.jpg')
        for coord in list_coord:  # Get face pic
            axes,center,angle = coord
            tl,br = convert_ellipse_to_rect(axes,center,angle)
            crop =np.copy(img[tl[1]:br[1],tl[0]:br[0]])
            crop = cv2.resize(crop,(img_size,img_size))
            X.append(normalize(crop))
        for coord in list_coord:  # Remove the faces
            axes,center,angle = coord
            tl,br = convert_ellipse_to_rect(axes,center,angle)
            img[tl[1]:br[1], tl[0]:br[0]] = 0

        img = cv2.resize(img,(10*img_size,10*img_size))
        for i in range(6):
            start = i*img_size
            end = start+img_size
            crop = np.copy(img[start:end,start:end])
            Y.append(normalize(crop))
    return X,Y

if __name__ == '__main__':
    # center,axes,angle = read_elip_coord('41.936870 27.064477 1.471906 184.070915 129.345601  1')
    # image = 'originalPics/2002/08/31/big/img_17676.jpg'
    # img = read(image)
    # img, elip = resize(img,(axes,center,angle),200,200)
    # axes,center,angle = elip
    # print(axes,center,angle)
    # img = cv2.ellipse(img, center, axes, angle, 0, 360, (0, 0, 255))
    # tl, br = convert_ellipse_to_rect(center, axes, angle)
    # img = cv2.rectangle(img, tl, br, (0, 0, 255))
    # display(img)
    # print(tl,br)
    import glob
    list_img = glob.glob('train/non-face/*.pgm')
    list_img.reverse()
    for img_name in list_img:
        img = read(img_name)
        img = cv2.resize(img,(100,100))
        display(img)