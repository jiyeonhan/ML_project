import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn import neighbors
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#plabel = [11, 111, 211, 2212]
#plabel_title = ['e','pi0','pi','p']
plabel = [11, 13, 211, 2212]
plabel_title = ['e', 'mu', 'pi','p']

def define_feature(mat, ind):
    
    inf = data[abs(mat['pdg']) == ind]
    feature = inf.values[:, 2:6]
    label = np.zeros((feature.shape[0], 5))
    label[:, plabel.index(ind)] = 1

    return feature, label

def label_binary(info):
    le = np.zeros((info.shape[0], 2))
    lel = np.zeros((info.shape[0],1))
    for i in range(len(info)):
        if info[i][0]==11 or info[i][0]==22:
            le[i][0] = 1
            lel[i][0] = 1
        else:
            le[i][1] = 1
            lel[i][0] = 0

    return le, lel

def label_multi(info):
    le = np.zeros((info.shape[0], 4))
    lel = np.zeros((info.shape[0],1))
    for i in range(len(info)):
        pind = plabel.index(abs(info[i][0]))
        le[i][pind] = 1
        lel[i][0] = pind

    return le, lel

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    


if __name__ == '__main__':

    data = pd.read_csv("../input/minerva_data/particle_info_more.csv")    
    print(data.head())

    data_filter = data[(abs(data.pdg)==11) | (abs(data.pdg)==22) | (abs(data.pdg)==111) | (abs(data.pdg)==211) | (abs(data.pdg)==2212)]
    #em_filter = data[(abs(data.pdg)==11) | (abs(data.pdg)==22)].values[:, 2:6]
    #em_filter = data[(abs(data.pdg)==11)].values[:, 2:6]
    #nevt = em_filter.shape[0]
    #pi0_filter = data[(abs(data.pdg)==111)].values[:nevt, 2:6]
    #pi_filter = data[(abs(data.pdg)==211)].values[:nevt, 2:6]
    #p_filter = data[(abs(data.pdg)==2212)].values[:nevt, 2:6]

    sel_feature = [2,3,4,5]
    #em_filter = data[(abs(data.pdg)==11) | (abs(data.pdg)==22)].values[:, (2,3,4,5)]
    #nevt = em_filter.shape[0]
    #pi0_filter = data[(abs(data.pdg)==111)].values[:nevt, (2,3)]
    #mu_filter = data[(abs(data.pdg)==13)].values[:nevt, (2,3,4,5)]
    #pi_filter = data[(abs(data.pdg)==211)].values[:nevt, (2,3,4,5)]
    #p_filter = data[(abs(data.pdg)==2212)].values[:nevt, (2,3,4,5)]
    """
    em_filter = data[(abs(data.pdg)==11) | (abs(data.pdg)==22)].values[:, (4,3,2,5)]
    nevt = em_filter.shape[0]
    #pi0_filter = data[(abs(data.pdg)==111)].values[:nevt, (2,3)]
    mu_filter = data[(abs(data.pdg)==13)].values[:nevt, (4,3,2,5)]
    pi_filter = data[(abs(data.pdg)==211)].values[:nevt, (4,3,2,5)]
    p_filter = data[(abs(data.pdg)==2212)].values[:nevt, (4,3,2,5)]
    """
    em_filter = data[(abs(data.pdg)==11) | (abs(data.pdg)==22)].values[:, (4,3,2,5,6,7,8)]
    nevt = em_filter.shape[0]
    #pi0_filter = data[(abs(data.pdg)==111)].values[:nevt, (2,3)]
    mu_filter = data[(abs(data.pdg)==13)].values[:nevt, (4,3,2,5,6,7,8)]
    pi_filter = data[(abs(data.pdg)==211)].values[:nevt, (4,3,2,5,6,7,8)]
    p_filter = data[(abs(data.pdg)==2212)].values[:nevt, (4,3,2,5,6,7,8)]

    print("total events = ", em_filter.shape[0], mu_filter.shape[0], pi_filter.shape[0], p_filter.shape[0])

    em_label = np.zeros((nevt, 4))
    #pi0_label = np.zeros((nevt, 4))
    mu_label = np.zeros((nevt, 4))
    pi_label = np.zeros((nevt, 4))
    p_label = np.zeros((nevt, 4))

    em_label_mark = np.zeros((nevt, 1))
    #pi0_label_mark = np.zeros((nevt, 1))
    mu_label_mark = np.zeros((nevt, 1))
    pi_label_mark = np.zeros((nevt, 1))
    p_label_mark = np.zeros((nevt, 1))

    for i in range(nevt):
        em_label[i][0] = 1
        #pi0_label[i][1] = 1
        mu_label[i][1] = 1
        pi_label[i][2] = 1
        p_label[i][3] = 1
        
        em_label_mark[i] = 0
        #pi0_label_mark[i] = 1
        mu_label_mark[i] = 1
        pi_label_mark[i] = 2
        p_label_mark[i] = 3

    #feature = np.vstack((em_filter,pi0_filter,pi_filter,p_filter))
    #label_mul = np.vstack((em_label, pi0_label, pi_label, p_label))
    #label_mul_mark = np.vstack((em_label_mark, pi0_label_mark, pi_label_mark, p_label_mark))
    feature = np.vstack((em_filter, mu_filter, pi_filter, p_filter))
    label_mul = np.vstack((em_label, mu_label, pi_label, p_label))
    label_mul_mark = np.vstack((em_label_mark, mu_label_mark, pi_label_mark, p_label_mark))
    #label_mul_mark = np.argmax(label_mul)
    print("label = ", label_mul_mark)

    x_train_mul, x_val_mul, y_train_mul, y_val_mul = train_test_split(feature, label_mul_mark.ravel(-1), test_size=0.2)
    
    classes_mul = plabel_title
    print("data manipulation end =============, total nevt for particle = ", nevt)

    """
    plt.figure(1)
    plt.xlabel('Particle Class')
    tick_marks_bi = np.arange(len(label_bi[0]))
    #plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks_bi, classes_bi)
    plt.hist(label_bi_mark, bins=[-0.5,0.5,1.5,2.5], range=(-0.5,1.0))
    #plt.hist(label_bi_mark, bins=[-0.5,0.5,1.5,2.5])
    plt.savefig('plot_label_binary_3.png')
    """
    
    plt.figure(3)
    plt.xlabel('Particle Class')
    tick_marks_mul = np.arange(len(label_mul[0]))
    plt.xticks(tick_marks_mul, classes_mul)
    #plt.hist(np.argmax(label_mul, axis=1), bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5], range=(-0.5,4.0))
    plt.hist(label_mul_mark, bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5], range=(-0.5,4.0))
    #plt.hist(label_mul_mark, bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5])
    plt.savefig('plot_label_multi_flat_reg_3.png')


    plt.figure(4)
    max = [10.0, 40.0, 7.5, 2.5]
    
    for i in range(len(sel_feature)):
        plt.subplot(2,2,i+1)
        eminfo = np.vstack((em_filter[:, i]))
        #pi0info = np.vstack((pi0_filter[:, i]))
        muinfo = np.vstack((mu_filter[:, i]))
        piinfo = np.vstack((pi_filter[:, i]))
        pinfo = np.vstack((p_filter[:, i]))
        
        plt.hist(eminfo, 20, range=[0.0, max[i]], facecolor='blue', histtype=u'step', label='e')
        #plt.hist(pi0info, 20, range=[0.0, max[i]], facecolor='black', histtype=u'step', label='pi0')
        plt.hist(muinfo, 20, range=[0.0, max[i]], facecolor='black', histtype=u'step', label='mu')
        plt.hist(piinfo, 20, range=[0.0, max[i]], facecolor='pink', histtype=u'step', label='pi')
        plt.hist(pinfo, 20, range=[0.0, max[i]], facecolor='red', histtype=u'step', label='p')
        if i==0:
            plt.legend(loc='upper right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.xlabel('Feature '+str(i+1))

    plt.subplots_adjust(bottom=0.10, right=0.95, top=0.92, left=0.13, hspace=0.25, wspace=0.35)
    plt.savefig("features_pid_with_mu_3.png")


    #n_neighbors = 4
    #model_knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    #model_knn = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    #model_knn = Pipeline((
    #        ("scaler", StandardScaler()),
    #        ("reg", LogisticRegression(multi_class="multinomial", solver="lbfgs")),
    #        ))
    #from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier

    """
    model_knn = Pipeline((
            ("scaler", StandardScaler()),
            #("reg", DecisionTreeClassifier(max_depth=5)),
            ("reg", LinearSVC()),
            ))
            """
    #model_knn = LogisticRegression(multi_class="multinomial", solver="lbfgs")

    model_knn = Pipeline((
            ("scaler", StandardScaler()),
            ("reg", SVC()),
            #("reg", DecisionTreeClassifier(max_depth=5)),
            #("reg", AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=600, learning_rate=1.5, algorithm="SAMME")),
            ))

    print("start fitting !!!! ")
    model_knn.fit(x_train_mul, y_train_mul)
    print("start predicting !!!! ")
    mul_pred = model_knn.predict(x_val_mul)
    mul_pred_prob = model_knn.predict_proba(x_val_mul)
    print("probability for classes = ", model_knn.predict_proba(x_val_mul)[:20])
    print("precision = ", model_knn.score(x_val_mul, y_val_mul))
    #print(bi_pred[:20], y_val_bi[:20])
    #print('--------------------------')
    #print(np.argmax(bi_pred[:20], axis=1), np.argmax(y_val_bi[:20], axis=1))
    mul_pred_label = np.argmax(mul_pred_prob, axis=1)
    mul_val_label = y_val_mul
    #mul_val_label = np.argmax(y_val_mul, axis=1)
    conf = confusion_matrix(mul_val_label, mul_pred_label)

    plt.figure(1)
    plot_confusion_matrix(conf, classes=plabel_title,
                          title='Confusion matrix, without normalization')
    plt.savefig("unnormalized_confusion_pid_mul_reg_3.png")

    plt.figure(2)
    plot_confusion_matrix(conf, classes=plabel_title, normalize=True,
                          title='Confusion matrix, with normalization')
    plt.savefig("normalized_confusion_pid_mul_reg_3.png")

    """
    em_pred_prob = model_knn.predict_proba(em_filter)
    print("em_filter probability : ", em_pred_prob[:20])
    nonem_pred_prob = model_knn.predict_proba(nonem_filter)
    print("nonem_filter probability : ", nonem_pred_prob[:20])
    """

    """
    plt.figure(2)
    #plot_hist(em_pred_prob[0], bins=[0.1*i for i in range(11)], range=[0.0,1.0])
    plt.hist(em_pred_prob[0], 10, range=[0.0,1.0], facecolor='blue')
    plt.hist(nonem_pred_prob[0], 10, range=[0.0,1.0], facecolor='gray')
    plt.savefig("knn_prob_bi_3.png")
    """
    plt.show()
