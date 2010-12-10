import scipy as sp
from scikits.learn import svm
from scikits.learn.logistic import LogisticRegression

'''SVM classifier module'''


def classifier_train(train_features,
                     train_labels,
                     test_features,
                     svm_eps = 1e-5,
                     svm_C = 10**4,
                     classifier_type = "liblinear"
                     ):
    """ Classifier training using SVMs

    Input:
    train_features = training features (both positive and negative)
    train_labels = corresponding label vector
    svm_eps = eps of svm
    svm_C = C parameter of svm
    classifier_type = liblinear or libsvm"""

    #sphering
    train_features, test_features = __sphere(train_features, test_features)

    if classifier_type == 'liblinear':
        clf = svm.LinearSVC(eps = svm_eps, C = svm_C)
    if classifier_type == 'libSVM':
        clf = svm.SVC(eps = svm_eps, C = svm_C, probability = True)
    elif classifier_type == 'LRL1':
        clf = LogisticRegression(C=svm_C, penalty = 'l1')
    elif classifier_type == 'LRL2':
        clf = LogisticRegression(C=svm_C, penalty = 'l1')

    clf.fit(train_features, train_labels)
    
    return clf



#sphere data
def __sphere(train_data, test_data):
    '''make data zero mean and unit variance'''

    fmean = train_data.mean(0)
    fstd = train_data.std(0)

    train_data -= fmean
    test_data -= fmean
    fstd[fstd==0] = 1
    train_data /= fstd
    test_data /= fstd

    return train_data, test_data




def classify(train_features,
                     train_labels,
                     test_features,
                     test_labels):

    '''Classify data and return
        accuracy
        area under curve
        average precision
        and svm raw data in a dictianary'''

    #mapping labels to 0,1
    labels = sp.unique(sp.concatenate((train_labels, test_labels)))
    assert labels.size == 2
    label_to_id = dict([(k,v) for v, k in enumerate(labels)])

    train_ys = sp.array([label_to_id[i] for i in train_labels])
    test_ys = sp.array([label_to_id[i] for i in test_labels])

    #train
    model = classifier_train(train_features, train_ys,
                            test_features)

    #test
    weights = model.coef_.ravel()
    bias = model.intercept_.ravel()
    predict = sp.dot(test_features, weights) + bias
    def_predict = model.predict(test_features)

    #raw data to be saved for future use
    cls_data = {'def_prdict' : def_predict, 'predict' : predict,  
                'test_lables' : test_labels, 'coef' : model.coef_, 
                'intercept' : model.intercept_}

    #accuracy
    hit = 0
    for i_ind in range(len(test_labels)):
        if (predict[i_ind]/abs(predict[i_ind])) * int(test_labels[i_ind]) == 1: hit += 1

    accu = sp.single(hit)/len(test_labels)

    #precison and recall
    c = predict
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(test_ys[si] == 1))
    fp = sp.cumsum(sp.single(test_ys[si] == 0))
    rec = tp /sp.sum(test_ys > 0)
    prec = tp / (fp + tp)

    ap = 0
    rng = sp.arange(0, 1.1, .1)
    for th in rng:
        p = prec[rec>=th].max()
        if p == []:
               p =0
        ap += p / rng.size

    #area under curve
    h = sp.diff(rec)
    auc = sp.sum(h * (prec[1:] + prec[:-1])) / 2.0


    return accu, auc, ap, cls_data

 

