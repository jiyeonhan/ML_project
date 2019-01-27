import itertools

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from tensorflow import keras
from keras.models import load_model

import matplotlib.pyplot as plt

f = h5py.File("../input/minerva_data/wholevtimgs_127x94_me1Amc0000001_train.hdf5")
ft = h5py.File("../input/minerva_data/wholevtimgs_127x94_me1Amc0000001_test.hdf5")

nlayer = 94

#img_view = np.array([f['img_data']['hitimes-x'], f['img_data']['hitimes-u'], f['img_data']['hitimes-v']])
#print(img_view[:10])
print(f['img_data']['hitimes-x'].size, f['img_data']['hitimes-x'].shape, f['img_data']['hitimes-x'].dtype)
print(f['img_data']['hitimes-u'].shape, f['img_data']['hitimes-u'].dtype)
print(f['img_data']['hitimes-v'].shape, f['img_data']['hitimes-v'].dtype)
print(f['img_data']['hitimes-x'].shape[0])

image_x = np.array(f['img_data']['hitimes-x'][:50000])
image_u = np.array(f['img_data']['hitimes-u'][:50000])
image_v = np.array(f['img_data']['hitimes-v'][:50000])
label_val = np.array(f['gen_data']['sig_type'][:50000])
print("label shape = ", label_val)

label = np.zeros((len(image_x),3))
label.astype(int)
for l in range(len(label_val)):
    label[l][label_val[l][0]] = 1

#u_e = np.repeat(image[:,0,:,:], 1, axis=1)
#u_t = np.repeat(image[:,1,:,:], 1, axis=1)

#plt.hist(label, bins=20)
#plt.show()

x_e = image_x[:,0,:,:]
x_t = image_x[:,1,:,:]
u_e = image_u[:,0,:,:]
u_t = image_u[:,1,:,:]
v_e = image_v[:,0,:,:]
v_t = image_v[:,1,:,:]


## For test data
timage_x = np.array(ft['img_data']['hitimes-x'])
timage_u = np.array(ft['img_data']['hitimes-u'])
timage_v = np.array(ft['img_data']['hitimes-v'])
tlabel_val = np.array(ft['gen_data']['sig_type'])
print("label shape = ", label_val)

tlabel = np.zeros((len(timage_x),3))
tlabel.astype(int)
for l in range(len(tlabel_val)):
    tlabel[l][tlabel_val[l][0]] = 1


tx_e = timage_x[:,0,:,:]
tx_t = timage_x[:,1,:,:]
tu_e = timage_u[:,0,:,:]
tu_t = timage_u[:,1,:,:]
tv_e = timage_v[:,0,:,:]
tv_t = timage_v[:,1,:,:]


#x = np.array((u_e, u_t))
#x = x.reshape(-1, 127, 94)
#print("x shape : ", x.shape)
#input_layer = tf.reshape(image, [-1,28,28,1])
plt.figure()
plt.subplot(3,2,1)
plt.imshow(x_e[0])
plt.subplot(3,2,2)
plt.imshow(x_t[0])
plt.subplot(3,2,3)
plt.imshow(u_e[0])
plt.subplot(3,2,4)
plt.imshow(u_t[0])
plt.subplot(3,2,5)
plt.imshow(v_e[0])
plt.subplot(3,2,6)
plt.imshow(v_t[0])
#plt.show()
plt.savefig('image_all_300batch_new.png')

u_e = np.repeat(u_e, 2, axis=2)
u_t = np.repeat(u_t, 2, axis=2)
v_e = np.repeat(v_e, 2, axis=2)
v_t = np.repeat(v_t, 2, axis=2)
print("image shape : ", u_e.shape, u_t.shape, v_e.shape, v_t.shape, x_e.shape, x_t.shape)
#u_e = u_e.reshape(-1, 127, 94, 1) / 255.0
#image_all = np.transpose(image_all)

tu_e = np.repeat(tu_e, 2, axis=2)
tu_t = np.repeat(tu_t, 2, axis=2)
tv_e = np.repeat(tv_e, 2, axis=2)
tv_t = np.repeat(tv_t, 2, axis=2)

timage_all = np.array((tx_e, tu_e, tv_e, tx_t, tu_t, tv_t))
timage_all = timage_all.reshape(-1, 127, nlayer, 6)

test_X = timage_all
test_y = tlabel
#print("train and test sample :", train_X.shape, test_X.shape)
print("where is error 2 -------")

training_iters = 5
learning_rate = 0.01
batch_size = 128
#batch_size = 200

n_classes = 3

#both placeholders are of type float
#x = tf.placeholder("float", [None, 28, 28, 1])
x = tf.placeholder("float", [None, 127, nlayer, 6])
y = tf.placeholder("float", [None, n_classes])

"""
def variable_summaries(var):
  ###Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
"""

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

weights = {
    'wc1': tf.get_variable('W0', shape=(4,4,6,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(4,4,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(4,4,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    #'wd1': tf.get_variable('W3', shape=(32*24*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(16*12*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    #'wd1': tf.get_variable('W3', shape=(2*2*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    #Important to understand 16*12 (depending on number of pooling procedure, 129/2/2/2 = 16, 94/2/2/2=12)
    'out': tf.get_variable('W4', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
    }

biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
    }

print("wd1 shape !!! ", weights['wd1'].get_shape().as_list()[0])

def conv_net(x, weights, biases):

    #here we call the conv2d function we had defined above and pass the input image x, weight wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2x2 matrix window and outputs a 14x14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
# and both will a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#Calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
# Initializing the variables
init = tf.global_variables_initializer()

split_dim = 2000
# Run the session for the training
with tf.Session() as sess:
    
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    #summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    merged = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter('./Output/temp')
    summary_writer = tf.summary.FileWriter('./Output/train', sess.graph)
    #summary_test_writer = tf.summary.FileWriter('./Output/test')
    print("error test inside of tf.Session() --------")
    for i in range(training_iters):

        for k in range(len(x_e)//split_dim):
            print("splitting sample ", k, "out of", len(x_e)//split_dim)
            st = k*split_dim
            ed = min((k+1)*split_dim, len(x_e))

            image_split = np.array((x_e[st:ed], u_e[st:ed], v_e[st:ed], x_t[st:ed], u_t[st:ed], v_t[st:ed]))
            #print("image shape for all : ", image_split.shape)

            image_split = image_split.reshape(-1, 127, nlayer, 6)
            label_split = label[k*split_dim:min((k+1)*split_dim, len(x_e))]

            #image = image.reshape(-1, 127, nlayer, 2) / 255.0
            train_X = image_split
            train_y = label_split
            #train_X, test_X, train_y, test_y = train_test_split(image_split, label_split, test_size=0.25, random_state=42)

            for batch in range(len(train_X)//batch_size):
            #print("Batch size check => ", len(train_X), len(train_y), len(train_X)//batch_size)
                batch_x = train_X[batch*batch_size:min((batch+1)*batch_size, len(train_X))]
                batch_y = train_y[batch*batch_size:min((batch+1)*batch_size, len(train_y))]
            #batch_x = train_X[0:500/12]
            #batch_y = train_y[0:492]
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            #print("batch shape => ", batch_x.shape, batch_y.shape)
                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            #print("batch shape 2 => ", batch_x.shape, batch_y.shape)
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            #summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_x, y: batch_y})
                summary_writer.add_summary(summary, i)
            #print("batch shape 3 => ", batch_x.shape, batch_y.shape)
                
        #print("Iter " + str(i) + ", Loss= " + \
            #          "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #          "{:.5f}".format(acc))

        print("Iter " + str(i) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    print("Optimization Finished!")


    # Calculate accuracy for all 10000 mnist test images                                                                                  
    test_acc, valid_loss, test_pred = sess.run([accuracy, cost, pred], feed_dict={x: test_X,y : test_y})
    #test_summary = sess.run(summary, feed_dict={x: test_X, y : test_y})
    train_loss.append(loss)
    test_loss.append(valid_loss)
    train_accuracy.append(acc)
    test_accuracy.append(test_acc)
    #summary_test_writer.add_summary(test_summary, i)    
    print("Testing Accuracy:","{:.5f}".format(test_acc))
    
    #test_confusion = sess.run(confusion, feed_dict={x: test_X, y: test_y})
    #print("confusion matrix = ", test_confusion)
    confusion = tf.confusion_matrix(labels=np.argmax(test_y, axis=1), predictions=np.argmax(test_pred, axis=1), num_classes=3, name='confusion')
    print("confusion matrix = ", tf.Tensor.eval(confusion, feed_dict=None, session=None))
    confusion_matrix = tf.Tensor.eval(confusion, feed_dict=None, session=None)

    """
    ncFlist=[]
    ncTlist=[]
    cceFlist=[]
    cceTlist=[]
    ccmFlist=[]
    ccmTlist=[]
    for ti in range(len(test_y)):
        tp = np.argmax(test_y, axix=1)
        pp = np.argmax(test_pred, axis=1)
        if test_y[ti][0]==1:
            if tp==pp:
                ncTlist.append([ti, test_pred])
            elif tp!=pp and pp==2:
                ncFlist.append([ti, test_pred])

        if test_y[ti][1]==1:
            if tp==pp:
                cceTlist.append([ti, test_pred])
            elif tp!=pp and pp==2:
                cceFlist.append([ti, test_pred])

        if test_y[ti][2]==1:
            if tp==pp:
                ccmTlist.append([ti, test_pred])
            elif tp!=pp and pp==2:
                ccmFlist.append([ti, test_pred])

    print("NC True list ==========")
    print(ncTlist[:20])

    print("NC False list ==========")
    print(ncFlist[:20])

    print("CCNuE True list ==========")
    print(cceTlist[:20])

    print("CCNuE False list ==========")
    print(cceFlist[:20])

    print("CCNuMu True list ==========")
    print(ccmTlist[:20])

    print("CCNuMu False list ==========")
    print(ccmFlist[:20])
    """

    summary_writer.close()
    #summary_test_writer.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.

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

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=['NC','CCNuE','CCNuMu'],
                      title='Confusion matrix, without normalization')

plt.savefig("unnormalized_confusion_test_sample_new.png")

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=['NC','CCNuE','CCNuMu'], normalize=True,
                      title='Confusion matrix, with normalization')

plt.savefig("normalized_confusion_test_sample_new.png")

plt.show()

