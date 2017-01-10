import pandas as pd
from networks_setups import *
from sklearn.metrics import confusion_matrix

filename = 'training_data.csv'
testname = 'facies_vectors.csv'

training_data = pd.read_csv(filename)
test_data = pd.read_csv(testname)

# Combine and shuffle our data
all_data = test_data.append(training_data)

np.random.seed(11)
rand_index = np.random.permutation(np.arange(all_data.shape[0]))

# Split train/test set
all_data, hot_vals = cleanup_csv(all_data)
dev_sample_index = -1 * int(TRAIN_RATIO * float(all_data.shape[0]))

labels_T = tf.convert_to_tensor(hot_vals[:dev_sample_index])
test_labels_T = tf.convert_to_tensor(hot_vals[dev_sample_index:])

# Output data one hot between 1-9. Facies
y_ = tf.placeholder(tf.float32, shape=[None, NUM_FACIES])

# network setup
y, x, features_T, test_features_T = two_layer_network(all_data, dev_sample_index)

# loss function used to train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# backprop
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# Accuracy calculations
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############################################## 2
labels_T2 = tf.convert_to_tensor(hot_vals[:dev_sample_index])
test_labels_T2 = tf.convert_to_tensor(hot_vals[dev_sample_index:])

y_2 = tf.placeholder(tf.float32, shape=[None, NUM_FACIES])

# network setup
y2, x2, features_T2, test_features_T2 = convolutional_network(all_data, dev_sample_index)

# loss function used to train
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2, y_2))

# backprop
train_step2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy2)

# Accuracy calculations
correct_prediction2 = tf.equal(tf.argmax(y2, 1), tf.argmax(y_2, 1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

# session init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(300000):
    x_vals, y_labels, x_vals_t, y_labels_t = sess.run([features_T, labels_T, test_features_T, test_labels_T])

    train_data = {x: x_vals, y_: y_labels}
    _, train_acc = sess.run([train_step, accuracy], feed_dict=train_data)

    test_data = {x: x_vals_t, y_: y_labels_t}
    test_acc = sess.run(accuracy, feed_dict=test_data)

    x_vals2, y_labels2, x_vals_t2, y_labels_t2 = sess.run([features_T2, labels_T2, test_features_T2, test_labels_T2])

    train_data2 = {x2: x_vals2, y_2: y_labels2}
    _, train_acc2 = sess.run([train_step2, accuracy2], feed_dict=train_data2)

    test_data2 = {x2: x_vals_t2, y_2: y_labels_t2}
    test_acc2 = sess.run(accuracy2, feed_dict=test_data2)

    if i % 1000 == 0:
        print('epoch', i / 1000)
        print('test acc', test_acc)
        print('train acc', train_acc, '\n')
        print('test acc d', test_acc2)
        print('train acc d', train_acc2, '\n')

print('test acc final', test_acc)
print('train acc final', sess.run(accuracy, feed_dict=train_data), '\n')

# predicted = sess.run(y_, feed_dict=train_data)
# conf = confusion_matrix(y_labels, predicted)
# facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS','WS', 'D','PS', 'BS']
# display_cm(conf, facies_labels, display_metrics=True, hide_zeros=True)

