from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')

import tensorflow as tf

import input_provider
from nn import NeuralNetwork as NN

max_step = 2000
step_to_report_loss = 100
step_to_eval = 1000
batch_size = 100
learning_rate = 0.01

nn_des = {'layer_description':[
				{	'name': 'input',
					'unit_size': 784,
				},
				{	'name': 'hidden1',
					'active_fun': tf.nn.relu,
					'unit_size': 128,
				},
				{	'name': 'output',
					'active_fun': None,
					'unit_size': 10, 
				},
			],
		}

data_sets = input_provider.read_data_sets('../data/mnist/')
nn_model = NN(nn_des) 

def fill_feed_dict(data_set, X_pl, Y_pl):
	images_feed, labels_feed = data_set.next_batch(batch_size)
	feed_dict = {
		X_pl: images_feed,
		Y_pl: labels_feed,
	}
	return feed_dict

def evaluate(sess, eval_op, X_pl, Y_pl, dataset):
	true_count = 0.0
	step_num = dataset.num_examples // batch_size
	num_examples = step_num*batch_size
	for step in range(step_num):
		feed = fill_feed_dict(dataset, X_pl, Y_pl)
		true_count += sess.run(eval_op, feed_dict=feed)
		precision = true_count / num_examples
	print('---Num examples: %d\tcorrect: %d  Precision @ 1: %0.04f'%(num_examples, true_count, precision))

with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, shape=(None, None))
    Y = tf.placeholder(tf.int32, shape=(None))

    predict_op = nn_model.inference(X)
    loss_op = nn_model.loss(predict_op, Y)
    train_op = nn_model.training(loss_op, learning_rate)
    eval_op = nn_model.evaluation(predict_op, Y)

    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    #tf.global_variables_initializer()
    
    for step in range(max_step):
        feed = fill_feed_dict(data_sets.train, X, Y)
        _, loss_value = sess.run([train_op, loss_op], feed_dict=feed)

        if step%step_to_report_loss==0 or step+1==max_step:
            print('Step %d\tloss: %0.3f'%(step, loss_value))
        if step%step_to_eval==0 or step+1==max_step:
            print('---Evaluate train set:')
            evaluate(sess, eval_op, X, Y, data_sets.train)
            print('---Evaluate valid set:')
            evaluate(sess, eval_op, X, Y, data_sets.validation)
            print('---Evaluate test set:')
            evaluate(sess, eval_op, X, Y, data_sets.test)
