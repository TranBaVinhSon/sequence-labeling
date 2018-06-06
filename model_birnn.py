'''
About dataset:
 - 180 elements word -> id (and id -> word) (include padding element)
 - vocab_size: 180
 - label size: 33
 - Max length of a sentence (num_steps): 28
 -
'''
import tensorflow as tf
import numpy as np
import collections
from dataloader import DataLoader
import sys
import Helper

def load_data():
    data_loader = DataLoader("data_10k.txt")
    train_data, test_data = data_loader.load_data(number_data=10000, train_data=0.8, test_data=0.2)
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    return train_data, test_data, data_loader.vocab_size, \
            data_loader.sentence_max_len, data_loader.id_to_label, \
            data_loader.word_to_id, data_loader.label_size

# For input data
def batch_producer(data, batch_size, num_steps):
    data_len = len(data)
    data = tf.convert_to_tensor(data, tf.int32)
    batch_len = int(data_len // batch_size)

    i = tf.train.range_input_producer(batch_len, shuffle=False).dequeue()

    x = data[i * batch_size : (i+1) * batch_size, 0, : ]
    x.set_shape([batch_size, num_steps])

    y = data[i * batch_size : (i+1) * batch_size, 1, : ]
    y.set_shape([batch_size, num_steps])
    return x, y

class Config():
    learning_rate = 1.0
    num_layers = 1
    hidden_size = 60
    batch_size = 50
    num_epochs = 30
    max_lr_epoch = 5
    lr_decay = 0.8
    print_iter = 50
    logs_path = "/tmp/tensorflow_logs/gr_final_result/"
    model_path = "model/birnn_full_results_epoch_10k_60_hidden_size_50_batch_size/model.ckpt"
    result_path = "result/birnn_full_results_epoch_10k_60_hidden_size_50_batch_size.csv"

class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.batch_len = int(len(data) // batch_size)
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)

class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, label_size, num_layers, dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # create the word embeddings
        embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
        inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction
        self.init_fw_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])
        self.init_bw_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        init_fw_state_per_layer_list = tf.unstack(self.init_fw_state, axis=0)
        init_bw_state_per_layer_list = tf.unstack(self.init_bw_state, axis=0)

        fw_rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(init_fw_state_per_layer_list[idx][0], init_fw_state_per_layer_list[idx][1])
            for idx in range(num_layers)]
        )
        bw_rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(init_bw_state_per_layer_list[idx][0], init_bw_state_per_layer_list[idx][1])
            for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled
        fw_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        bw_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
        fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell for _ in range(num_layers)], state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell for _ in range(num_layers)], state_is_tuple=True)
        # bidirectional merge module fw + bw, output_state = [fw_state, bw_state]
        output, outputs_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                  inputs,
                                                  initial_state_fw=fw_rnn_tuple_state,
                                                  initial_state_bw=bw_rnn_tuple_state,
                                                  dtype=tf.float32)
        self.fw_state, self.bw_state = outputs_state
        output = tf.concat(output, 2)
        # reshape to (batch_size * num_steps, hidden_size * 2)
        output = tf.reshape(output, [-1, hidden_size * 2])
        softmax_w = tf.Variable(tf.random_uniform([2 * hidden_size, label_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([label_size], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, label_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            # just return tensor with all elements set to 1
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True
        )

        # Update the cost
        self.cost = tf.reduce_mean(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, label_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        correct_prediction_sequence = tf.reshape(correct_prediction, [self.batch_size, self.num_steps])
        sentence_compare = tf.reduce_min(tf.cast(correct_prediction_sequence, tf.float32), axis=1)
        self.sentence_accuracy = tf.reduce_mean(sentence_compare)

        if not is_training:
            return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        # for optimize leanring rate
        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

        # Create s summary to monitor cost, accuracy tensor
        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("sentence accuracy", self.sentence_accuracy)
        # Create summaries to visualize weights
        for var in tvars:
            tf.summary.histogram(var.name, var)
        # Summarize all gradients
        for grad, var in list(zip(grads, tvars)):
            tf.summary.histogram(var.name + "/gradient", grad)
        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

def train_model(train_data, vocabulary, label_size, num_steps, config):
    training_input = Input(config.batch_size, num_steps, data=train_data)
    model = Model(training_input, True, config.hidden_size, vocabulary, label_size, config.num_layers)
    init_op = tf.global_variables_initializer()
    orig_decay = config.lr_decay
    with tf.Session() as sess:
        sess.run([init_op])
        summary_writer = tf.summary.FileWriter(config.logs_path,
                                                graph=tf.get_default_graph())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver(max_to_keep=config.num_epochs)
        for epoch in range(config.num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - config.max_lr_epoch, 0.0)
            print("New lr decay: {}\n learning rate: {}".format(new_lr_decay, config.learning_rate * new_lr_decay))
            model.assign_lr(sess, config.learning_rate * new_lr_decay)
            fw_state = np.zeros((config.num_layers, 2, config.batch_size, model.hidden_size))
            bw_state = np.zeros((config.num_layers, 2, config.batch_size, model.hidden_size))
            for step in range(training_input.batch_len):
                if step % config.print_iter != 0:
                    cost, _, fw_state, bw_state, summary = sess.run([model.cost, model.train_op, model.fw_state, model.bw_state, model.merged_summary_op],
                                                        feed_dict={model.init_fw_state: fw_state, model.init_bw_state: bw_state})
                else:
                    cost, _, fw_state, bw_state, sentence_acc, acc, summary = sess.run([model.cost, model.train_op, model.fw_state, model.bw_state, model.sentence_accuracy, model.accuracy, model.merged_summary_op],
                                                            feed_dict={model.init_fw_state: fw_state, model.init_bw_state: bw_state})
                    print("Epoch {}, Step {}, Cost: {:.3f} Accuracy: {:.3f} Sentence_Acc: {:.6f}".format(epoch, step, cost, acc, sentence_acc))
                # Write logs at every iteration
                summary_writer.add_summary(summary, config.num_epochs * training_input.batch_len + step)
            # save a model checkpoint at each epoch
            saver.save(sess, config.model_path, global_step=epoch)
        # close threads
        coord.request_stop()
        coord.join(threads)

def test_model(model, test_data, id_to_label, num_steps, vocab_size, label_size, config, epoch):
    batch_len = int(len(test_data) // config.batch_size)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        fw_state = np.zeros((config.num_layers, 2, config.batch_size, model.hidden_size))
        bw_state = np.zeros((config.num_layers, 2, config.batch_size, model.hidden_size))
        # restore the trained model
        saver.restore(sess, config.model_path + "-" + str(epoch))
        # get an average accuracy over batch_len
        accuracy = 0
        sentence_accuracy = 0
        predicts = []
        for batch in range(batch_len):
            true_vals, pred, fw_state, bw_state, acc, sentence_acc = sess.run([model.input_obj.targets, model.predict, model.fw_state, model.bw_state, model.accuracy, model.sentence_accuracy],
                                                            feed_dict={model.init_fw_state: fw_state, model.init_bw_state: bw_state})

            pred = np.reshape(pred, [config.batch_size, num_steps])
            predicts.append(pred)
            accuracy += acc
            sentence_accuracy += sentence_acc

        predict_all = np.concatenate(predicts, axis=0)
        target_all = test_data[: config.batch_size * batch_len, 1, :]
        precision, recall, f1_score = f1(predict_all, target_all, num_steps, label_size)
        final_acc = accuracy / batch_len
        final_sentence_acc = sentence_accuracy / batch_len
        print("Average accuracy: {:.3f}".format(final_acc))
        print("Average Sentence accuracy: {:.3f} \n\n".format(final_sentence_acc))
        with open(config.result_path, "a") as f:
            f.write("{}, {:.3f}, {:.3f}, {:.3f}\n".format(epoch, final_acc, final_sentence_acc, f1_score))
        # close threads
        coord.request_stop()
        coord.join(threads)

def predict_model(query, word_to_id, id_to_label, config, num_steps, vocabulary, label_size):
    MAX_LENGTH = num_steps
    query_words = query.split(" ")
    length_of_query = len(query_words)
    batch_size = 1
    # add padding
    for i in range(MAX_LENGTH - length_of_query):
        query_words.append("<pad>")
    # word -> id
    query_ids = []
    for word in query_words:
        id = word_to_id["<u>"]
        if Helper.represents_int(word):
            id = word_to_id["<number>"]
        else:
            if word in word_to_id:
                id = word_to_id[word]
        query_ids.append(id)

    target = [0] * num_steps
    query_ids = [[query_ids, target]]
    pred_input = Input(batch_size, num_steps, query_ids)
    model = Model(pred_input, False, config.hidden_size, vocabulary, label_size, config.num_layers, dropout=1)
    # embedding
    saver = tf.train.Saver()
    prediction = []
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        fw_state = np.zeros((config.num_layers, 2, batch_size, model.hidden_size))
        bw_state = np.zeros((config.num_layers, 2, batch_size, model.hidden_size))
        # restore the trained model
        saver.restore(sess, config.model_path + "-" + str(config.num_epochs))
        # get an average accuracy over batch_len
        prediction = sess.run([model.predict], feed_dict={model.init_fw_state: fw_state, model.init_bw_state: bw_state})
        # close threads
        coord.request_stop()
        coord.join(threads)
    # because out input has just one sentence
    prediction = prediction[0][:length_of_query]
    result = []
    for i in prediction:
        if i in id_to_label:
            result.append(id_to_label[i])
        else:
            result.append("<u>")
    print("Query: {}\n".format(query.split(" ")))
    print("Label: {}".format(result))

def f1(prediction, target, max_length, label_size):
    # label_size is included padding element
    tp = np.array([0] * label_size) # true positive
    fp = np.array([0] * label_size) # false positive
    fn = np.array([0] * label_size) # false negative

    for i in range(len(target)):
        for j in range(max_length):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    UNLABLED = 0
    for i in range(label_size - 1):
        if i != UNLABLED:
            tp[label_size - 1] += tp[i]
            fp[label_size - 1] += fp[i]
            fn[label_size - 1] += fn[i]
    precision = []
    recall = []
    f1_score = []
    for i in range(label_size):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        f1_score.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print("Precision: {}\nRecall: {}\nF1 score: {}\n".format(precision[label_size - 1], recall[label_size - 1], f1_score[label_size - 1]))
    return precision[label_size - 1], recall[label_size - 1], f1_score[label_size - 1]

if __name__ == "__main__":
    train_data, test_data, vocabulary, num_steps, id_to_label, word_to_id, label_size = load_data()
    config = Config()
    if len(sys.argv) < 2:
        train_model(train_data, vocabulary, label_size, num_steps, config)
    else:
        # For prediction
        if sys.argv[1] in ["prediction", "predict", "pre"]:
            query = input("query: > ")
            predict_model(query, word_to_id, id_to_label, config, num_steps, vocabulary, label_size)
        # For Testing
        else:
            test_input = Input(config.batch_size, num_steps, test_data)

            model = Model(test_input, False, config.hidden_size, vocabulary, label_size, config.num_layers, dropout=1)
            with open(config.result_path, "a") as f:
                f.write("Epoch, Acc, Sentence Acc, F1 score\n")
            for i in range(config.num_epochs):
                test_model(model, test_data, id_to_label, num_steps, vocabulary, label_size, config, i)
