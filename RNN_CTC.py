from GoogleCmdReader import *
import time
import tensorflow as tf

model_folder = 'save_model'
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
model_save_path = os.path.join(model_folder, 'mfcc_lstm_ctc.ckpt')

db = DataBase()
db.ReadAndTransfer()


def decode_a_seq(indexes, spars_tensor):
    decoded = ''
    for m in indexes:
        tmpi = spars_tensor[1][m]
        if tmpi in db.num_word_map:
            tmps = db.num_word_map[tmpi]
        elif tmpi == -1:
            tmps = ' '
        else:
            tmps = ' '
        decoded += tmps
    return decoded


def decode_sparse_tensor(sparse_tensor):
    # print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    # print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        # print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        # print(result)
    return result


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        for i in range(len(seq)):
            if seq[i] >= 0:
                indices.append([n, i])
                values.append(seq[i])
                # indices.extend(zip([n] * len(seq), range(len(seq))))
                # values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape


# type == 0 : train data
# type == 1 : val data
# type == 2 : test data
def Get_Batch(type):
    if type == 0:
        ret_idxs = np.random.choice(db.len_train_data, db.batch_size)
        ret_data = db.train_data[ret_idxs]
        ret_seqlens = db.train_seqlens[ret_idxs]
        ret_prelbls = db.train_label_vectors[ret_idxs]
    elif type == 1:
        ret_idxs = np.random.choice(db.len_val_data, db.batch_size)
        ret_data = db.val_data[ret_idxs]
        ret_seqlens = db.val_seqlens[ret_idxs]
        ret_prelbls = db.val_label_vectors[ret_idxs]
    else:
        ret_idxs = np.random.choice(db.len_test_data, db.batch_size)
        ret_data = db.test_data[ret_idxs]
        ret_seqlens = db.test_seqlens[ret_idxs]
        ret_prelbls = db.test_label_vectors[ret_idxs]
    ret_sparse_target = sparse_tuple_from(ret_prelbls)
    return ret_data, ret_sparse_target, ret_seqlens

def Match_Batch(type, i):
    if type == 1:
        anchor_data = db.val_data[db.val_label == i]
        anchor_seqlen = db.val_seqlens[db.val_label == i]
        anchor_labels = db.val_label_vectors[db.val_label == i]
        compare_data = db.val_data[db.val_label != i]
        compare_seqlen = db.val_seqlens[db.val_label != i]
        compare_labels = db.val_label_vectors[db.val_label != i]
    else:
        anchor_data = db.test_data[db.test_label == i]
        anchor_seqlen = db.test_seqlens[db.test_label == i]
        anchor_labels = db.test_label_vectors[db.test_label == i]
        compare_data = db.test_data[db.test_label != i]
        compare_seqlen = db.test_seqlens[db.test_label != i]
        compare_labels = db.test_label_vectors[db.test_label != i]
    al = len(anchor_data)
    cl = len(compare_data)
    anchor_sltidx = np.random.choice(al, db.batch_size)
    compare_sltidx = np.random.choice(cl, db.batch_size)
    return anchor_data[anchor_sltidx], anchor_seqlen[anchor_sltidx], anchor_labels[anchor_sltidx], \
           compare_data[compare_sltidx], compare_seqlen[compare_sltidx], compare_labels[compare_sltidx]


class RNN_CTC(object):
    def __init__(self, num_hidden=512):
        tf.reset_default_graph()
        self.num_hidden = num_hidden

        self.input_placeholder = tf.placeholder(tf.float32, [None, None, 64])
        self.targets = tf.sparse_placeholder(tf.int32)
        self.seq_lens = tf.placeholder(tf.int32, [None])
        self.n_classes = len(db.word_num_map) + 2
        # self.n_classes = 25

        self.logits = self.RNN_Module()

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.001, self.global_step, 1000, 0.9, staircase=True)
        self.loss = tf.nn.ctc_loss(labels=self.targets, inputs=self.logits, sequence_length=self.seq_lens)
        self.cost = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                           global_step=self.global_step)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_lens, merge_repeated=False)
        self.accuracy = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets))
        self.session = None
        self.val_edit_dist = 100
        self.saver = tf.train.Saver()

    def RNN_Module(self):
        cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
        outputs, states = tf.nn.dynamic_rnn(cell, self.input_placeholder, sequence_length=self.seq_lens,
                                            dtype=tf.float32)

        outputs = tf.reshape(outputs, [-1, self.num_hidden])
        logits = tf.layers.dense(outputs, self.n_classes)

        shape = tf.shape(self.input_placeholder)
        batch_s, max_timesteps = shape[0], shape[1]
        logits = tf.reshape(logits, [batch_s, -1, self.n_classes])
        logits = tf.transpose(logits, (1, 0, 2))

        return logits

    def report_accuracy(self, decoded_list, test_targets):
        original_list = decode_sparse_tensor(test_targets)
        detected_list = decode_sparse_tensor(decoded_list)
        true_numer = 0

        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, '<---------------------------------------------------------->')
            print("(", len(number), ")", number)
            print("(", len(detect_number), ")", detect_number)
            # print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def do_report(self, session, type):
        test_inputs, test_targets, test_seq_len = Get_Batch(type)
        test_feed = {self.input_placeholder: test_inputs,
                     self.targets: test_targets,
                     self.seq_lens: test_seq_len}
        dd, log_probs, accuracy = session.run([self.decoded[0], self.log_prob, self.accuracy], test_feed)
        self.report_accuracy(dd, test_targets)

    def do_batch(self, session, type):
        train_inputs, train_targets, train_seq_len = Get_Batch(0)
        feed = {self.input_placeholder: train_inputs,
                self.targets: train_targets,
                self.seq_lens: train_seq_len}
        b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
            [self.loss, self.targets, self.logits, self.seq_lens,
             self.cost, self.global_step, self.optimizer], feed)

        if steps % 100 == 0:
            print(b_cost, steps)
        if steps > 0 and steps % 250 == 0:
            print("validate data")
            self.do_report(session, 1)
        if steps > 0 and steps % 1000 == 0:
            print("test data")
            self.do_report(session, 2)

        return b_cost, steps

    def Model_Init(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())
        print("model initialize")
        print("_________________________________________________________________________________")

    def Model_Close(self):
        self.session.close()
        print(" This model has been shut down ")

    def Model_Save(self):
        print('save model at ', model_save_path)
        self.saver.save(self.session, model_save_path)

    def Model_Load(self):
        print("load this model ", model_save_path)
        self.saver.restore(self.session, model_save_path)

    def Model_Train(self):
        try:
            print("_____________________________________________________________________________")
            print("_____________________________________________________________________________")
            print(" model exists, load it")
            self.Model_Load()
            print("_____________________________________________________________________________")
            print("_____________________________________________________________________________")
        except:
            print("error happens, retrain model")
            self.Model_Init()
        start = time.time()
        for curr_epoch in range(100001):
            print("Epoch.......", curr_epoch)
            train_cost = train_ler = 0
            for batch in range(32):
                c, steps = self.do_batch(self.session, 0)
                train_cost += c * 16
                #print("Step:", steps, ", batch seconds:", seconds)

            train_cost /= (16 * db.batch_size)
            #print(" to next")
            if curr_epoch % 10 == 0:
                seconds = time.time() - start
                print("train time : ", seconds)
                train_inputs, train_targets, train_seq_len = Get_Batch(1)
                val_feed = {self.input_placeholder: train_inputs,
                            self.targets: train_targets,
                            self.seq_lens: train_seq_len}

                val_cost, val_ler, lr, steps = self.session.run([self.cost, self.accuracy,
                                                                 self.learning_rate,
                                                                 self.global_step],
                                                                feed_dict=val_feed)
                print("val get to next")
                log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
                print(
                    log.format(curr_epoch + 1, 1000001, steps, train_cost, train_ler, val_cost, val_ler,
                               time.time() - start,
                               lr))

                if curr_epoch % 1000 == 0:
                    self.val_edit_dist = val_ler
                    print("********************************************")
                    print("model save")
                    self.Model_Save()

    # make sure that every time, anchors and compares are all the same or the diff
    def Compare(self, anchor_data, anchor_seqlens, compare_data, compare_seqlens, anchor_labels=None, compare_labels=None):
        # totally the same
        anchor_numseq = self.session.run(tf.cast(self.decoded[0], tf.int32), feed_dict={
            self.input_placeholder : anchor_data,
            self.seq_lens : anchor_seqlens
        })
        compare_numseq = self.session.run(tf.cast(self.decoded[0], tf.int32), feed_dict={
            self.input_placeholder : compare_data,
            self.seq_lens : compare_seqlens
        })
        ac_edit_dist = self.session.run(tf.edit_distance(anchor_numseq, compare_numseq))
        anchor_decode = decode_sparse_tensor(anchor_numseq)
        compare_decode = decode_sparse_tensor(compare_numseq)
        #print('anchor  : ', anchor_decode)
        #print('compare : ', compare_decode)
        if anchor_decode[0] == compare_decode[0]:
            return ac_edit_dist, 0
        else:
            return ac_edit_dist, 1

    def ResultAnalysis(self, type=1):
        acc_same = []
        acc_diff = []
        ed_same = []
        ed_diff = []
        if type == 1:
            start_time = time.time()
            for x in set(db.val_label):
                this_data, this_seqlens, this_labelves, that_data, that_seqlens, that_labelvecs = Match_Batch(type, x)
                tl = len(this_data)
                # compute the same part
                count = 0
                for i in range(tl):
                    thisone_data = []
                    thisone_seqlen = []
                    thisone_data.append(this_data[i])
                    thisone_seqlen.append(this_seqlens[i])
                    for j in range(i+1, tl):
                        thatone_data = []
                        thatone_seqlen = []
                        thatone_data.append(this_data[j])
                        thatone_seqlen.append(this_seqlens[j])
                        aced, res = self.Compare(thisone_data, thisone_seqlen, thatone_data, thatone_seqlen)
                        ed_same.append(aced)
                        acc_same.append(res)
                        count += 1
                        if count % 256 == 0:
                            print(count, 'over')
                # compute the diff
                cl = len(that_data)
                count = 0
                for i in range(tl):
                    thisone_data = []
                    thisone_seqlen = []
                    thisone_data.append(this_data[i])
                    thisone_seqlen.append(this_seqlens[i])
                    for j in range(cl):
                        thatone_data = []
                        thatone_seqlen = []
                        thatone_data.append(that_data[j])
                        thatone_seqlen.append(that_seqlens[j])
                        aced, res = self.Compare(thisone_data, thisone_seqlen, thatone_data, thatone_seqlen)
                        ed_diff.append(aced)
                        acc_diff.append(res)
                        count += 1
                        if count % 256 == 0:
                            print(count, 'over')
                the_time = time.time()
                print(x, ' over ', the_time - start_time)
        print("totally same : ", 1 - np.mean(acc_same))
        print("totally diff : ", np.mean(acc_diff))
        print("distance judgement")
        same_mean = np.mean(ed_same)
        diff_mean = np.mean(ed_diff)
        ed_same = np.asarray(ed_same)
        ed_diff = np.asarray(ed_diff)
        mid_mean = (same_mean + diff_mean) / 2
        print(mid_mean)
        mid_mean = 0.65
        print("same dists ", same_mean, np.std(ed_same))
        print("diff dists ", diff_mean, np.std(ed_diff))
        print("same acc ", np.mean(ed_same <= mid_mean))
        print("diff acc ", np.mean(ed_diff > mid_mean))
        the_time = time.time()
        print(' over ', the_time - start_time)

model = RNN_CTC()
model.Model_Init()
model.Model_Load()
model.ResultAnalysis()
model.Model_Close()