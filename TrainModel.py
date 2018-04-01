from Model import *

def AccuracyShow(raw_lbel, predict_label):
    print("accuracy : ", np.mean(raw_lbel==predict_label))
    matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            matrix[i][j] = np.sum(predict_label[raw_lbel==j]==i)
    print(matrix)

kwsdata = KWSReader()

model = Model()
model.Model_Init()

for e in range(300001):
    train_data, train_label = kwsdata.TrainBatch()
    #print("train data batch", train_data.shape, train_label.shape)
    if e < 1000:
        lr = 0.01
    elif e >= 1000 and e <30000:
        lr = 0.005
    elif e >=30000 and e < 100000:
        lr = 0.001
    elif e >=100000 and e < 200000:
        lr = 0.0005
    else:
        lr = 0.0002
    _, l = model.sess.run([model.cls_optimizer, model.classify_loss],
                          feed_dict={
                              model.input : train_data,
                              model.label : train_label,
                              model.train : np.True_,
                              model.cls_lr : lr
                          })
    print(e, l)
    if e % 10 == 0:
        print(" accuracy in train set ")
        lbl_pld = model.sess.run(model.predict, feed_dict={
            model.input : train_data,
            model.train : np.False_
        })
        AccuracyShow(train_label, lbl_pld)
    if e % 25 == 0:
        print("accuracy in val set")
        val_data, val_label = kwsdata.ValBatch()
        lbl_pld = model.sess.run(model.predict, feed_dict={
            model.input: val_data,
            model.train : np.False_
        })
        AccuracyShow(val_label, lbl_pld)

model.Model_Close()