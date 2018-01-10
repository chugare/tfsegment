# coding:utf-8
import tensorflow as tf
import os
import jieba
import numpy as np
READ_LENGTH = 10
BATCHSIZE = 100
VEC_SIZE = 1000
KERNEL_WIDTH = 20
H1SIZE = 300
H2SIZE = 200
STDDEV = 0.04
MAX_STEP = 100000
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001
ULSW = [u' ',u'\r',u'\n']
LOG_FREQUENCY = 10
CHECKPOINT_DIR = 'tmp/ckp/'
EMA_DECAY = 0.001
def get_variable(name,shape,initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer=initializer)
    return var

def weight_decay_var(name,shape,stddev,wd):
    initializer = tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32)
    var = get_variable(name,shape,initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="weight_loss")
        tf.add_to_collection('losses',weight_decay)
        return var
def input(logits,labels):
    # logits,labels = tf.train.shuffle_batch([logits,labels],BATCHSIZE,num_threads=4,capacity=5000,min_after_dequeue=1000)
    return logits,labels
def interface (logits):
    with tf.variable_scope("cov") as scope:
        print logits.shape
        logits =tf.reshape(logits,[-1,KERNEL_WIDTH,VEC_SIZE,1])
        kernel = weight_decay_var('conv1',[KERNEL_WIDTH,1,1,1],stddev=STDDEV,wd=0.0)
        conv = tf.nn.conv2d(logits,kernel,[1,1,1,1],padding='VALID')
        bias = get_variable('bia_conv',[1],tf.constant_initializer(0.0))
        prea = tf.nn.bias_add(conv,bias)
        conv1 = tf.nn.relu(prea,name='conv1-relu')
        conv1 = tf.reshape(conv1,[-1,VEC_SIZE])
    with tf.variable_scope("Hidden1") as scope:
        hid = weight_decay_var(name='hidden1',shape=[VEC_SIZE,H1SIZE],stddev=STDDEV,wd=WEIGHT_DECAY)
        bias = get_variable(name='hidden1_bias',shape=[H1SIZE],initializer=tf.constant_initializer(0.1))
        h1 = tf.nn.relu(tf.matmul(conv1,hid)+bias,name=scope.name)
    with tf.variable_scope("Hidden2") as scope:
        hid2 = weight_decay_var(name='hidden2',shape=[H1SIZE,H2SIZE],stddev=STDDEV,wd=WEIGHT_DECAY)
        bias2 = get_variable(name='hidden1_bias',shape=[H2SIZE],initializer=tf.constant_initializer(0.1))
        h2=tf.nn.relu(tf.matmul(h1,hid2)+bias2,name=scope.name)
        print  h2.shape
    with tf.variable_scope("softmax_linear") as scope:

        sfw = weight_decay_var(name='Logit',shape=[H2SIZE,1],stddev=STDDEV,wd=WEIGHT_DECAY)
        sfbias = get_variable(name='Logit_bias',shape=[1],initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(h2,sfw),sfbias,name='softmax')
        print softmax_linear.shape
    return softmax_linear
def loss(logits,label):
    label = tf.cast(label,tf.float32)
    label = tf.reshape(label,shape=[BATCHSIZE,1])
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=logits,name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')
def calcul_loss(total_loss):
    loss = tf.train.ExponentialMovingAverage(0.9,name='mvg')
    losses = tf.get_collection('losses')
    return total_loss

def train(totalloss,step):
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    grads = opt.compute_gradients(totalloss)
    app = opt.apply_gradients(grads_and_vars=grads,global_step=step)
    with tf.control_dependencies([app]):
        train = tf.no_op(name='train')
    return train


def preprocess():
    txt = open('text','r')
    seg = open('seg.txt','w')
    for line in txt:
        line = line.decode('utf-8')
        seg.write(" ".join(jieba.lcut(line)).encode('utf-8'))
def read_dict():
    try:
        ldc_file = open('dict.txt','r')
    except IOError:
        ldc_file = open('dict.txt','w')
        ldc_file.close()
        return {}
    l_dic = {}
    for line in ldc_file:
        line = line.decode('utf-8');
        turple = line.split(' ')
        l_dic[turple[0]] = int(turple[1])
    ldc_file.close()
    return l_dic
def write_dict(name,index):
    ldc_file = open('dict.txt', 'a')
    ldc_file.write(name.encode('utf-8')+' '+str(index)+'\n');
def one_hot_encode(file):
    infile = open(name=file,mode='r')


    letter_dict = read_dict()
    count = len(letter_dict)
    storeF = []
    storeB = []
    for line in infile:
        line = line.decode('utf-8')
        c_pointer = 0
        f_pointer = 0
        for letter in line:
            if letter not in ULSW:
                if letter not in letter_dict:
                    letter_dict[letter] = count
                    write_dict(letter, count)
                    count += 1
                one_hot_vec = np.zeros((VEC_SIZE))
                one_hot_vec[letter_dict[letter]] = 1
                storeB.append(one_hot_vec)
                # storeB.append(letter)
            f_pointer += 1
            while f_pointer >= len(line) or len(storeB)>READ_LENGTH:

                if c_pointer >= len(line):
                    break
                if line[c_pointer] in ULSW:
                    c_pointer += 1
                    continue
                if len(storeF)==READ_LENGTH:
                    storeF.remove(storeF[0])
                storeF.append(storeB[0])
                storeB.remove(storeB[0])
                c_pointer+=1
                flag = 0
                if c_pointer<len(line):
                    if line[c_pointer] in ULSW:
                        flag = 1
                # print '|'.join(storeF)+flag+'|'.join(storeB)
                barray = np.array([])
                if len(storeB)<READ_LENGTH:
                    barray = np.zeros((READ_LENGTH - len(storeB), VEC_SIZE))
                for vec in storeB:
                    barray = np.append(barray,vec)
                farray = np.array([])
                for vec in storeF:
                    farray = np.append(farray,vec)
                if len(storeF) < READ_LENGTH:
                    farray = np.append(farray,np.zeros((READ_LENGTH - len(storeF), VEC_SIZE)))
                # barray = barray.reshape([10,VEC_SIZE])
                # farray = farray.reshape([10,VEC_SIZE])
                logits = np.append(barray,farray).reshape([READ_LENGTH*2,VEC_SIZE])

                yield (logits,flag)


def run_train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0,name='global_step',trainable=False)
        oni = tf.placeholder(dtype=tf.float32,shape=[BATCHSIZE,READ_LENGTH*2,VEC_SIZE],name='one-hot-input')
        onl = tf.placeholder(dtype=tf.int32,shape=(BATCHSIZE),name='label')
        logits,labels = input(oni,onl)
        sft = interface(logits)
        y_ = tf.nn.sigmoid(sft,name='result')
        pre =tf.floor(y_*2)
        totalloss = loss(sft,labels)
        train_op = train(totalloss,global_step)
        class log_hook(tf.train.SessionRunHook):
            def begin(self):
                self._step=0
                pass
            def before_run(self,run_context):
                self._step+=1
                return tf.train.SessionRunArgs(totalloss)
                pass

            def after_run(self, run_context, run_values):
                if self._step%LOG_FREQUENCY ==0:
                    out = run_values.results
                    print 'Step %d Output value: %f'%(self._step,out);
        class pre_log_hook(tf.train.SessionRunHook):

            def after_run(self, run_context, run_values):
                if self._step%10 ==0:
                    out = run_values.results
                    print '\t '


            def before_run(self, run_context):
                self._step+=1
                return  tf.train.SessionRunArgs(y_)
            def end(self, session):
                pass

            def begin(self):
                self._step = 0

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir='tmp/ckp/',
            hooks=[tf.train.StopAtStepHook(num_steps=MAX_STEP)
                   ,log_hook()
                   ]
        ) as monss:
            gl_count = 1
            segfile = one_hot_encode('seg_train.txt')
            seg_test = one_hot_encode('seg_test.txt')

            while not monss.should_stop():
                c = BATCHSIZE
                oni_batch = []
                onl_batch = []
                if gl_count % 10 != 0:
                    while c>0:
                        try:
                            o_h_e, seg_label = segfile.next()
                        except StopIteration:
                            segfile = one_hot_encode('seg_train.txt')
                        oni_batch.append(o_h_e)
                        onl_batch.append(seg_label)
                        c-=1
                    monss.run(train_op, feed_dict={oni: oni_batch, onl: onl_batch})
                else:
                    while c > 0:
                        try:
                            o_h_e, seg_label = segfile.next()
                        except StopIteration:
                            segfile = one_hot_encode('seg_test.txt')
                        oni_batch.append(o_h_e)
                        onl_batch.append(seg_label)
                        c -= 1
                    res = monss.run(pre, feed_dict={oni: oni_batch, onl: onl_batch})
                    pre_count = 0.0
                    for i in range(BATCHSIZE) :
                        if res[i][0] == onl_batch[i]:
                            pre_count+=1.0;

                    print "Global step %d , precision: %f"%(gl_count,pre_count/BATCHSIZE)
                gl_count += 1


run_train()
def run_seg(filename):
    with tf.Graph().as_default() as g:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        oni = tf.placeholder(dtype=tf.float32, shape=[BATCHSIZE, READ_LENGTH * 2, VEC_SIZE], name='one-hot-input')
        ema = tf.train.ExponentialMovingAverage()
        cal_label = interface(oni)
        y_ = tf.nn.sigmoid(cal_label, name='result')
        pre = tf.floor(y_ * 2)
    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
