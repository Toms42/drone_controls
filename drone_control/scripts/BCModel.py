import tensorflow as tf


# Original Source: http://hameddaily.blogspot.com/2017/09/imitation-learning-in-tensorflow-hopper.html
class BCModel(object):
    def __init__(self, log_name,
                 observation_dim,
                 action_dim,
                 learning_rate=0.001,
                 batch_size=64,
                 num_epochs=50,
                 checkpoint_path='./checkpoint/'):
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = num_epochs
        self.l2_loss = None
        self.optimizer = None
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.observations = None
        self.actions = None
        self.sess = tf.InteractiveSession()
        self.build_graph()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init_g)
        self.sess.run(init_l)

        self.checkpoint_path = checkpoint_path
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(log_name, graph=tf.get_default_graph())


    def build_graph(self):
        # (number of rows == number of training samples) is unknown, so None
        self.observations = tf.placeholder(shape=(None, self.observation_dim), dtype=tf.float32, name='observations')
        self.actions = tf.placeholder(shape=(None, self.action_dim), dtype=tf.float32, name='actions')

        # regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        # activator
        activator = tf.nn.tanh

        # layers
        W1 = tf.get_variable(shape=(self.observation_dim, 128),
                             regularizer=regularizer,
                             initializer=tf.contrib.layers.xavier_initializer(),
                             name='W1')
        # print(W1.get_shape())
        b1 = tf.get_variable(shape=(1, 128),
                         initializer=tf.contrib.layers.xavier_initializer(),
                         name='b1')
        logit1 = tf.matmul(self.observations, W1) + b1
        layer1 = activator(logit1, 'layer1')

        W2 = tf.get_variable(shape=(128, 64),
                             regularizer=regularizer,
                             initializer=tf.contrib.layers.xavier_initializer(),
                             name='W2')
        b2 = tf.get_variable(shape=(1, 64),
                         initializer=tf.contrib.layers.xavier_initializer(),
                         name='b2')
        logit2 = tf.matmul(layer1, W2) + b2
        layer2 = activator(logit2, 'layer2')

        x_reshape = tf.reshape(layer2, [-1, 64], name='reshape_input')
        x_split = tf.split(x_reshape, 64, 1, name='input_split')
        encoderCell = tf.contrib.rnn.BasicLSTMCell(32, activation=activator)
        lstm_outputs, state = tf.contrib.rnn.static_rnn(encoderCell, x_split, dtype=tf.float32)
        lstm_output = lstm_outputs[-1]

        W3 = tf.get_variable(shape=(32, self.action_dim),
                             regularizer=regularizer,
                             initializer=tf.contrib.layers.xavier_initializer(),
                             name='W3')
        b3 = tf.get_variable(shape=(1, self.action_dim),
                             initializer=tf.contrib.layers.xavier_initializer(),
                             name='b3')
        logit3 = tf.matmul(lstm_output, W3) + b3
        output = logit3
        output_action = tf.identity(output, 'output_action')

        # TODO: pure imitation to mimic expert actions, possibly modify loss function
        # as error from reference pose, would need to simulate result of output action of network
        # in simulator.. this would allow model to possibly outperform expert
        self.l2_loss = tf.losses.mean_squared_error(labels=self.actions, predictions=output)

        tf.summary.scalar("loss", self.l2_loss)
        grad_wrt_input = tf.gradients(self.l2_loss, self.observations)
        tf.summary.histogram("Input gradient", grad_wrt_input)
        grad_wrt_activations = tf.gradients(self.l2_loss, W1)
        tf.summary.histogram("W1 gradient", grad_wrt_activations)
        tf.summary.histogram("W1", W1)
        tf.summary.histogram("b1", b1)
        tf.summary.histogram("W3", W3)
        self.merged_summary_op = tf.summary.merge_all()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(self.l2_loss)

    def train(self, expert_data):
        num_points = len(expert_data['observations'])
        print("number of points= ", num_points)
        num_batches = int(num_points/self.BATCH_SIZE)
        for epoch in range(self.NUM_EPOCHS):
            for i in range(num_batches+1):
                observations_data = expert_data['observations'][i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
                actions_data = expert_data['actions'][i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]

                feed_data = {
                    self.observations: observations_data,
                    self.actions: actions_data
                }
                loss, _, summary = self.sess.run([self.l2_loss, self.optimizer, self.merged_summary_op], feed_dict=feed_data)
                print("epoch %d \t batch number %d \t loss is %f" % (epoch, i, loss))
                self.summary_writer.add_summary(summary, epoch * num_batches + i)

    def save(self, filename):
        self.saver.save(self.sess, self.checkpoint_path + filename)

    def restore(self, path='./checkpoint/bc-model.meta'):
        saver = tf.train.import_meta_graph(path)
        saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint/'))

    def init_infer(self, path):
        self.restore(path)
        self.graph = tf.get_default_graph()
        self.observations = self.graph.get_tensor_by_name('observations:0')
        self.output = self.graph.get_tensor_by_name('output_action:0')

    def infer(self, observation_data):
        """
        :param observation_data:
        :return:
        """
        return self.sess.run([self.output], feed_dict={
            self.observations: [observation_data]}
         )