import tensorflow as tf 
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './checkpoints_1/test_refine_total_0',
                            """dir to store trained net""")
tf.app.flags.DEFINE_string('train_file_dir','/scratch/ys1297/LSTM_tracking/data_origin/train_list_0.txt', """dir to read trainlist""")
tf.app.flags.DEFINE_string('test_file_dir','/scratch/ys1297/LSTM_tracking/data_origin/test_list_0.txt', """dir to read trainlist""")
tf.app.flags.DEFINE_integer('length', 32,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('max_step', 10000000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .0001,
                            """for dropout""")
tf.app.flags.DEFINE_float('clip', .0001,
                            """for norm clip""")
tf.app.flags.DEFINE_float('epsilon', .000001,
                            """epsilon for small division""")
tf.app.flags.DEFINE_float('momentum', .9,
                            """momentum""")
tf.app.flags.DEFINE_float('l2_decay', .0001,
                            """for norm clip""")
tf.app.flags.DEFINE_float('eps_decay_factor', .9,
                            """for norm clip""")
tf.app.flags.DEFINE_float('eps_decay_after', 10000,
                            """for norm clip""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
