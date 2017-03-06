import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from .log_util import configure_logger, suppress_stdout
from .tools.freeze_graph import freeze_graph

logger = logging.getLogger("tensorplay.multigraph")

ROOT_DIR = '/tmp/multigraph'
SUM_MODEL_DIR = ROOT_DIR + '/sum'
FINAL_MODEL_DIR = ROOT_DIR + '/final'
SUM_MODEL_META_GRAPH_PATH = SUM_MODEL_DIR + '/model.pb'
SUM_MODEL_CHECKPOINT_PATH = SUM_MODEL_DIR + '/model.ckpt'
SUM_FROZEN_MODEL_PATH = SUM_MODEL_DIR + '/frozen_model.ckpt'
FINAL_MODEL_MODEL_PATH = FINAL_MODEL_DIR + '/model.ckpt'


def print_graph_ops(name, graph):
    logger.info(name + " has the following ops:")
    for op in graph.get_operations():
        logger.info('  - ' + op.name)


def print_graph_tensors(name, graph):
    logger.info(name + " has the following tensors:")
    for t in graph.as_graph_def().node:
        logger.info('  - ' + t.name)


def init_sum_model(inputs):
    with tf.name_scope('sum'):
        x = tf.placeholder(tf.int16, shape=[None, 5], name='input')
        zero = tf.get_variable(shape=[5], initializer=tf.constant_initializer(0), name='zero')
        x = x + tf.cast(zero, dtype=tf.int16)
        y_ = tf.reduce_sum(x, axis=1, name='output')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        os.makedirs(SUM_MODEL_DIR, exist_ok=True)
        tf.train.write_graph(sess.graph_def, '', SUM_MODEL_META_GRAPH_PATH, as_text=False)
        tf.train.Saver().save(sess, save_path=SUM_MODEL_CHECKPOINT_PATH)
        tf.summary.FileWriter(SUM_MODEL_DIR + '/summary', sess.graph)

    # Suppress the stdout print out
    with suppress_stdout():
        freeze_graph(input_graph=SUM_MODEL_META_GRAPH_PATH,
                     input_checkpoint=SUM_MODEL_CHECKPOINT_PATH,
                     output_graph=SUM_FROZEN_MODEL_PATH,
                     output_node_names='sum/output',
                     input_binary=True,
                     clear_devices=True,
                     input_saver=None,
                     initializer_nodes=None,
                     filename_tensor_name=None,
                     restore_op_name='save/restore_all')

    with tf.gfile.GFile(SUM_FROZEN_MODEL_PATH, "rb") as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            name="FROZEN",
                            op_dict=None,
                            producer_op_list=None)

    return graph


def double_model(x=None):
    with tf.name_scope('double'):
        if x is None:
            x = tf.placeholder(tf.int16, name='input', shape=[None])
        two = tf.get_variable(shape=[4], initializer=tf.constant_initializer(2), name='two', dtype=tf.int16)
        y_ = x * two
    return x, y_


def single_session():
    inputs = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8]
    ])

    sum_graph = init_sum_model(inputs)
    x1 = sum_graph.get_tensor_by_name('FROZEN/sum/input:0')
    y1 = sum_graph.get_tensor_by_name('FROZEN/sum/output:0')

    with tf.Session(graph=sum_graph) as sess:
        _, y2 = double_model(y1)
        sess.run(tf.global_variables_initializer())
        y1_outs, y2_outs = sess.run([y1, y2], feed_dict={x1: inputs})
        os.makedirs(FINAL_MODEL_DIR + '/single', exist_ok=True)
        tf.train.Saver().save(sess, save_path=FINAL_MODEL_DIR + '/single/model.ckpt')
        tf.summary.FileWriter(FINAL_MODEL_DIR + '/single/summary', sess.graph)
        print_graph_ops('Single session graph', sess.graph)

    assert y1_outs.tolist() == [15, 20, 25, 30]
    assert y2_outs.tolist() == [30, 40, 50, 60]


def multi_session():
    inputs = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8]
    ])

    sum_graph = init_sum_model(inputs)
    x1 = sum_graph.get_tensor_by_name('FROZEN/sum/input:0')
    y1 = sum_graph.get_tensor_by_name('FROZEN/sum/output:0')

    with tf.Session(graph=sum_graph) as sess:
        y1_outs = sess.run(y1, feed_dict={x1: inputs})
        print_graph_ops('Frozen graph', sess.graph)

    with tf.Graph().as_default() as another_graph:
        x2, y2 = double_model()
        with tf.Session(graph=another_graph) as another_sess:
            another_sess.run(tf.global_variables_initializer())
            y2_outs = another_sess.run(y2, feed_dict={x2: y1_outs})
            os.makedirs(FINAL_MODEL_DIR + '/multi', exist_ok=True)
            tf.train.Saver().save(another_sess, save_path=FINAL_MODEL_DIR + '/multi/model.ckpt')
            tf.summary.FileWriter(FINAL_MODEL_DIR + '/multi/summary', another_sess.graph)
            print_graph_ops('Final graph', another_sess.graph)


    assert y1_outs.tolist() == [15, 20, 25, 30]
    assert y2_outs.tolist() == [30, 40, 50, 60]


if __name__ == '__main__':
    configure_logger()
    logger.info("Running single session...")
    single_session()
    tf.reset_default_graph()
    logger.info("Running multi session...")
    multi_session()
