import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.DEBUG)

MODEL_DIR = '/tmp/dynamic_inout'
NUM_SAMPLE = 10000000
BATCH_SIZE = 100000
NUM_INPUT = 3
NUM_OUTPUT = 2
INPUT_DIMS = np.random.randint(low=5, high=10, size=NUM_INPUT)
OUTPUT_CLASSES = np.random.randint(low=3, high=8, size=NUM_OUTPUT)

def rand_data_generator(num_sample, num_input, num_output, input_dims, output_classes):
    x, y = dict(), dict()
    for i in range(0, num_input):
        x[i] = np.random.rand(num_sample, input_dims[i]).astype(np.float32)
    for i in range(0, num_output):
        y[i] = np.random.randint(0, output_classes[i], size=num_sample)
    for i in range(0, num_sample):
        inputs, outputs = [], []
        for j in range(0, num_input):
            inputs.append(x[j][i])
        for j in range(0, num_output):
            outputs.append(y[j][i])
        yield inputs, outputs


def batch_iter(data_generator, batch_size=10):
    in_batch, out_batch = [], []
    for i, (inputs, outputs) in enumerate(data_generator, 1):
        in_batch.append(inputs)
        out_batch.append(outputs)
        if i % batch_size == 0:
            yield in_batch, out_batch
            del in_batch[:]
            del out_batch[:]
    if in_batch and out_batch:
        yield in_batch, out_batch
        del in_batch[:]
        del out_batch[:]

class Dataset:

    def __init__(self, data_generator, batch_size, num_input, num_output, input_dims, output_classes):
        self.data = data_generator
        self.fetch_fn = batch_iter(self.data, batch_size).__next__
        self._batch_size = batch_size
        self.num_input = num_input
        self.num_output = num_output
        self.input_dims = input_dims
        self.output_classes = output_classes

    def input_fn(self):
        out_types = [tf.float32] * self.num_input + [tf.int32] * self.num_output
        data = tf.py_func(func=self.input_fn_np, inp=[], Tout=out_types, stateful=True)

        input_dict, output_dict = dict(), dict()
        for i, _input in enumerate(data[0:self.num_input]):
            _input.set_shape([None, self.input_dims[i]])
            input_dict['in_{}'.format(i)] = _input
        for i, _output in enumerate(data[self.num_input:]):
            _output.set_shape([None])
            output_dict['out_{}'.format(i)] = _output
        return input_dict, output_dict

    def input_fn_np(self):
        inputs, outputs = self.fetch_fn()
        in_batch = np.array(inputs, dtype=object)
        out_batch = np.array(outputs, dtype=object)
        np_inputs, np_outputs = [], []
        for i in range(0, self.num_input):
            np_inputs.append(np.array(in_batch[:,i].tolist(), dtype=np.float32))
        for i in range(0, self.num_output):
            np_outputs.append(np.array(out_batch[:,i].tolist(), dtype=np.int32))
        return np_inputs + np_outputs


def model_fn(features, labels, mode, params):
    output_classes = params['output_classes']

    feature_embeds = []
    for i, (k, v) in enumerate(features.items()):
        # pylint: disable=not-context-manager
        with tf.variable_scope('encoder_{}'.format(k)):
        # pylint: enable=not-context-manager
            #v = tf.Print(v, [i, v], name="feature_{}".format(k), summarize=100)
            feature_embed = tf.layers.dense(v, units=10, name='dense_{}'.format(k))
        feature_embeds.append(feature_embed)

    all_feature_embed = tf.concat(feature_embeds, axis=1)

    logits = []
    for i, (k, _) in enumerate(labels.items()):
        # pylint: disable=not-context-manager
        with tf.variable_scope('decoder_{}'.format(k)):
        # pylint: enable=not-context-manager
            logit = tf.layers.dense(all_feature_embed, units=output_classes[i], name='dense_{}'.format(k))
        logits.append(logit)

    predictions, eval_metric_ops = {}, {}
    for i, (k, v) in enumerate(labels.items()):
        #v = tf.Print(v, [v], name="label_{}".format(k), summarize=100)
        onehot_label = tf.one_hot(v, output_classes[i])
        #logits[i] = tf.Print(logits[i], [logits[i]], name="logits_{}".format(k), summarize=100)
        #onehot_label = tf.Print(onehot_label, [onehot_label], name="onehot_label_{}".format(k), summarize=100)
        loss = tf.losses.softmax_cross_entropy(logits=logits[i], onehot_labels=onehot_label)
        tf.summary.scalar('training_loss_{0}'.format(k), loss)
        predicted_label = tf.argmax(logits[i], 1)
        predictions['classes_{0}'.format(k)] = predicted_label
        predictions['probabilities_{0}'.format(k)] = tf.nn.softmax(logits[i])
        predictions['logits_{0}'.format(k)] = logits[i]
        eval_metric_ops['accuracy_{0}'.format(k)] = tf.metrics.accuracy(v, predicted_label)
        eval_metric_ops['precision_{0}'.format(k)] = tf.metrics.precision(v, predicted_label)
        eval_metric_ops['recall_{0}'] = tf.metrics.recall(v, predicted_label)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    tf.summary.scalar('training_total_loss', total_loss)
    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    print([x.name for x in tf.global_variables()])

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

        
estimator = tf.contrib.learn.Estimator(
    model_fn = model_fn,
    model_dir = MODEL_DIR,
    config = tf.contrib.learn.RunConfig(
        save_checkpoints_steps = 10,
        save_summary_steps = 10,
        save_checkpoints_secs = None
    ),
    params={'output_classes': OUTPUT_CLASSES}
)

d = rand_data_generator(NUM_SAMPLE, NUM_INPUT, NUM_OUTPUT, INPUT_DIMS, OUTPUT_CLASSES)
ds = Dataset(d, BATCH_SIZE, NUM_INPUT, NUM_OUTPUT, INPUT_DIMS, OUTPUT_CLASSES)
estimator.fit(input_fn=ds.input_fn)
