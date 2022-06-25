import tensorflow as tf
import test_data_process
import numpy as np
import os
import argparse
import pandas as pd

slim = tf.contrib.slim



# Create FLAGS
def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='The batch size to use.')
    parser.add_argument('--sequence_length', type=int, default=80,
                        help='the sequence length: how many consecutive frames to use for the RNN; '
                            'if the network is only CNN then put here any number you want : total_batch_size = batch_size * seq_length')
    parser.add_argument('--image_size', type=int, default=96, 
                        help='dimensions of input images, e.g. 96x96')
    parser.add_argument('--network', type=str, default='affwildnet_resnet',
                        choices=['vggface_4096', 'vggface_2000', 'affwildnet_vggface', 'affwildnet_resnet'],
                        help='which network architecture we want to use')
    parser.add_argument('--input_file', type=str, default='./images_align/zf',
                        help='the input file')
    parser.add_argument('--output_dir', type=str, default='D:/Project/child_eyetrace/emo_result/group1')
    parser.add_argument('--pretrained_model_checkpoint_path', type=str, default='./models/resnet_rnn/model.ckpt-0',
                        help='the pretrained model checkpoint path to restore,if there exists one')

    args = parser.parse_args()

    return args


def predict_va(args):
    g = tf.Graph()
    with g.as_default():

        image_list = []
        img_ls = os.listdir(args.input_file)
        img_ls.sort(key=lambda x: int(x.split('.')[0]))
        for image_name in img_ls:
            image_dir = os.path.join(args.input_file, image_name)
            image_list.append(image_dir)
        # split into sequences, note: in the cnn models case this is splitting into batches of length: seq_length ;
        #                             for the cnn-rnn models case, I do not check whether the images in a sequence are consecutive or the images are from the same video/the images are displaying the same person
        image_list = test_data_process.make_rnn_input_per_seq_length_size(image_list, args.sequence_length)
        img_ls_select = np.array(img_ls[0:len(img_ls) // args.sequence_length * args.sequence_length])

        images = tf.convert_to_tensor(image_list)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images], num_epochs=None, shuffle=False, seed=None,
                                                    capacity=1000, shared_name=None, name=None)
        images_batch = test_data_process.decodeRGB(input_queue, args.sequence_length, args.image_size)
        images_batch = tf.to_float(images_batch)
        images_batch -= 128.0
        images_batch /= 128.0  # scale all pixel values in range: [-1,1]

        images_batch = tf.reshape(images_batch, [-1, 96, 96, 3])

        if args.network == 'vggface_4096':
            from vggface import vggface_4096x4096x2 as net
            network = net.VGGFace(args.batch_size * args.sequence_length)
            network.setup(images_batch)
            prediction = network.get_output()

        elif args.network == 'vggface_2000':
            from vggface import vggface_4096x2000x2 as net
            network = net.VGGFace(args.batch_size * args.sequence_length)
            network.setup(images_batch)
            prediction = network.get_output()

        elif args.network == 'affwildnet_resnet':
            from tensorflow.contrib.slim.python.slim.nets import resnet_v1
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, _ = resnet_v1.resnet_v1_50(inputs=images_batch, is_training=False, num_classes=None)

                with tf.variable_scope('rnn') as scope:
                    cnn = tf.reshape(net, [args.batch_size, args.sequence_length, -1])
                    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(128) for _ in range(2)])
                    outputs, _ = tf.nn.dynamic_rnn(cell, cnn, dtype=tf.float32)
                    outputs = tf.reshape(outputs, (args.batch_size * args.sequence_length, 128))

                    weights_initializer = tf.truncated_normal_initializer(
                        stddev=0.01)
                    weights = tf.get_variable('weights_output',
                                              shape=[128, 2],
                                              initializer=weights_initializer,
                                              trainable=True)
                    biases = tf.get_variable('biases_output',
                                             shape=[2],
                                             initializer=tf.zeros_initializer, trainable=True)

                    prediction = tf.nn.xw_plus_b(outputs, weights, biases)

        elif args.network == 'affwildnet_vggface':
            from affwildnet import vggface_gru as net
            network = net.VGGFace(args.batch_size, args.sequence_length)
            network.setup(images_batch)
            prediction = network.get_output()

        num_batches = int(len(image_list) / args.batch_size)

        variables_to_restore = tf.global_variables()

        with tf.Session() as sess:

            init_fn = slim.assign_from_checkpoint_fn(
                args.pretrained_model_checkpoint_path, variables_to_restore,
                ignore_missing_vars=False)

            init_fn(sess)
            print('Loading model {}'.format(args.pretrained_model_checkpoint_path))

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)



            evaluated_predictions = []

            try:
                for _ in range(num_batches):

                    pr = sess.run(prediction)
                    evaluated_predictions.append(pr)

                    if coord.should_stop():
                        break
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)

        predictions = np.reshape(evaluated_predictions, (-1, 2))

        return predictions, img_ls_select




if __name__ == '__main__':

    args = parsers()
    predictions, img_id = predict_va(args)
    save_data = pd.DataFrame({'img_id': img_id, 'valence': predictions[:, 0], 'arousal': predictions[:, 1]})
    save_path = os.path.join(args.output_dir, args.input_file.split('\\')[-1] + '.csv')
    save_data.to_csv(save_path, sep=' ', index=0)
