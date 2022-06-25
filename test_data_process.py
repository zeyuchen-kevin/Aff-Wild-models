import tensorflow as tf

def decodeRGB(input_queue, seq_length, size=96):
    """ Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Three tensors: one with the decoded images, one with the corresponding labels and another with the image file locations
    """
    images = []

    for i in range(seq_length):
        file_content = tf.read_file(input_queue[0][i])
        image = tf.image.decode_jpeg(file_content, channels=3)
        image = tf.image.resize_images(image, tf.convert_to_tensor([size, size]))
        images.append(image)

    return images


def make_rnn_input_per_seq_length_size(images, seq_length):
    """
        Args:
        images : the images file locations with shape (N,1) where N is the total number of images
        labels: the corresponding labels with shape (N,2) where N is the total number of images
        seq_length: the sequence length that we want
        Returns:
        Two tensors: the images file locations with shape ( int(N/80),80 ) and corresponding labels with shape ( int(N/80),80,2 )
    """
    ims = []
    for l in range(int(len(images) / seq_length)):
        a = images[int(l) * seq_length:int(l) * seq_length + seq_length]
        ims.append(a)

    return ims
