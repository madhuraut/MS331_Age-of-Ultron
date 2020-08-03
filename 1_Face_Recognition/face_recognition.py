"""
code to train the SVM model to classify the input image (identify the person present)
Note: We are passing already face alligened images to train the SVM
The facece are alligened first by MTCNN then slitted in to train and test
The freatures are being extracted from pretrained Inception ResNet v1 with triplet Loss trained on
"""




from __future__ import print_function
from __future__ import absolute_import
from __future__ import division



import os
import sys
import tensorflow as tf
import numpy as np
import math
import pickle
import argparse
import facenet
from sklearn.svm import SVC


def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            #load data
            data = facenet.get_dataset(args.data_folder)

            # assert the need of at least one training image per class
            for cls in data:
                assert (len(cls.image_paths) > 0, 'there should be at least one image in the folder corresponding to person')

            paths, labels = facenet.get_image_paths_and_labels(data)

            print('Number of distinct person: %d' % len(data))
            print('Total number of images all classes: %d' % len(paths))

            # Load the pretrained FaceNet model for feature extraction
            print('Loading facenet CNN')
            facenet.load_model(args.model)

            # Define input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass theoufh FaceNet and get Feature vector
            print('Calculating feature vector for images')
            no_of_images = len(paths)
            no_of_batches_per_epoch = int(math.ceil(1.0 * no_of_images / args.batch_size))
            emb_array = np.zeros((no_of_images, embedding_size))
            for i in range(no_of_batches_per_epoch):
                start_indx = i * args.batch_size
                end_indx = min((i + 1) * args.batch_size, no_of_images)
                paths_batch = paths[start_indx:end_indx]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_indx:end_indx, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_format = os.path.expanduser(args.classifier_file_name)

            if (args.mode == 'train'):

                # Create a list of names of all peple present in dataset
                ppl_names = [cls.name.replace('_', ' ') for cls in data]

                # Train SVM classifier
                print('Training SVM classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
                print('SVM classifier trained')

                # Save trained classifier model
                with open(classifier_filename_format, 'wb') as outfile:
                    pickle.dump((model, ppl_names), outfile)
                print('Saved SVM classifier model to file "%s"' % classifier_filename_format)

            elif (args.mode == 'test'):
                # predict the person present in images
                print('Testing trained SVM classifier')
                with open(classifier_filename_format, 'rb') as infile:
                    (model, ppl_names) = pickle.load(infile)

                print('SVM classifier model loaded from file "%s"' % classifier_filename_format)

                pred = model.predict_proba(emb_array)
                best_class_indx = np.argmax(pred, axis=1)
                best_class_prob = pred[np.arange(len(best_class_indx)), best_class_indx]

                for i in range(len(best_class_indx)):
                    print('%4d  %s: %.3f' % (i, ppl_names[best_class_indx[i]], best_class_prob[i]))

                print('all images processed')

                accuracy = np.mean(np.equal(best_class_indx, labels))
                print('Accuracy obtained in testing: %.3f' % accuracy)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['train', 'test'],
                        help='state if want to train or test ' +
                             'model should be used for classification', default='test')
    parser.add_argument('data_folder', type=str,
                        help='Path to the folder containing aligned images.')
    parser.add_argument('model', type=str,
                        help='path to pretrained FaceNet')
    parser.add_argument('classifier_file_name',
                        help='path to output trained SVM classifire (.pkl) file. ' +
                             'For training its is output and input for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to be processed in each batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='(height, width) of image.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility.', default=666)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

    
