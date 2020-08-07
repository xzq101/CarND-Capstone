import tensorflow as tf
import numpy as np
from utils import dataset_util
import yaml

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

DATADIR='test_images/sim_training_data/'

LABEL_DICT =  {
    "Red" : 0,
    "Yellow" : 1,
    "Green" : 2,
    "Undefined" : 4,
    }

def create_tf_example(label_and_data_info):
  height = 600 # Image height
  width = 800 # Image width
  filename = label_and_data_info['filename'] # Filename of the image. Empty if image is not from file

  with tf.gfile.GFile(DATADIR + filename, 'rb') as fid:
    encoded_image_data = fid.read()

  image_format = b'jpg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for box in label_and_data_info['annotations']:
        #if box['occluded'] is False:
        #print("adding box")
        xmins.append(float(box['xmin'] / width))
        xmaxs.append(float((box['xmin'] + box['x_width']) / width))
        ymins.append(float(box['ymin'] / height))
        ymaxs.append(float((box['ymin']+ box['y_height']) / height))
        classes_text.append(box['class'].encode())
        classes.append(int(LABEL_DICT[box['class']]))

  tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode()),
      'image/source_id': dataset_util.bytes_feature(filename.encode()),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_label_and_data


def main(_):

  file_loc = DATADIR + "sim_data_annotations.yaml"
  all_data_and_label_info = yaml.load(open(file_loc))

  np.random.seed(123)
  np.random.shuffle(all_data_and_label_info)
  num_train = int(len(all_data_and_label_info) * .8)
  train_data = all_data_and_label_info[:num_train]
  # test_data = all_data_and_label_info[num_train:]
  print ("The train set will be", num_train, "records long.")

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  for data_and_label_info in train_data:
    tf_example = create_tf_example(data_and_label_info)
    writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()

