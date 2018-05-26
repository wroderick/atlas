import json
import logging
import os
import glob
from skimage import io
import sys
import tensorflow as tf
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy import ndimage

from split import setup_train_dev_split
from data_batcher import SliceBatchGenerator

# Relative path of the main directory
MAIN_DIR = os.path.relpath(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Relative path of the data directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data")

#Relative path of the output masked data directory
DEFAULT_OUTPUT_DATA_DIR = os.path.join(MAIN_DIR, "data_output_masks")

# Relative path of the experiments directory
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments")

# General
tf.app.flags.DEFINE_integer("batch_size", 100, "Sets the batch size.")
tf.app.flags.DEFINE_integer("eval_every", 500,
                            "How many iterations to do per calculating the "
                            "dice coefficient on the dev set. This operation "
                            "is time-consuming, so should not be done often.")
tf.app.flags.DEFINE_string("experiment_name", "",
                           "Creates a dir with this name in the experiments/ "
                           "directory, to which checkpoints and logs related "
                           "to this experiment will be saved.")
tf.app.flags.DEFINE_integer("gpu", 0,
                            "Sets which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_integer("keep", None,
                            "How many checkpoints to keep. None means keep "
                            "all. These files are storage-consuming so should "
                            "not be kept in aggregate.")
tf.app.flags.DEFINE_string("mode", "train",
                           "Options: {train,eval,save_output_masks}.")
tf.app.flags.DEFINE_integer("print_every", 1,
                            "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500,
                            "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("summary_every", 100,
                            "How many iterations to do per TensorBoard "
                            "summary write.")

# TensorBoard
tf.app.flags.DEFINE_integer("num_summary_images", 64,
                            "How many images to write to summary.")

# Data
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR,
                           "Sets the dir in which to find data for training. "
                           "Defaults to data/.")
tf.app.flags.DEFINE_string("output_data_dir", DEFAULT_OUTPUT_DATA_DIR,
                           "Sets the dir in which to find output masked data for training. "
                           "Defaults to data_output_masks/.")
tf.app.flags.DEFINE_string("input_regex", None,
                           "Sets the regex to use for input paths. If set, "
                           "{FLAGS.p} will be ignored and train and dev sets "
                           "will use this same input regex. Only works when "
                           "{FLAGS.split_type} is by_slice.")
tf.app.flags.DEFINE_boolean("merge_target_masks", True,
                            "Sets whether to merge target masks or not.")
tf.app.flags.DEFINE_boolean("use_fake_target_masks", False,
                            "Sets whether to use fake target masks or not.")
tf.app.flags.DEFINE_boolean("use_volumetric", False,
                            "Sets whether to use volumetric data or not.")
tf.app.flags.DEFINE_boolean("use_masked_train_set", False,
                            "Sets whether to use the masked output dataset for training.")
tf.app.flags.DEFINE_string("original_data_dir", DEFAULT_DATA_DIR,
                           "Sets the dir in which to find original train target dataset masks. "
                           "Defaults to data/.")

# Split
tf.app.flags.DEFINE_string("cv_type", "lpocv",
                           "Sets the type of cross validation. Options: "
                           "{lpocv,loocv}.")
tf.app.flags.DEFINE_integer("p", None,
                            "Sets p in leave-p-out cross-validation. Defaults "
                            "to floor(0.3 * n) where n represents the number "
                            "groups implied by {split_type}; e.g. n=220 for "
                            "by_patient, n=229 for by_scan, n=9 for by_site.")
tf.app.flags.DEFINE_string("split_type", "by_slice",
                           "Sets the type of split between the train and dev "
                           "sets i.e. whether certain slices or volumes of "
                           "slices must be part of the same split. Options: "
                           "{by_patient,by_scan,by_slice,by_site}. e.g. for "
                           "by_patient, slices from a given patient must be "
                           "either all part of the train split or all part "
                           "of the dev split.")

# Training
tf.app.flags.DEFINE_float("dropout", 0.15,
                          "Sets the fraction of units randomly dropped on "
                          "non-recurrent connections.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Sets the learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clips the gradients to this norm.")
tf.app.flags.DEFINE_float("pos_weight", 50.0,
                          "Allows tradeoff between recall and precision by"
                          "up or down weighting the cost of positive error"
                          "relative to a negative error.")
tf.app.flags.DEFINE_integer("num_epochs", None,
                            "Sets the number of epochs to train. None means "
                            "train indefinitely.")
tf.app.flags.DEFINE_string("train_dir", "",
                           "Sets the dir to which checkpoints and logs will "
                           "be saved. Defaults to "
                           "experiments/{experiment_name}.")

# Dev
tf.app.flags.DEFINE_integer("dev_num_samples", None,
                            "Sets the number of samples to evaluate from the "
                            "dev set. None means evaluate on all.")

# Model
tf.app.flags.DEFINE_string("model_name", "ATLASModel",
                           "Sets the name of the model to use; the name must "
                           "correspond to the name of a class defined in "
                           "atlas_model.py.")
tf.app.flags.DEFINE_integer("slice_height", 232, "Sets the image height.")
tf.app.flags.DEFINE_integer("slice_width", 196, "Sets the image width.")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(sess, model, train_dir, expect_exists=False):
  """
  Initializes the model from {train_dir}.

  Inputs:
  - sess: A TensorFlow Session object.
  - model: An ATLASModel object.
  - train_dir: A Python str that represents the relative path to train dir
    e.g. "../experiments/001".
  - expect_exists: If True, throw an error if no checkpoint is found;
    otherwise, initialize fresh model if no checkpoint is found.
  """
  ckpt = tf.train.get_checkpoint_state(train_dir)
  v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
  if (ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path)
      or tf.gfile.Exists(v2_path))):
    print(f"Reading model parameters from {ckpt.model_checkpoint_path}")
    model.saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    if expect_exists:
      raise Exception(f"There is no saved checkpoint at {train_dir}")
    else:
      print(f"There is no saved checkpoint at {train_dir}. Creating model "
            f"with fresh parameters.")
      sess.run(tf.global_variables_initializer())


def main(_):
  #############################################################################
  # Configuration                                                             #
  #############################################################################
  # Checks for Python 3.6
  if sys.version_info[0] != 3:
    raise Exception(f"ERROR: You must use Python 3.6 but you are running "
                    f"Python {sys.version_info[0]}")

  # Prints Tensorflow version
  print(f"This code was developed and tested on TensorFlow 1.7.0. "
        f"Your TensorFlow version: {tf.__version__}.")

  # Defines {FLAGS.train_dir}, maybe based on {FLAGS.experiment_dir}
  if not FLAGS.experiment_name:
    raise Exception("You need to specify an --experiment_name or --train_dir.")
  FLAGS.train_dir = (FLAGS.train_dir
                     or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name))

  # Sets GPU settings
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  #############################################################################
  # Train/dev split and model definition                                      #
  #############################################################################
  # Initializes model from atlas_model.py
  module = __import__("atlas_model")
  model_class = getattr(module, FLAGS.model_name)
  atlas_model = model_class(FLAGS)

  if FLAGS.mode == "train":
    if not os.path.exists(FLAGS.train_dir):
      os.makedirs(FLAGS.train_dir)

    # Sets logging configuration
    logging.basicConfig(filename=os.path.join(FLAGS.train_dir, "log.txt"),
                        level=logging.INFO)

    # Saves a record of flags as a .json file in {train_dir}
    # TODO: read the existing flags.json file
    with open(os.path.join(FLAGS.train_dir, "flags.json"), "w") as fout:
      flags = {k: v.serialize() for k, v in FLAGS.__flags.items()}
      json.dump(flags, fout)

    with tf.Session(config=config) as sess:
      # Loads the most recent model
      initialize_model(sess, atlas_model, FLAGS.train_dir, expect_exists=False)
      # Trains the model
      atlas_model.train(sess, *setup_train_dev_split(FLAGS))
  elif FLAGS.mode == "eval":
    with tf.Session(config=config) as sess:
      # Sets logging configuration
      logging.basicConfig(level=logging.INFO)

      # Loads the most recent model
      initialize_model(sess, atlas_model, FLAGS.train_dir, expect_exists=True)

      # Shows examples from the dev set
      _, _, dev_input_paths, dev_target_mask_paths =\
        setup_train_dev_split(FLAGS)
      dev_dice = atlas_model.calculate_dice_coefficient(sess,
                                                        dev_input_paths,
                                                        dev_target_mask_paths,
                                                        "dev",
                                                        num_samples=1000,
                                                        plot=True)
      logging.info(f"dev dice_coefficient: {dev_dice}")

      dev_recall_pix,dev_precision_pix,dev_recall_img,dev_precision_img = atlas_model.calculate_recall_precision(sess,
                                                        dev_input_paths,
                                                        dev_target_mask_paths,
                                                        "dev",
                                                        num_samples=1000,
                                                        plot=True)
      logging.info(f"dev recall_pix: {dev_recall_pix}")
      logging.info(f"dev precision_pix: {dev_precision_pix}")
      logging.info(f"dev recall_img: {dev_recall_img}")
      logging.info(f"dev precision_img: {dev_precision_img}")

  elif FLAGS.mode == "save_output_masks":    #run with this line: python main.py --experiment_name=0002 --mode=save_output_masks --num_epochs=3 --eval_every=100 --print_every=1 --save_every=100 --summary_every=20 --model_name=ATLASModel

   
    with tf.Session(config=config) as sess:
      # Sets logging configuration
      logging.basicConfig(level=logging.INFO)

      # Loads the most recent model
      initialize_model(sess, atlas_model, FLAGS.train_dir, expect_exists=True)

      # Creates a new dataset of saved output masks

      # For each image in the dataset
      # Perform a forward pass and store the resulting mask
      # Use a boolean mask to form the final output
      # Save the final output

      prefix = os.path.join(FLAGS.data_dir, "ATLAS_R1.1")
      new_prefix = os.path.join(FLAGS.output_data_dir, "ATLAS_R1.1")
      if FLAGS.input_regex == None:
        input_paths_regex = "Site*/**/*_t1w_deface_stx/*.jpg"
      else:
        input_paths_regex = FLAGS.input_regex
    
      slice_paths = glob.glob(os.path.join(prefix, input_paths_regex),
                            recursive=True)
      #iter = 0
      struct1 = ndimage.generate_binary_structure(2, 1) # 2x2 with connectivity 1 for dilating
      for curr_file_path in slice_paths:
        curr_img = io.imread(curr_file_path)
        # opens input, resizes it, converts to a numpy array
        curr_input = Image.open(curr_file_path).convert("L")
        curr_shape=(FLAGS.slice_height,
                    FLAGS.slice_width)
        curr_input = curr_input.crop((0, 0) + curr_shape[::-1])
        curr_input = np.asarray(curr_input) / 255.0
        curr_input = np.expand_dims(curr_input,0)
        predicted_mask = atlas_model.get_predicted_masks_for_training_example(sess,curr_input)
        output_masked_image = curr_input * predicted_mask
        # dilate masked image
        output_masked_image = np.squeeze(output_masked_image)
        ndimage.binary_dilation(output_masked_image, structure=struct1, output=output_masked_image).astype(output_masked_image.dtype)               
        output_masked_image = np.dstack((output_masked_image,output_masked_image,output_masked_image))
        #create new filepath to output masked images
        old_folderpath = os.path.split(curr_file_path)[0]
        filename = os.path.split(curr_file_path)[1]
        new_slice_path = old_folderpath.replace(FLAGS.data_dir,FLAGS.output_data_dir)
        #if folder doesn't exist, make it in specified folder
        if not os.path.exists(new_slice_path):
            os.makedirs(new_slice_path)
        #save the image
        io.imsave(new_slice_path + '/' + filename, output_masked_image, quality=100)
        print("Finished saving file: " + new_slice_path + '/' + filename)
        #iter += 1
        #outpath = "../data_output_masks/"
        #io.imsave(outpath + str(iter) + '.jpg',output_masked_image,quality=100)


if __name__ == "__main__":
  tf.app.run()
