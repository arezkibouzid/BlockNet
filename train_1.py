from absl import app
from absl.flags import argparse_flags 
import argparse
import sys 
import glob
import tensorflow as tf
from model import BLOCKNet 
from dataloader import DataLoader
tf.config.run_functions_eagerly(True)


def EPE(flows_gt, flows):
    # Given ground truth and estimated flow must be unscaled
    return tf.reduce_mean(tf.norm(flows_gt-flows, ord = 2, axis = 3))

def L2loss(x, y): # shape(# batch, h, w, 2)
    return tf.reduce_mean(tf.reduce_sum(tf.norm(x-y, ord = 2, axis = 3), axis = (1,2)))



def multiscale_loss(flows_gt, flows_pyramid,
                    weights, name = 'multiscale_loss'):
    # Argument flows_gt must be unscaled, scaled inside of this loss function
    # Scale the ground truth flow, stated Sec.4 in the original paper
    flows_gt_scaled = flows_gt/20.
    # Calculate mutiscale loss
    loss = 0.
    for l, (weight, fs) in enumerate(zip(weights, flows_pyramid)):
        # Downsampling the scaled ground truth flow
        _, h, w, _ = tf.unstack(tf.shape(fs))
        fs_gt_down = tf.image.resize_nearest_neighbor(flows_gt_scaled, (h, w))
        # Calculate l2 loss
        loss += weight*L2loss(fs_gt_down, fs)

    return loss




def scheduler(epoch, lr):
    # len(train_set) // batch_size = 1389
    if epoch*1389 in args.iterations:
        return lr*(1/2)
    else:
        return lr


def train(args):
    """Instantiates and trains the model."""
    if args.precision_policy:
        tf.keras.mixed_precision.set_global_policy(args.precision_policy)
    if args.check_numerics:
        tf.debugging.enable_check_numerics()


    data_loader = DataLoader(args.dd, args.tlist, args.vlist)
    train_dataset,validation_dataset = data_loader.create_tf_dataset(flags=args)
    print("Train samples : ",len(train_dataset))
    print("Validation samples : ",len(validation_dataset))

    model = BLOCKNet(gamma= args.gamma , weights=args.weights,num_levels = args.num_levels, 
                              search_range = args.search_range, warp_type = args.warp_type,
                                    output_level = args.output_level, filters= args.filters)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.Learning_rate))

    '''if args.train_glob:
        train_dataset = get_custom_dataset("train", args)
        validation_dataset = get_custom_dataset("validation", args)
    else:
        train_dataset = get_dataset("clic", "train", args)
        validation_dataset = get_dataset("clic", "validation", args)'''


    callback = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=0)
    
    model.fit(
        train_dataset.prefetch(8),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_dataset.cache(),
        validation_freq=1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(
              log_dir=args.train_path,
              histogram_freq=1, update_freq="epoch"),
            tf.keras.callbacks.BackupAndRestore(args.train_path),
            callback,
        ],
        verbose=int(args.verbose),
    )
    model.save(args.model_path)














def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
    parser.add_argument(
      "--model_path", default="res/BlockNet_model_ckpt",
      help="Path where to save/load the trained model.")
    subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'eval' for evalution pre trained model ")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
    train_cmd.add_argument(
      "--Learning_rate","-lr", type=float, default=0.01, dest="Learning_rate",
      help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
      "--train_list","-tl", type=str, default="FlyingChairs_release/FlyingChairs_train_list.txt", dest="tlist",
      help="file name of train list")

    train_cmd.add_argument(
      "--val_list","-vl", type=str, default="FlyingChairs_release/FlyingChairs_val_list.txt", dest="vlist",
      help="file name of val list")
      
    train_cmd.add_argument(
      "--data_set","-d", type=str, default='SintelClean', dest="d",
      help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
      "--data_dir","-dd", type=str, default=None, dest="dd",
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of RGB images in PNG format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")

    train_cmd.add_argument(
       "--num_levels", type=int, default=3,
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--search_range", type=int, default=32,
      help="Number of filters per layer.")
      

    train_cmd.add_argument(
       "--warp_type", type=str, default='bilinear',
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--output_level", type=int, default=1,
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--random_scale", type=bool, default=False,
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--crop_size", nargs="+", type=int, default=[384, 448],
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--weights", nargs="+", type=int,dest="weights", default=[0.32,0.08,0.02],
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--iterations", nargs="+", type=int,dest="iterations", default=[200000,250000,300000,350000],
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--num_filters", nargs="+", type=int,dest="filters", default=[16, 32, 64],
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--gamma", type=int,dest="gamma", default=0.0004,
      help="Number of filters per layer.")

    train_cmd.add_argument(
       "--random_flip", type=bool, default=False,
      help="Number of filters per layer.")

    train_cmd.add_argument(
      "--batch_size", type=int, default=4,
      help="Batch size for training and validation.")
    train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
    train_cmd.add_argument(
      "--epochs", type=int, default=600,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")

    #to check
    train_cmd.add_argument(
      "--train_path", default="res/train_zyc2022",
      help="Path where to log training metrics for TensorBoard and back up "
           "intermediate model checkpoints.")
    train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
    train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation. If -1, use one "
           "patch from each image in the training set.")
    train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
    train_cmd.add_argument(
      "--precision_policy", type=str, default=None,
      help="Policy for `tf.keras.mixed_precision` training.")
    train_cmd.add_argument(
      "--check_numerics", action="store_true",
      help="Enable TF support for catching NaN and Inf in tensors.")

    # 'evaluation' subcommand.
    compress_cmd = subparsers.add_parser(
      "evaluation",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="evaluate model .")

    compress_cmd.add_argument(
      "--eval_glob", type=str, default=None,
      help="Glob pattern identifying custom evalutaion data.")

    # 'inference' subcommand.
    decompress_cmd = subparsers.add_parser(
      "inference",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="inference ....")

  # Arguments for both 'compress' and 'decompress'.
  #for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    #cmd.add_argument(
    #    "input_file",
    #    help="Input filename.")
    #cmd.add_argument(
    #    "output_file", nargs="?",
    #    help=f"Output filename (optional). If not provided, appends '{ext}' to "
    #         f"the input filename.")

    # Parse arguments.
    global args
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args




def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "eval":
    if not args.output_file:
        print("")   
  elif args.command == "infernce":
        print("")   




if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)