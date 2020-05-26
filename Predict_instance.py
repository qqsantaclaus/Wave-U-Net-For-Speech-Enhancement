from sacred import Experiment
from sacred.observers import FileStorageObserver #CMedit

from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import math
import librosa
import soundfile as sf

# import Datasets
from Input import Input as Input
# from Input import batchgenerators as batchgen
import Utils
import Models.UnetAudioSeparator
# import cPickle as pickle
# import Validation
import data_reader_Audio_RIRs
from data_reader_Audio_RIRs import DataReader

@config_ingredient.capture
def predict(model_config, input_path, output_path=None, output_name=None, load_model=None):
    # Determine input and output shapes
    
    print("Producing source estimates for input mixture file " + input_path)
    # Prepare input audio for prediction function
#     audio, sr = Utils.load(input_path, sr=None, mono=False)
    audio, sr = librosa.load(input_path, sr=model_config["sample_rate"], mono=False)
    expon = math.ceil(np.log(len(audio))/ np.log(2))
    padded_total_len = int(2**expon)
    padded_len = padded_total_len - len(audio)
    padded_audio = np.pad(audio, (0, padded_len), 'constant')
    padded_audio = np.reshape(padded_audio, (1, len(padded_audio), 1))
    
    disc_input_shape = [1, padded_audio.shape[1], 0]  # Shape of input
    
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config["num_layers"],
                                                                       model_config["num_initial_filters"],
                                                                   output_type=model_config["output_type"],
                                                                   context=model_config["context"],
                                                                   mono=model_config["mono_downmix"],
                                                                   upsampling=model_config["upsampling"],
                                                                   num_sources=model_config["num_sources"],
                                                                   filter_size=model_config["filter_size"],
                                                                   merge_filter_size=model_config["merge_filter_size"])

    else:
        raise NotImplementedError
    
    
    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output
    print(sep_input_shape, sep_output_shape)
    # Creating the batch generators
    assert((sep_input_shape[1] - sep_output_shape[1]) % 2 == 0)

    # Placeholders and input normalisation
    mix_context, sources = Input.get_multitrack_placeholders(sep_output_shape, model_config["num_sources"], sep_input_shape, "sup")

    # BUILD MODELS
    # Separator
    separator_sources = separator_func(mix_context, False, not model_config["raw_audio_loss"], reuse=False) # Sources are output in order [noise, speech] for speech enhancement
#     separator_loss = tf.reduce_mean(tf.abs(sources - separator_sources[0]))
    
    # Set up optimizers
    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables " + str(len(tf.global_variables())))

    # Start session and queue input threads
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        print("Num of variables" + str(len(tf.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    separator_sources_value = sess.run([separator_sources], feed_dict={mix_context:padded_audio})

    sess.close()
    tf.reset_default_graph()
    
    output_audio = separator_sources_value[0][0][0, :len(audio), 0]

    # Save source estimates as audio files into output dictionary
    input_folder, input_filename = os.path.split(input_path)
    if output_name is None:
        output_name = input_filename
    if output_path is None:
        # By default, set it to the input_path folder
        output_path = input_folder
        output_filename = os.path.join(output_path, output_name)
    else:
        output_filename = os.path.join(output_path, output_name)
            
    if not os.path.exists(output_path):
        print("WARNING: Given output path " + output_path + " does not exist. Trying to create it...")
        os.makedirs(output_path)
    assert(os.path.exists(output_path))
    
    sf.write(output_filename, output_audio, model_config["sample_rate"])

    
ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])
ex.observers.append(FileStorageObserver.create('my_runs/Predictions'))

session_id = "506984"
ckpt = "496000"

@ex.config
def cfg():
    load_model = os.path.join("checkpoints", session_id, "-".join([session_id, ckpt])) # Load model from checkpoints folder. E.g. a particular model, "105373-450225" from "checkpoints/105373"
    input_path = os.path.join("/home/code-base/runtime/experiments/MetricGAN/data/Noisy/f10_script5_ipad_office1.wav") # Which audio file to separate. In this example, within path
#             if not os.path.exists(os.path.join("./infer_outputs/daps", session_id, dirpath.split("/")[-2], "WaveUNet")):
#                 os.makedirs(os.path.join("./infer_outputs/daps", session_id, dirpath.split("/")[-2], "WaveUNet"))
    output_path = os.path.join("./infer_outputs") # Where to save results. Default: Same location as input.
    output_name = "1-noisy-float.wav"

@ex.automain
def run(cfg, input_path, output_path, output_name, load_model):
    model_config = cfg["model_config"]
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    predict(model_config, input_path, output_path, output_name, load_model)
    