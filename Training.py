from sacred import Experiment
from sacred.observers import FileStorageObserver #CMedit

from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

# import Datasets
# from Input import Input as Input
# from Input import batchgenerators as batchgen
import Utils
import Models.UnetAudioSeparator_no_att
# import cPickle as pickle
# import Validation
import data_reader_Audio_RIRs
from data_reader_Audio_RIRs import DataReader

ex = Experiment('Waveunet Training', ingredients=[config_ingredient])
ex.observers.append(FileStorageObserver.create('my_runs'))

@config_ingredient.capture
def train(model_config, experiment_id, load_model=None,    
          epoch = 0, best_loss = 10000, best_loss_test = 10000):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of input
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator_no_att.UnetAudioSeparator_no_att(model_config["num_layers"],
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
#     pad_durations = np.array([float((sep_input_shape[1] - sep_output_shape[1])/2), 0, 0])  # Input context that the input audio has to be padded ON EACH SIDE
#     sup_batch_gen = batchgen.BatchGen_Paired(
#         model_config,
#         sup_dataset,
#         sep_input_shape,
#         sep_output_shape,
#         pad_durations[0]
#     )
    coord = tf.train.Coordinator()
    with tf.name_scope('create_inputs'):
        reader = DataReader(
                model_config["data_dir"],
                coord,
                sample_size=sep_input_shape[1],
                hint_size=0,
                target_size=sep_output_shape[1],
                sample_rate=model_config["sample_rate"],
                queue_size = 128,
                random_crop = True,
                data_range = data_reader_Audio_RIRs.CLEAN_DATA_RANGE,
                test_data_range = data_reader_Audio_RIRs.CLEAN_TEST_DATA_RANGE,
                disc_thread_enabled = False,
                spec_generator = None,
                use_label_class = False,
                hint_window = 128,
                inject_noise = True,
                augment_reverb=True,
                augment_speech=True,
                norm_volume=False,
                stft_similarity=None)
        
        train_batches = reader.dequeue(model_config["batch_size"])

        """For test set"""
        test_batches = reader.dequeue_test(model_config["batch_size"])
        test_ext_batches = reader.dequeue_test_ext(model_config["batch_size"])

#     print("Starting worker")
#     sup_batch_gen.start_workers()
#     print("Started worker!")

    # Placeholders and input normalisation
#     mix_context, sources = Input.get_multitrack_placeholders(sep_output_shape, model_config["num_sources"], sep_input_shape, "sup")
    #tf.summary.audio("mix", mix_context, 16000, collections=["sup"]) #Enable listening to source estimates via Tensorboard

    mix_context, sources = train_batches
#     mix = Utils.crop(mix_context, sep_output_shape)
    
    print("Training...")

    # BUILD MODELS
    # Separator
    separator_sources = separator_func(mix_context, True, not model_config["raw_audio_loss"], reuse=False) # Sources are output in order [noise, speech] for speech enhancement

    # Supervised objective: MSE in log-normalized magnitude space
#     separator_loss = 0
#     for (real_source, sep_source) in zip(sources, separator_sources):
    separator_loss = tf.reduce_mean(tf.abs(sources - separator_sources[0]))
#     separator_loss = separator_loss / float(len(sources)) # Normalise by number of sources

    # TRAINING CONTROL VARIABLES
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)
    
    # Create Tests
    test_mix_context, test_sources = test_batches
    test_prediction = separator_func(test_mix_context, False, not model_config["raw_audio_loss"], reuse=True)
    test_ext_mix_context, test_ext_sources = test_ext_batches
    test_ext_prediction = separator_func(test_ext_mix_context, False, not model_config["raw_audio_loss"], reuse=True)
    
    test_loss = tf.reduce_mean(tf.abs(test_sources - test_prediction[0]))
    test_ext_loss = tf.reduce_mean(tf.abs(test_ext_sources - test_ext_prediction[0]))
    
    # Set up optimizers
    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables " + str(len(tf.global_variables())))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope("separator_solver"):
            separator_solver = tf.train.AdamOptimizer(learning_rate=model_config["init_sup_sep_lr"]).minimize(separator_loss, var_list=separator_vars)

    # SUMMARIES
    tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    sup_summaries = tf.summary.merge_all(key='sup')
    test_loss_summary = tf.summary.scalar("sep_test_loss", test_loss)
    test_ext_loss_summary = tf.summary.scalar("sep_test_ext_loss", test_ext_loss)

    # Start session and queue input threads
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep + str(experiment_id),graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        print("Num of variables" + str(len(tf.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess, n_threads=model_config["num_workers"])
    
    # Start training loop
    worse_epochs = 0
    best_model_path = None
    model_path = None
    while worse_epochs < model_config["worse_epochs"]: # Early stopping on validation set after a few epochs
        try:
            print("EPOCH: " + str(epoch))    
            _global_step = sess.run(global_step)
            _init_step = _global_step
            moving_avg_loss_value = 0.0
            run = True
            for i in tqdm(range(model_config["epoch_it"])):
                try:
                    _, _sup_summaries, train_loss_value = sess.run([separator_solver, sup_summaries, separator_loss])
                    writer.add_summary(_sup_summaries, global_step=_global_step)

                    # Increment step counter, check if maximum iterations per epoch is achieved and stop in that case
                    _global_step = sess.run(increment_global_step)
                    if _global_step - _init_step > 1:
                        moving_avg_loss_value = 0.8 * moving_avg_loss_value + 0.2 * train_loss_value
                    else:
                        moving_avg_loss_value = train_loss_value

                    if _global_step - _init_step > model_config["epoch_it"]:
                        run = False
                        print("Finished training phase, stopping batch generators")
                        break
                except Exception as e:
                    print(e)
                    run = False
                    break
            print("Finished epoch!")
            # Epoch finished - Save model
            model_path = saver.save(sess, model_config["model_base_dir"] + os.path.sep + str(experiment_id) + os.path.sep + str(experiment_id), global_step=int(_global_step))
    
            test_loss_list = []
            test_ext_loss_list = []
            for i in tqdm(range(40)):
                _test_loss_summary, _test_ext_loss_summary, _test_loss, _test_ext_loss = sess.run([test_loss_summary, test_ext_loss_summary, test_loss, test_ext_loss])
                writer.add_summary(_test_loss_summary, global_step=_global_step + i)
                writer.add_summary(_test_ext_loss_summary, global_step=_global_step + i)
                test_loss_list.append(_test_loss)
                test_ext_loss_list.append(_test_ext_loss)
            curr_loss_val = np.mean(test_loss_list)
            curr_loss_test = np.mean(test_ext_loss_list)
            print("End Test (", epoch, ") :", moving_avg_loss_value, curr_loss_val, curr_loss_test)  
            
            epoch += 1
            if curr_loss_val < best_loss:
                worse_epochs = 0
                print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss_val))
                best_model_path = model_path
                best_loss = curr_loss_val
            else:
                worse_epochs += 1
                print("Performance on validation set worsened to " + str(curr_loss_val))
            
            if curr_loss_test < best_loss_test:
                print("Performance on test set improved from " + str(best_loss_test) + " to " + str(curr_loss_test))
                best_loss_test = curr_loss_test
            else:
                print("Performance on test set worsened to " + str(curr_loss_test))
            
        except Exception as e:
            print(e)
            break
    
    writer.flush()
    writer.close()
    coord.request_stop()
    sess.close()
    tf.reset_default_graph()

    print("TRAINING FINISHED - TESTING NOW AVAILABLE WITH BEST MODEL " + best_model_path)
    return best_model_path, epoch, best_loss, best_loss_test

@config_ingredient.capture
def optimise(model_config, experiment_id):
    epoch = 0
    best_loss = 10000
    best_loss_test = 10000
    best_model_path = "/home/code-base/runtime/experiments/Wave-U-Net-For-Speech-Enhancement/checkpoints/772306/772306-574000"
    for i in range(1, 2):
#         worse_epochs = 0
#         if i==1:
#             print("Finished first round of training, now entering fine-tuning stage")
#             model_config["batch_size"] *= 2
#             model_config["cache_size"] *= 2
#             model_config["min_replacement_rate"] *= 2
#             model_config["init_sup_sep_lr"] = 1e-5
        print(model_config)
        best_model_path, epoch, best_loss, best_loss_test = train(load_model=best_model_path, epoch = epoch, 
                                           best_loss = best_loss,
                                           best_loss_test = best_loss_test)
    print("TRAINING FINISHED - TESTING NOW AVAILABLE WITH BEST MODEL " + best_model_path)
    return best_model_path, best_loss

@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

#     # Set up data input
#     pickle_file = "dataset.pkl"
#     if os.path.exists(pickle_file): # Check whether our dataset file is already there, then load it
#         with open(pickle_file, 'r') as file:
#             dataset = pickle.load(file)
#         print("Loaded dataset from pickle!")
#     else: # Otherwise create the dataset pickle

#         print("Preparing dataset! This could take a while...")

#         # Specify path to dataset, as a tracklist composed by an XML file parsed using etree in Datasets.getAudioData
#         # Each track element, describing 3 sources [speech.wav, noise.wav, mix.wav] and their relevant metadata, is parsed using etree in Datasets.py
#         dataset_train = Datasets.getAudioData("")

        # Pick 10 random songs for validation from train set (this is always the same selection each time since the random seed is fixed)
#         val_idx = np.random.choice(len(dataset_train), size=10, replace=False)
#         train_idx = [i for i in range(len(dataset_train)) if i not in val_idx]
#         print("Validation with training items no. " + str(train_idx))

#         # Draw randomly from datasets
#         dataset = dict()
#         dataset["train"] = dataset_train
#         dataset["valid"] = [dataset_train[i] for i in val_idx]

#         # Now create dataset, for source separation task for speech enhancement
#         assert(model_config["task"] == "speech")

#         for subset in ["train", "valid"]:
#             for i in range(len(dataset[subset])):
#                 dataset[subset][i] = (dataset[subset][i][0], dataset[subset][i][1], dataset[subset][i][2])

#         # Save dataset
#         with open("dataset.pkl", 'wb') as file:
#             pickle.dump(dataset, file)
#         print("Wrote source separation for speech enhancement dataset!")

#         print("LOADED DATASET")

    # The dataset structure is a dictionary with "train", "valid", "test" keys, whose entries are lists, where each element represents a noisy speech file.
    # Each noisy speech file is represented as a tuple of (mix, noise, speech) in the source separation task for speech enhancement.

    # Optimize in a supervised fashion until validation loss worsens
    sup_model_path, sup_loss = optimise()
    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))
