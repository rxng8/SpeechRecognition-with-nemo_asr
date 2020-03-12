# %%

# NeMo's "core" package
import nemo
# NeMo's ASR collection
import nemo_asr

# %%

import pandas
# %%

# Read CSV File

dataset = pandas.read_csv('./dataset/label/scripts.csv', sep='|', names=['fileName', 'text', 'duration'])

dataset.loc[0,'text']

# %%

# Write JSON with the same form as example dataset on nemo.

import json

data = []

for i, datum in enumerate(dataset.values):
    datum_block = {}
    datum_block['audio_filepath'] = datum[0]
    datum_block['duration'] = datum[2]
    datum_block['text'] = datum[1]
    data.append(datum_block)

# %%

with open('./dataset/label/scripts.json', 'w', encoding='utf8') as outfile:
    json.dump(data, outfile, ensure_ascii=False)
# %%

for datum in dataset.values:
    print(datum)


# %%

data

# %%

# Create a Neural Factory
# It creates log files and tensorboard writers for us among other functions
nf = nemo.core.NeuralModuleFactory(
    log_dir='jasper12x1SEP',
    create_tb_writer=True)
tb_writer = nf.tb_writer
logger = nf.logger

# Path to our training manifest
train_dataset = "./dataset/label/scripts.json"

# Path to our validation manifest
eval_datasets = "./dataset/label/scripts.json"

# Jasper Model definition
from ruamel.yaml import YAML

# Here we will be using separable convolutions
# with 12 blocks (k=12 repeated once r=1 from the picture above)
yaml = YAML(typ="safe")
with open("./NeMo/examples/asr/configs/jasper12x1SEP.yaml") as f:
    jasper_model_definition = yaml.load(f)
labels = jasper_model_definition['labels']

# Instantiate neural modules
data_layer = nemo_asr.AudioToTextDataLayer(
    manifest_filepath=train_dataset,
    labels=labels, batch_size=32)
data_layer_val = nemo_asr.AudioToTextDataLayer(
    manifest_filepath=eval_datasets,
    labels=labels, batch_size=32, shuffle=False)

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor()
spec_augment = nemo_asr.SpectrogramAugmentation(rect_masks=5)

jasper_encoder = nemo_asr.JasperEncoder(
    feat_in=64,
    **jasper_model_definition['JasperEncoder'])
jasper_decoder = nemo_asr.JasperDecoderForCTC(
    feat_in=1024, num_classes=len(labels))
ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

# Training DAG (Model)
audio_signal, audio_signal_len, transcript, transcript_len = data_layer()
processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal, length=audio_signal_len)
aug_signal = spec_augment(input_spec=processed_signal)
encoded, encoded_len = jasper_encoder(
    audio_signal=aug_signal, length=processed_signal_len)
log_probs = jasper_decoder(encoder_output=encoded)
predictions = greedy_decoder(log_probs=log_probs)
loss = ctc_loss(
    log_probs=log_probs, targets=transcript,
    input_length=encoded_len, target_length=transcript_len)

# Validation DAG (Model)
# We need to instantiate additional data layer neural module
# for validation data
audio_signal_v, audio_signal_len_v, transcript_v, transcript_len_v = data_layer_val()
processed_signal_v, processed_signal_len_v = data_preprocessor(
    input_signal=audio_signal_v, length=audio_signal_len_v)
# Note that we are not using data-augmentation in validation DAG
encoded_v, encoded_len_v = jasper_encoder(
    audio_signal=processed_signal_v, length=processed_signal_len_v)
log_probs_v = jasper_decoder(encoder_output=encoded_v)
predictions_v = greedy_decoder(log_probs=log_probs_v)
loss_v = ctc_loss(
    log_probs=log_probs_v, targets=transcript_v,
    input_length=encoded_len_v, target_length=transcript_len_v)

# These helper functions are needed to print and compute various metrics
# such as word error rate and log them into tensorboard
# they are domain-specific and are provided by NeMo's collections
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch

from functools import partial
# Callback to track loss and print predictions during training
train_callback = nemo.core.SimpleLossLoggerCallback(
    tb_writer=tb_writer,
    # Define the tensors that you want SimpleLossLoggerCallback to
    # operate on
    # Here we want to print our loss, and our word error rate which
    # is a function of our predictions, transcript, and transcript_len
    tensors=[loss, predictions, transcript, transcript_len],
    # To print logs to screen, define a print_func
    print_func=partial(
        monitor_asr_train_progress,
        labels=labels,
        logger=logger
    ))

saver_callback = nemo.core.CheckpointCallback(
    folder="./",
    # Set how often we want to save checkpoints
    step_freq=100)

# PRO TIP: while you can only have 1 train DAG, you can have as many
# val DAGs and callbacks as you want. This is useful if you want to monitor
# progress on more than one val dataset at once (say LibriSpeech dev clean
# and dev other)
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_v, predictions_v, transcript_v, transcript_len_v],
    # how to process evaluation batch - e.g. compute WER
    user_iter_callback=partial(
        process_evaluation_batch,
        labels=labels
        ),
    # how to aggregate statistics (e.g. WER) for the evaluation epoch
    user_epochs_done_callback=partial(
        process_evaluation_epoch, tag="DEV-CLEAN", logger=logger
        ),
    eval_step=500,
    tb_writer=tb_writer)

# Run training using your Neural Factory
# Once this "action" is called data starts flowing along train and eval DAGs
# and computations start to happen
nf.train(
    # Specify the loss to optimize for
    tensors_to_optimize=[loss],
    # Specify which callbacks you want to run
    callbacks=[train_callback, eval_callback, saver_callback],
    # Specify what optimizer to use
    optimizer="novograd",
    # Specify optimizer parameters such as num_epochs and lr
    optimization_params={
        "num_epochs": 50, "lr": 0.02, "weight_decay": 1e-4
        }
    )