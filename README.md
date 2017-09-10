# ee
My Computer Science/ Music Extended Essay for the IB Diploma...
I will be comparing RBM and RNN music generation with respect to Video Game music.

Do This:
git clone https://github.com/tensorflow/magenta.git
	follow install procedure
	install bazel

TO GENERATE NOTE SEQUENCES:::
	./bazel-bin/magenta/scripts/convert_dir_to_note_sequences --input_dir=~/ee/midi/ --output_file=notesequences.tfrecord --recursive

BACKGROUND:::
      1. CTRL Z
      2. bg 1
      3. disown -h %1

-------------

# to create melody
0) cd magenta/
1) ./bazel-bin/magenta/models/melody_rnn/melody_rnn_create_dataset --config=attention_rnn --input=/home/spencer_l_churchill/ee/notesequences.tfrecord --output_dir=/home/spencer_l_churchill/ee/out/ --eval_ratio=0.10
2) ./bazel-bin/magenta/models/melody_rnn/melody_rnn_train --config=attention_rnn --run_dir=/home/spencer_l_churchill/ee/rundir/ --sequence_example_file=/home/spencer_l_churchill/ee/out/training_melodies.tfrecord --hparams="batch_size=64,rnn_layer_sizes=[64,64]"
3) tensorboard --port 6969 --logdir=/home/spencer_l_churchill/ee/rundir/
4) ./bazel-bin/magenta/models/melody_rnn/melody_rnn_generate --config=attention_rnn --run_dir=/home/spencer_l_churchill/ee/rundir/ --output_dir=/home/spencer_l_churchill/ee/generated/ --num_outputs=10 --num_steps=480 --hparams="batch_size=64,rnn_layer_sizes=[64,64]" --primer_melody="[]"

-------------

# to create polyphony
0) cd magenta/
1) ./bazel-bin/magenta/models/polyphony_rnn/polyphony_rnn_create_dataset --input=/home/spencer_l_churchill/ee/res/notesequences.tfrecord --output_dir=/home/spencer_l_churchill/ee/res/out/ --eval_ratio=0.10
2) ./bazel-bin/magenta/models/polyphony_rnn/polyphony_rnn_train --run_dir=/home/spencer_l_churchill/ee/res/rundir/ --sequence_example_file=/home/spencer_l_churchill/ee/res/out/training_poly_tracks.tfrecord --hparams="batch_size=64,rnn_layer_sizes=[64,64]"
3) tensorboard --port 6969 --logdir=/home/spencer_l_churchill/ee/res/rundir/
4) ./bazel-bin/magenta/models/polyphony_rnn/polyphony_rnn_generate --run_dir=/home/spencer_l_churchill/ee/res/rundir/ --output_dir=/home/spencer_l_churchill/ee/res/generated/ --num_outputs=10 --num_steps=128 --hparams="batch_size=64,rnn_layer_sizes=[64,64]" --primer_melody="[]"
