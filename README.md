# Generating Ambient Music From Raw Audio Using SampleRNN

### Overview

Music generation in machine learning often operates with reperesentations of audio, such as spectrograms or MIDI files. This approach comes with an obvious drawback: hard-to-represent qualities of audio, such as timbre or dynamics, are compromised or lost entirely. To retain these qualities, one must operate with raw audio. This is the approach of [SampleRNN](https://arxiv.org/abs/1612.07837), which was invented by Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Manuel, Rodriguez Sotelo, Aaron Courville and Yoshua Bengio, and takes raw audio as input and generates new audio at the level of individual samples.

This project borrows heavily from the [basic approach](https://arxiv.org/pdf/1811.06633.pdf) of the [Dadabots](http://dadabots.com/) team (CJ Carr, Zack Zukowski) in the way they utilized SampleRNN. By training on full albums, they showed that SampleRNN gives results of extremely high fidelity, and in the case of black metal even produces song-like output that mimics the style of the artist. However, this same approach applied to, for example, Beatles music demonstrates that SampleRNN can only go so far in capturing song structure, with the results sounding more like Beatles 'soup'.

This was my rationale for generating ambient music, where the structure is much less important than the sonic textures, which can't be represented symbolically, and are best preserved with the raw audio approach.

### Data

Because SampleRNN is designed to accept raw audio, the data here require a minimal amount of preprocessing. After downloading full albums from YouTube and converting them to WAV format using [YouTube_to_WAV.py](https://github.com/timothymeehan/ambient_music_generator/blob/master/code/YouTube_to_WAV.py), I cut them into 8 second long, overlapping chunks using [chunk_audio_overlap.py](https://github.com/timothymeehan/ambient_music_generator/blob/master/code/chunk_audio_overlap.py). This produced ~3000-4000 chunks depending on the album length.

### Training

The chunked audio serves directly as the training data for SampleRNN. Because SampleRNN generates new audio one sample at a time, training is extremely computationally expensive. For decent quality output, a minimal sampling rate of 16 kHz is advised. This means for each second of audio, SampleRNN is making 16000 'predictions', or newly generated samples. To reduce training time, use a machine with a GPU. I used a p3.xlarge instance on AWS and trained for several days in order to generate a sufficient amount of output to sort through.

The model implementation is in Pytorch, and was provided by deepsound (forked from [thier repository](https://github.com/deepsound-project/samplernn-pytorch)). It consists of the scripts [train.py](https://github.com/timothymeehan/ambient_music_generator/blob/master/code/train.py), [model.py](https://github.com/timothymeehan/ambient_music_generator/blob/master/code/model.py), [nn.py](https://github.com/timothymeehan/ambient_music_generator/blob/master/code/nn.py), [optim.py](https://github.com/timothymeehan/ambient_music_generator/blob/master/code/optim.py), and [utils.py](https://github.com/timothymeehan/ambient_music_generator/blob/master/code/utils.py). The main script, train.py, was modified to account for changes in Pytorch since deepsound's release. Many thanks to Fabian Stahl for his assistance with this.

To run the model with all default hyperparameters (which should provide good quality results), run:
```
python train.py --exp my_experiment_name --frame_sizes 16 4 --n_rnn 2 --dataset my_dataset_directory
```
where you choose the name of your 'experiment' and provide the dataset directory with the audio chunks. Be sure to first create a 'results' directory for the output.

### Results

I was fairly happy with my results. One interesting thing I found was that input with more variation both within and across input chunks led to more interesting, 'creative' sounding output, whereas music with slower dynamics produced more drone-y and monotonous output. To hear some examples, please check out the [PowerPoint presentation](https://github.com/timothymeehan/ambient_music_generator/blob/master/generating_ambient_music.pptx).

### Conclusions

In my opinion, SampleRNN holds a lot of potential for machine-generated music. Its most impressive attribute is the quality of audio it outputs, its ability to retain the hard-to-represent qualities of the original audio, and output something novel all at once. Its main drawback is the extreme cost in either time or computing power. If one is able to handle those costs, there are a lot of ways it could be used, such as creating mashups by mixing input from different albums and/or artists. This is something I'd like to return to in the future.
