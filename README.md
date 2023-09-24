# GANs
My work on Generative Adversarial Networks is in this respositories

# This repositories contains couple of different projects
- CycleGAN for Image-to-Image Translation

# First Project: CycleGAN for Image-to-Image Translation

## Introduction

This repository contains the code for a CycleGAN implementation that performs image-to-image translation. CycleGAN is a type of generative adversarial network (GAN) designed for unsupervised image translation tasks.

#### - The code for this work is given in filename:  CycleGAN.ipynb
#### - The same work of CycleGAN for Image-to-Image Translation with ResNet in it code is given in this filename:  CycleGAN_with_resnet.ipynb

## Dependencies

Make sure you have the following dependencies installed:
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- TensorFlow Addons

You can install the required packages using the provided `requirements.txt` file:
```bash 
pip install -r requirements.txt
``````

## Usage
- Data Preparation: Organize your dataset by placing images of apples in the apples directory and images of guavas in the guava directory.

- Training: Run the train function in the Jupyter Notebook or your preferred Python environment. Training will iterate through epochs, performing image translation and updating the models.

- Testing: To test the trained model, you can load a test image and apply one of the generators to generate a translated image

## Model Architecture
- Two generator models (generator_ab and generator_ba) for translating between apple and guava images.
- Two discriminator models (discriminator_a and discriminator_b) for distinguishing real from fake images.
- A combined model (combined) that includes both generators and discriminators.

## Results
During training, the code periodically displays generated images to visualize the translation progress. After training, you can also test the model's translation capabilities.

## Acknowledgments
This implementation is inspired by the original CycleGAN paper by Jun-Yan Zhu et al.

<br>
<br>
<br>
<br>


# Second Project: Character-Level Text Generation with LSTM

## Introduction

This repository contains code for a character-level text generation model using an LSTM (Long Short-Term Memory) network. The model generates text based on a given input sequence of characters.

## Dependencies

Make sure you have the following dependencies installed:
- TensorFlow
- NumPy
- Matplotlib

You can install the required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
``````

## Usage
Text Preprocessing
The code reads and preprocesses text from a file ('aesop.txt'). It converts the text to lowercase, splits it into sentences, handles punctuation, and prepares it for training.

## Model Architecture
The model architecture includes an LSTM-based sequence generator. It takes sequences of characters as input and predicts the next character.

## Model Training
The LSTM model is trained on the processed text data. It learns to generate text that resembles the input text.

## Text Generation
The trained model can generate text by providing an initial seed text. It generates text one character at a time, gradually extending the sequence.

## Results
The generated text can be used for various natural language processing tasks, including text completion and creative text generation.


<br>
<br>
<br>
<br>

# Third Project: Music Generation with GAN

## Introduction

This repository contains code for generating music using a Generative Adversarial Network (GAN). The GAN consists of a generator and a discriminator, trained adversarially to generate musical scores.

## Dependencies

Make sure you have the following dependencies installed:
- TensorFlow
- NumPy
- Matplotlib
- music21

You can install the required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
``````
## Usage
### Dataset
The code downloads and preprocesses the JSB Chorales dataset, which contains musical scores.
### Model Architecture
The GAN consists of the following components:

- Generator (generator): Generates musical scores based on input vectors representing style, chords, melody, and groove.

- Discriminator (discriminator): Distinguishes between real and generated musical scores.

- Critic Model (critic_model): Combines the discriminator with gradient penalty for improved stability during training.

- Generator Model (generator_model): Used to train the generator to minimize the critic's output.

### Training
The GAN is trained using an alternating training loop, where the critic is trained to distinguish real from generated data, and the generator is trained to fool the critic.

### TensorBoard Visualization
Use TensorBoard to visualize the training progress by running the following command:

```%tensorboard --logdir logs```


### Generating Music
The trained generator can be used to generate music by providing input vectors representing different aspects of the music.

### Converting to MIDI
The generated musical scores can be converted into MIDI files using the music21 library.

<br>
<br>
<br>


# Music Generation with Recurrent Neural Networks

## Introduction

This repository contains code for generating music sequences using a Recurrent Neural Network (RNN). The RNN model is designed to predict both notes and their corresponding durations to generate musical sequences.

#### - The code for this work is given in filename:  Music_RNN.ipynb

## Dependencies

Make sure you have the following dependencies installed:

- TensorFlow
- NumPy
- Matplotlib
- music21

You can install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
