# Anime GAN: Co-Creative Sketch Tracing
This repository contains the implementation of a Generative Adversarial Network (GAN) designed to generate high-quality anime-style images from input sketches. This project leverages TensorFlow and Colab to provide an intuitive and efficient pipeline for training and generating images.

**Overview**

The Anime GAN project is designed to train a GAN model for transforming input sketches into stylized anime images. The project workflow involves:

1) Preprocessing input data to normalize and resize images.
2) Training a GAN architecture that consists of a Generator and a Discriminator.
3) Evaluating the model on test data to generate stylized outputs.
The core idea is to enhance sketch-based creativity using deep learning techniques, enabling automatic and co-creative sketch-to-anime conversion.

**Key Features**
1) Dataset Handling: Supports loading and processing datasets for both training and testing (e.g., Pokémon Pix2Pix dataset).
2) GAN Architecture:
   i. Generator: Constructs stylized outputs using a U-Net-like architecture with down-sampling 
      and up-sampling layers.
   ii. Discriminator: Classifies real vs. generated outputs using a PatchGAN model.
3) Image Preprocessing:
  - Normalizes images between -1 and 1.
  - Resizes images to 256x256 pixels.
  - Includes data augmentation techniques like flipping.
4) Loss Functions: Binary cross-entropy and L1 loss to stabilize training.
5) Performance: Generates high-quality outputs with clear outlines and anime features after 100 epochs of training.

**Setup Instructions**

Prerequisites
  - Python 3.x
  - TensorFlow 2.x
  - Google Colab (optional for running in the cloud)


**Installation**
1. Clone the repository:
   git clone https://github.com/Naveen8297/Anime_GAN.git
2. Install required dependencies:
   pip install -r requirements.txt
3. Launch the Jupyter Notebook or open the project in Google Colab:
   - Open in Colab

**Training the Model**

**Steps**

- Mount Google Drive (if using Colab):
  ```
  from google.colab import drive
  drive.mount('/content/drive')
- Load and Preprocess Dataset:
  - Images are loaded and split into input sketches and ground truth targets.
- Model Training:
  - Set hyperparameters such as BATCH_SIZE, BUFFER_SIZE, and EPOCHS.
  - Use train_step() for training the Generator and Discriminator together.
- Save the Model:
  - Save trained Generator and Discriminator for reuse:
    ```
    generator.save('path_to_save_generator')
    discriminator.save('path_to_save_discriminator')
    ```
**File Structure**

CC_FinalProject_Group7.ipynb: Main notebook containing the implementation and execution of the  project.
Dataset: (Path provided during training in Colab).

**Results**

- The trained model generates high-quality anime-style images from sketches.
- Generated outputs retain crucial details of the input sketches while applying anime-style transformations.

**Future Work**

- Integrate advanced data augmentation techniques to improve generalization.
- Experiment with different GAN architectures (e.g., StyleGAN).
- Fine-tune models for specific anime styles.


