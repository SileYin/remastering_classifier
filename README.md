# Classifier for remastering/CD vs vinyl records
## Requirements
Librosa

Numpy

tqdm

sklearn for baseline system (process_dataset.py and train.py)

pytorch for CNN (cnn_\*.py and classify.py)

## Usage
Training:

  `python cnn_train.py`
  
Using trained model for classification:

  `python classify.py <path_to_model> <path_to_audio_file>`
