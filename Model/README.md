
This folder contains the core components of the CNN-based Cats vs Dogs classification project.

## Contents

- `cnn.py`  
  → Python script containing the full CNN architecture, training pipeline, and evaluation logic.

- `best_model.h5`  
  → Saved Keras model after training.  
  → Can be loaded for inference or further fine-tuning.

- `accuracy_loss_graph.png`  
  → Accuracy and loss graph across training epochs.  
  → Useful for visualizing model performance and detecting overfitting.

## Model Details

- Framework: TensorFlow 2.x (Keras API)
- Base model: VGG16 (pre-trained, frozen layers)
- Input shape: (150, 150, 3)
- Classes: Binary (Cats vs Dogs)
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Epochs: 10
- Callbacks: ModelCheckpoint, EarlyStopping

## Usage

To load the trained model:
```python
from tensorflow.keras.models import load_model
model = load_model('model/cnn_model.h5')
