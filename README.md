
# 🎨 Deep Learning Sketch Cleanup: Fully Convolutional Neural Network


An advanced **computer vision system** that transforms rough hand-drawn sketches into clean, professional line art using a **fully convolutional neural network (FCN)** with **U-Net architecture** and **skip connections**. This project demonstrates state-of-the-art image-to-image translation techniques for artistic applications.

## 🎯 Project Overview

**Sketch cleanup** is a fundamental challenge in digital art and computer graphics. This project addresses the problem of automatically refining rough, noisy sketches into clean, precise line drawings using deep learning.

### Key Innovation
- **Custom Loss Function**: Combines foreground and background pixel analysis for superior sketch refinement
- **Skip Connection Architecture**: Preserves fine details while enabling global structure understanding  
- **Multi-GPU Training**: Distributed training strategy for efficient processing
- **Adaptive Thresholding**: Smart binarization based on image statistics

## 🏗️ Architecture Deep Dive

### Fully Convolutional U-Net Design

```
Input (424×424×1) → Encoder → Bottleneck → Decoder → Output (424×424×1)
                      ↓           ↓          ↑
                   Skip Connections (3 levels)
```

### Detailed Network Structure

| Stage | Operation | Filters | Resolution | Key Features |
|-------|-----------|---------|------------|--------------|
| **Encoder Block 1** | 4× Conv2D + BN + ReLU | 32 | 424×424 | Feature extraction |
| **Encoder Block 2** | 6× Conv2D + BN + ReLU | 64 | 212×212 | Downsampling + processing |
| **Encoder Block 3** | 9× Conv2D + BN + ReLU | 128 | 106×106 | Deep feature learning |
| **Bottleneck** | 9× Conv2D + BN + ReLU | 256 | 53×53 | High-level representation |
| **Decoder Block 1** | Conv2DTranspose + 8× Conv2D | 128 | 106×106 | Upsampling + refinement |
| **Decoder Block 2** | Conv2DTranspose + 5× Conv2D | 64 | 212×212 | Feature reconstruction |
| **Decoder Block 3** | Conv2DTranspose + 3× Conv2D | 32 | 424×424 | Detail restoration |
| **Output Layer** | 2× Conv2D + Sigmoid | 1 | 424×424 | Final sketch output |

### Skip Connections Strategy
- **Level 1**: Connects encoder block 1 (32 filters) to decoder block 3
- **Level 2**: Connects encoder block 2 (64 filters) to decoder block 2  
- **Level 3**: Connects encoder block 3 (128 filters) to decoder block 1

## 🧮 Custom Loss Function

### Mathematical Foundation

The project implements a sophisticated loss function that considers both **positive** (sketch lines) and **negative** (background) spaces:

```python
Loss = 1 - (γ × PMI + (1-γ) × GMI)

Where:
- PMI = Positive Match Index (background similarity)
- GMI = General Match Index (foreground similarity)  
- γ = Balance parameter (0.5)
```

### Intersection Metrics
```python
Intersection(target, pred, α) = α × (DP) + (1-α) × (FP)

Where:
- DP = Detection Precision = |P∩G| / |G|
- FP = False Positive Rate = |P∩G| / |P|  
- α = Precision/recall balance (0.5)
```

### Why This Matters
- **Balanced Learning**: Equal attention to sketch lines and background
- **Edge Preservation**: Maintains fine detail integrity
- **Noise Reduction**: Suppresses unwanted artifacts
- **Line Continuity**: Promotes smooth, connected strokes

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install tensorflow>=2.8.0 opencv-python numpy matplotlib

# Additional utilities  
pip install pillow scikit-image

# For visualization
pip install seaborn plotly

# Or install from requirements
pip install -r requirements.txt
```

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (RTX 3070/V100)
- RAM: 16GB system memory
- Storage: 10GB free space

**Recommended:**
- GPU: 16GB+ VRAM (RTX 4080/A100)
- RAM: 32GB system memory
- Multi-GPU setup supported

### Dataset Setup

```bash
# Organize your data structure
project/
├── data/
│   ├── input/          # Rough sketches (.png/.jpg)
│   └── target/         # Clean line art (.png/.jpg)
├── models/             # Saved model checkpoints
└── results/           # Output predictions
```

### Basic Usage

```python
# Initialize and train the model
from sketch_cleanup import Skip_S

# Create model with distributed training
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Skip_S()
    fcn = model.get_model()
    
# Train on your dataset
history = fcn.fit(
    input_sketches, target_sketches,
    batch_size=6,
    epochs=100,
    callbacks=[checkpoint_callback]
)
```

## 📁 Project Structure

```
├── sketch_cleanup_fcn.py         # Main model architecture
├── data_loader.py               # Image preprocessing utilities
├── custom_loss.py              # Advanced loss function
├── train.py                   # Training pipeline
├── evaluate.py               # Model evaluation and metrics
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── models/                  # Model checkpoints
│   ├── FCCN_checkpoint_*.h5  # Training checkpoints
│   └── Model2022.h5         # Final trained model
├── data/                   # Dataset directory
│   ├── input/             # Input rough sketches
│   └── target/           # Target clean sketches
├── results/              # Generated outputs
│   ├── predictions/      # Model predictions
│   ├── comparisons/     # Before/after comparisons
│   └── metrics/        # Performance analysis
└── utils/             # Utility functions
    ├── visualization.py  # Plot generation
    ├── metrics.py       # Evaluation metrics
    └── preprocessing.py # Image processing
```

## 🎨 Data Preprocessing Pipeline

### Image Processing Steps

1. **Loading & Validation**
   ```python
   img_input = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
   img_target = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
   ```

2. **Normalization**
   ```python
   img_target = img_target / img_target.max()  # [0,1] range
   ```

3. **Adaptive Thresholding**
   ```python
   mean_threshold = np.mean(img_target)
   binary_target = np.where(img_target >= mean_threshold, 1, 0)
   ```

4. **Resizing**
   ```python
   resized = cv2.resize(img, (424, 424), interpolation=cv2.INTER_AREA)
   ```

### Why 424×424 Resolution?
- **Optimal Memory Usage**: Balances detail preservation with computational efficiency
- **GPU Friendly**: Divisible by common batch sizes and memory alignment
- **Detail Retention**: Sufficient resolution for sketch line detection
- **Processing Speed**: Fast training and inference times

## ⚙️ Training Configuration

### Distributed Training Setup

```python
# Multi-GPU configuration
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)

# Optimal batch sizes by GPU memory
GPU_CONFIGS = {
    "8GB":  {"batch_size": 4, "gradient_accumulation": 2},
    "16GB": {"batch_size": 6, "gradient_accumulation": 1}, 
    "24GB": {"batch_size": 8, "gradient_accumulation": 1}
}
```

### Training Parameters

```python
TRAINING_CONFIG = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "batch_size": 6,
    "epochs": 100,
    "checkpoint_frequency": 50
}
```

### Callbacks & Monitoring

```python  
callbacks = [
    ModelCheckpoint(
        'FCCN_checkpoint_{epoch:08d}.h5',
        monitor='loss',
        save_freq=50
    ),
    TimeHistory(),  # Track training time per epoch
    EarlyStopping(patience=10, restore_best_weights=True)
]
```

## 📊 Evaluation Metrics

### Quantitative Metrics

| Metric | Purpose | Typical Range |
|--------|---------|---------------|
| **Custom Loss** | Overall sketch quality | 0.0 - 1.0 |
| **Pixel Accuracy** | Correct pixel classification | 85% - 95% |
| **IoU (Intersection over Union)** | Sketch overlap accuracy | 0.7 - 0.9 |
| **Precision** | Clean line prediction accuracy | 80% - 90% |
| **Recall** | Sketch line detection rate | 85% - 95% |
| **F1-Score** | Balanced precision/recall | 82% - 92% |

### Qualitative Assessment

- **Line Continuity**: Smooth, unbroken strokes
- **Noise Reduction**: Elimination of rough artifacts  
- **Detail Preservation**: Retention of important features
- **Edge Sharpness**: Clean, well-defined boundaries

## 📈 Model Performance Analysis

### Training Visualization

The system generates comprehensive training analytics:

```python
# Loss curves
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')

# Accuracy trends  
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')

# Custom metrics
plt.plot(pmi_scores, label='Positive Match Index')
plt.plot(gmi_scores, label='General Match Index')
```

### Expected Performance Benchmarks

| Training Stage | Loss Range | Accuracy Range | Notes |
|----------------|------------|----------------|-------|
| **Initial (1-10 epochs)** | 0.8 - 0.6 | 60% - 75% | Basic pattern learning |
| **Intermediate (11-50)** | 0.6 - 0.4 | 75% - 85% | Feature refinement |
| **Advanced (51-100+)** | 0.4 - 0.2 | 85% - 95% | Fine detail mastery |

## 🔧 Advanced Configuration

### Memory Optimization

```python
# Gradient checkpointing for large models
tf.config.experimental.enable_memory_growth(gpu)

# Mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Dataset pipeline optimization
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

### Hyperparameter Tuning

```python
# Learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)

# Data augmentation options
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05)
])
```

## 🐛 Troubleshooting

### Common Issues & Solutions

1. **Out of Memory (OOM) Errors**
   ```python
   # Reduce batch size
   BATCH_SIZE = 4  # Instead of 6
   
   # Enable memory growth
   tf.config.experimental.enable_memory_growth(gpu)
   ```

2. **Poor Convergence**
   ```python
   # Adjust learning rate
   LEARNING_RATE = 0.0005  # Instead of 0.001
   
   # Increase model capacity
   FILTERS = [64, 128, 256, 512]  # Instead of [32, 64, 128, 256]
   ```

3. **Overfitting**
   ```python
   # Add dropout layers
   dropout_rate = 0.2
   
   # Implement data augmentation
   # Reduce model complexity
   ```

4. **Slow Training**
   ```python
   # Use mixed precision
   tf.keras.mixed_precision.set_global_policy('mixed_float16')
   
   # Optimize data pipeline
   dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
   ```

## 🔬 Research Applications

### Potential Use Cases

- **Digital Art Production**: Professional sketch refinement
- **Animation Pre-production**: Clean key frame generation
- **Architectural Drawing**: Technical sketch enhancement
- **Educational Tools**: Drawing skill development assistance
- **Game Development**: Concept art pipeline integration

### Academic Extensions

- **Style Transfer**: Combine with neural style transfer
- **Multi-resolution Training**: Hierarchical detail enhancement  
- **Temporal Consistency**: Video sketch cleanup
- **Interactive Refinement**: User-guided enhancement

## 📚 Technical References

### Core Research Papers

- **U-Net**: Ronneberger, O. et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Skip Connections**: He, K. et al. "Deep Residual Learning for Image Recognition" (2016)
- **Image-to-Image Translation**: Isola, P. et al. "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
- **Sketch Processing**: Simo-Serra, E. et al. "Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup" (2016)

### Mathematical Foundations

- **Convolutional Neural Networks**: LeCun, Y. et al. (1989)
- **Batch Normalization**: Ioffe, S. & Szegedy, C. (2015)
- **Adam Optimization**: Kingma, D.P. & Ba, J. (2014)

## 🤝 Contributing

### Development Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/architecture-improvement`)
3. Implement changes with proper documentation  
4. Add unit tests for new functionality
5. Submit pull request with detailed description

### Contribution Areas

- **Architecture Improvements**: New layer types, attention mechanisms
- **Loss Function Enhancements**: Advanced perceptual losses
- **Data Augmentation**: Sketch-specific transformations
- **Evaluation Metrics**: New quality assessment methods
- **Performance Optimization**: Speed and memory improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Computer Vision Community** for foundational research
- **TensorFlow Team** for deep learning framework
- **OpenCV Contributors** for image processing utilities
- **Digital Art Community** for inspiration and use case validation

## ⚠️ Important Notes

### For Production Use
- **Model Validation**: Extensive testing on diverse sketch styles
- **Performance Monitoring**: Real-time quality assessment
- **Scalability**: Consider cloud deployment for high throughput
- **User Experience**: Interactive feedback mechanisms

### For Research Use  
- **Reproducibility**: Set random seeds for consistent results
- **Baseline Comparisons**: Compare against state-of-the-art methods
- **Statistical Significance**: Proper experimental validation
- **Ethical Considerations**: Respect artist copyrights and attribution

---

**🎨 Key Innovation**: This project demonstrates that carefully designed loss functions combined with skip-connection architectures can achieve professional-quality sketch cleanup, bridging the gap between rough artistic concepts and polished digital art.
