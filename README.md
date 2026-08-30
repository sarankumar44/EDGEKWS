# EdgeKWS

## Edge Keyword Spotting System

EdgeKWS is a lightweight keyword spotting system designed to detect the
"Dear Folk" wake word locally using an INT8 quantized CNN model.

## Features

- Real-time microphone input
- MFCC-based audio preprocessing
- CNN keyword classification
- INT8 quantized TensorFlow Lite model
- Wake-word confidence threshold
- Low inference latency
- Offline/local inference

## Classes

The model recognizes three classes:

1. Dear Folk
2. Unknown
3. Background

## Model

Input shape:

1 × 13 × 32 × 1

Model type:

INT8 Quantized CNN

Model size:

~35.4 KB

## Results

Test accuracy:

95.92%

INT8 inference latency:

~0.335 ms

MFCC + CNN latency:

~26.85 ms

Wake-word threshold:

95%

## Pipeline

Microphone
↓
Audio preprocessing
↓
MFCC
↓
INT8 CNN
↓
Classification
↓
Wake-word detection

## Installation

Install Python and the required packages:

```bash
pip install -r requirements.txt
