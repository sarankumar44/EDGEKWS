import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf


# ==========================================
# EdgeKWS Configuration
# ==========================================

MODEL_PATH = "dear_folk_int8.tflite"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

MIC_DEVICE = 1

THRESHOLD = 0.95

LABELS = [
    "Dear Folk",
    "Unknown",
    "Background"
]


# ==========================================
# Load INT8 TFLite Model
# ==========================================

print("Loading EdgeKWS model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully.")

print("Input shape:",
      input_details[0]["shape"])

print("Output shape:",
      output_details[0]["shape"])


# ==========================================
# MFCC Processing
# ==========================================

def extract_mfcc(audio):

    # Make sure audio is exactly 1 second
    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]


    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )


    # Force exactly 32 frames

    if mfcc.shape[1] < 32:

        mfcc = np.pad(
            mfcc,
            (
                (0, 0),
                (0, 32 - mfcc.shape[1])
            )
        )

    elif mfcc.shape[1] > 32:

        mfcc = mfcc[:, :32]


    # CNN input shape
    # (1, 13, 32, 1)

    mfcc = mfcc[
        np.newaxis,
        ...,
        np.newaxis
    ]

    return mfcc.astype(np.float32)


# ==========================================
# INT8 Prediction
# ==========================================

def predict(mfcc):

    # Input quantization
    input_scale, input_zero = \
        input_details[0]["quantization"]


    # Convert float MFCC → INT8

    mfcc_int8 = np.round(
        mfcc / input_scale + input_zero
    ).clip(
        -128,
        127
    ).astype(np.int8)


    # Give input to model

    interpreter.set_tensor(
        input_details[0]["index"],
        mfcc_int8
    )


    # Run CNN

    interpreter.invoke()


    # Get output

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )


    # Output quantization

    output_scale, output_zero = \
        output_details[0]["quantization"]


    # INT8 → probability

    probabilities = (
        output.astype(np.float32)
        - output_zero
    ) * output_scale


    prediction = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[0, prediction]
    )


    return prediction, confidence


# ==========================================
# Main Real-Time Loop
# ==========================================

print()
print("==========================================")
print("        EDGEKWS REAL-TIME DETECTOR")
print("==========================================")
print("Wake word : Dear Folk")
print("Threshold : 95%")
print("Sample rate: 16000 Hz")
print()
print("Speak 'Dear Folk'...")
print("Press CTRL+C to stop.")
print("==========================================")
print()


try:

    while True:

        # Record 1 second

        audio = sd.rec(
            SAMPLES,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=MIC_DEVICE
        )

        sd.wait()


        # Convert (16000,1) → (16000,)

        audio = audio[:, 0]


        # MFCC

        start_time = time.perf_counter()

        mfcc = extract_mfcc(audio)


        # CNN prediction

        prediction, confidence = \
            predict(mfcc)


        end_time = time.perf_counter()


        latency = (
            end_time - start_time
        ) * 1000


        # Detection

        if (
            prediction == 0
            and confidence >= THRESHOLD
        ):

            print(
                f"🔔 DEAR FOLK DETECTED! "
                f"| Confidence: "
                f"{confidence * 100:.2f}% "
                f"| Processing: "
                f"{latency:.2f} ms"
            )

        else:

            print(
                f"Prediction: "
                f"{LABELS[prediction]:10s} "
                f"| Confidence: "
                f"{confidence * 100:.2f}% "
                f"| Processing: "
                f"{latency:.2f} ms"
            )


except KeyboardInterrupt:

    print()
    print("==========================================")
    print("EdgeKWS stopped.")
    print("==========================================")