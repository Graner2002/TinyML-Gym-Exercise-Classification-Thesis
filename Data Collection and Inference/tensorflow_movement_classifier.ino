#include "Arduino_BMI270_BMM150.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"
#include <ArduinoBLE.h>

// BLE Definitions
const char* UUID_SERV = "fae3e634-7c2a-467e-a635-a4c030f90728";
const char* UUID_EXERCISE = "4067ca05-d468-4406-82b8-04011e7d7f9d";
const char* UUID_REPETITIONS = "6bfea489-ee20-4545-9b88-22fdd40fc513";

BLEService gymService(UUID_SERV);
BLEFloatCharacteristic detectedExercise(UUID_EXERCISE, BLERead | BLENotify);
BLEFloatCharacteristic numRepetitions(UUID_REPETITIONS, BLERead | BLENotify);

// Constant Values
const float ACCELERATION_THRESHOLD = 0.3;
const float GYROSCOPE_THRESHOLD = 40;
const int NUM_SAMPLES = 95;
const int MAX_SAMPLES = 300;
const int WINDOW_SIZE = 25;
const int FILTER_WINDOW_SIZE = 25;
const unsigned long REPETITION_TIMEOUT = 5000;
const unsigned long GESTURE_TIMEOUT = 3000;

// Sampling Arrays
float aX_data[MAX_SAMPLES], aY_data[MAX_SAMPLES], aZ_data[MAX_SAMPLES];
float gX_data[MAX_SAMPLES], gY_data[MAX_SAMPLES], gZ_data[MAX_SAMPLES];
float mX_data[MAX_SAMPLES], mY_data[MAX_SAMPLES], mZ_data[MAX_SAMPLES];
float mXPrev = 0, mYPrev = 0, mZPrev = 0;

// Variables
int samplesRead = 0;
int currentExercise = 100;
int previousExercise = 100;
int previousGesture = -1;
int currentGesture = 100;
int change = 0;
int firstHalf = 0;
int repetitions = 0;
bool inSet = false;
float END_THRESHOLD = 17;

unsigned long lastRepetitionTime = 0;
unsigned long lastGestureTime = 0;

enum State {
  IDLE,
  SAMPLING,
  ANALYZING
};

State state = IDLE;

// TensorFlow Lite Model and Interpreter
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
constexpr int TENSOR_ARENA_SIZE = 8 * 1024;
byte tensorArena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

// Exercise Names
const char* EXERCISE_NAMES[] = {
    "shoulder_press",
    "bent_row",
    "biceps_curl",
    "lateral_raise",
    "skull_crusher"
};

const char* EXERCISES[] = {
    "shoulder_press_half",
    "bent_row_half",
    "biceps_curl_first_half",
    "biceps_curl_second_half",
    "lateral_raise_first_half",
    "lateral_raise_second_half",
    "skull_crusher_first_half",
    "skull_crusher_second_half"
};

#define NUM_EXERCISES (sizeof(EXERCISES) / sizeof(EXERCISES[0]))

void setup() {
    Serial.begin(115200);
    if (!IMU.begin()) {
        while (1);
    }

    tflModel = tflite::GetModel(model);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
        while (1);
    }

    tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, TENSOR_ARENA_SIZE);
    tflInterpreter->AllocateTensors();
    tflInputTensor = tflInterpreter->input(0);
    tflOutputTensor = tflInterpreter->output(0);

    if (!BLE.begin()) {
        while (1);
    }

    BLE.setLocalName("EXERCISE CLASSIFIER");
    BLE.setDeviceName("Arduino Nano 33 BLE Sense");
    BLE.setAdvertisedService(gymService);

    gymService.addCharacteristic(detectedExercise);
    gymService.addCharacteristic(numRepetitions);
    BLE.addService(gymService);

    detectedExercise.writeValue(0);
    numRepetitions.writeValue(0);
    BLE.advertise();
}

void loop() {
    BLEDevice central = BLE.central();

    if (central) {
        resetVariables();

        while (central.connected()) {

            switch (state) {
              case IDLE:
                checkForMotionStart();
                break;
              case SAMPLING:
                collectIMUSamples();
                break;
              case ANALYZING:
                processSamples();
                break;
            }
            if (inSet && (millis() - lastRepetitionTime > REPETITION_TIMEOUT)) {
                detectedExercise.writeValue(-1);
                inSet = false;
            }
        }
    }
}

void resetVariables() {
    repetitions = 0;
    currentExercise = 100;
    previousExercise = 100;
    previousGesture = -1;
    currentGesture = 100;
}

void checkForMotionStart() {
    float aX, aY, aZ, gX, gY, gZ;

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(aX, aY, aZ);
        IMU.readGyroscope(gX, gY, gZ);

        float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
        float gSum = fabs(gX) + fabs(gY) + fabs(gZ);

        if (aSum >= ACCELERATION_THRESHOLD && gSum >= GYROSCOPE_THRESHOLD) {
            samplesRead = 0;
            state = SAMPLING;
        }
    }
}

void collectIMUSamples() {
    float aX, aY, aZ, gX, gY, gZ, mX, mY, mZ;

    if (samplesRead >= MAX_SAMPLES) {
      state = IDLE;
      return;
    }

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(aX, aY, aZ);
        IMU.readGyroscope(gX, gY, gZ);

        aX_data[samplesRead] = aX;
        aY_data[samplesRead] = aY;
        aZ_data[samplesRead] = aZ;
        gX_data[samplesRead] = gX;
        gY_data[samplesRead] = gY;
        gZ_data[samplesRead] = gZ;

        if (IMU.magneticFieldAvailable()) {
            IMU.readMagneticField(mX, mY, mZ);
            mX_data[samplesRead] = mX;
            mY_data[samplesRead] = mY;
            mZ_data[samplesRead] = mZ;

            mXPrev = mX;
            mYPrev = mY;
            mZPrev = mZ;
        } else {
            mX_data[samplesRead] = mXPrev;
            mY_data[samplesRead] = mYPrev;
            mZ_data[samplesRead] = mZPrev;
        }

        samplesRead++;

        if (samplesRead >= WINDOW_SIZE) {
            float filteredMagnitude = computeFilteredMagnitude(gX_data, gY_data, gZ_data, samplesRead, WINDOW_SIZE);

            if (filteredMagnitude < END_THRESHOLD) {
                if (samplesRead > 30) {
                    state = ANALYZING;
                }
            }
        }
    }
}

void processSamples() {
    if (samplesRead != NUM_SAMPLES) {
        adjustDataArrays(NUM_SAMPLES, samplesRead, gX_data, gY_data, gZ_data, aX_data, aY_data, aZ_data, mX_data, mY_data, mZ_data);
    }

    applyFilter(aX_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);
    applyFilter(aY_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);
    applyFilter(aZ_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);
    applyFilter(gX_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);
    applyFilter(gY_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);
    applyFilter(gZ_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);
    applyFilter(mX_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);
    applyFilter(mY_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);
    applyFilter(mZ_data, NUM_SAMPLES, FILTER_WINDOW_SIZE);

    if (samplesRead == NUM_SAMPLES) {
        enterDataModel(NUM_SAMPLES);

        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
            while (1);
        }

        change = 0;
        detectExercise();

        previousGesture = currentGesture;
    }

    if (currentExercise >= 0 && currentExercise < 8 && change) {
        handleRepetition();
    }

    state = IDLE;
}

void detectExercise() {
    float maxVal = 0;
    int index = 0;

    for (int i = 0; i < NUM_EXERCISES; ++i) {
        float outputValue = tflOutputTensor->data.f[i];
        if (outputValue > maxVal) {
            maxVal = outputValue;
            index = i;
        }
    }

    currentGesture = index;

    if (index > 1) {
      END_THRESHOLD = 28;
    }
    else {
      END_THRESHOLD = 8;
    }

    unsigned long currentTime = millis();

    if (index < 2 && !firstHalf) {
      currentGesture = index;
      firstHalf = 1;
    }
    else if (index < 2 && firstHalf && index == previousGesture) {
      if (currentTime - lastGestureTime <= GESTURE_TIMEOUT) {
        currentGesture = index;
        currentExercise = index;
        change = 1;
        firstHalf = 0;
      }
      else {
        currentGesture = index;
        firstHalf = 1;
      }
    }
    else if (index < 2 && firstHalf) {
      currentGesture = index;
      firstHalf = 1;
    }
    else if (index >= 2 && (index % 2 == 0) && !firstHalf) {
      currentGesture = index;
      firstHalf = 1;
    }
    else if (index >= 2 && (index % 2 == 0) && firstHalf) {
      currentGesture = index;
      firstHalf = 1;
    }
    else if (index >= 3 && (index % 2 == 1) && firstHalf && (index == previousGesture + 1)) {
      if (currentTime - lastGestureTime <= GESTURE_TIMEOUT) {
        currentGesture = index;
        currentExercise = (index-2)/2 + 2;
        change = 1;
        firstHalf = 0;
      }
      else {
        currentGesture = index;
      }
    }

    lastGestureTime = millis();
}

void handleRepetition() {
    if (currentExercise == previousExercise) {
        repetitions++;
    } else if (previousExercise == 100) {
        repetitions = 1;
    } else {
        if (inSet) {
            detectedExercise.writeValue(-1);
            inSet = false;
        }
        repetitions = 1;
    }

    if (repetitions > 1) {
        unsigned long currentTime = millis();
        if (currentTime - lastRepetitionTime <= REPETITION_TIMEOUT) {
            if (repetitions == 2) {
                inSet = true;
            }
            detectedExercise.writeValue(currentExercise);
            numRepetitions.writeValue(repetitions);
        } else {
            repetitions = 1;
        }
    }

    lastRepetitionTime = millis();
    previousExercise = currentExercise;
}

void enterDataModel(int numSamples) {
  for (int i = 0; i < numSamples; i++) {
    tflInputTensor->data.f[i * 9 + 0] = (aX_data[i] + 4.0) / 8.0;
    tflInputTensor->data.f[i * 9 + 1] = (aY_data[i] + 4.0) / 8.0;
    tflInputTensor->data.f[i * 9 + 2] = (aZ_data[i] + 4.0) / 8.0;
    tflInputTensor->data.f[i * 9 + 3] = (gX_data[i] + 2000.0) / 4000.0;
    tflInputTensor->data.f[i * 9 + 4] = (gY_data[i] + 2000.0) / 4000.0;
    tflInputTensor->data.f[i * 9 + 5] = (gZ_data[i] + 2000.0) / 4000.0;
    tflInputTensor->data.f[i * 9 + 6] = (mX_data[i] + 400.0) / 800.0;
    tflInputTensor->data.f[i * 9 + 7] = (mY_data[i] + 400.0) / 800.0;
    tflInputTensor->data.f[i * 9 + 8] = (mZ_data[i] + 400.0) / 800.0;
  }
}

void applyFilter(float* data, int numSamples, int filterWindowSize) {
    float filteredData[numSamples];

    for (int i = 0; i < numSamples; i++) {
        int start = max(0, i - filterWindowSize + 1);
        int end = i + 1;
        float sum = 0;

        for (int j = start; j < end; j++) {
            sum += data[j];
        }

        filteredData[i] = sum / (end - start);
    }

    for (int i = 0; i < numSamples; i++) {
        data[i] = filteredData[i];
    }
}

float computeFilteredMagnitude(float* gX_data, float* gY_data, float* gZ_data, int numSamples, int windowSize) {
  static float sumX = 0, sumY = 0, sumZ = 0;
  static float meanMagnitude = 0;

  if (numSamples == windowSize) {
    sumX = 0;
    sumY = 0;
    sumZ = 0;

    for (int i = 0; i < windowSize; i++) {
      sumX += gX_data[i];
      sumY += gY_data[i];
      sumZ += gZ_data[i];
    }

    float meanX = sumX / windowSize;
    float meanY = sumY / windowSize;
    float meanZ = sumZ / windowSize;

    meanMagnitude = sqrt(meanX * meanX + meanY * meanY + meanZ * meanZ);
  } else if (numSamples > windowSize) {
    int removeIndex = numSamples - windowSize - 1;
    int addIndex = numSamples - 1;

    sumX = sumX - gX_data[removeIndex] + gX_data[addIndex];
    sumY = sumY - gY_data[removeIndex] + gY_data[addIndex];
    sumZ = sumZ - gZ_data[removeIndex] + gZ_data[addIndex];

    float meanX = sumX / windowSize;
    float meanY = sumY / windowSize;
    float meanZ = sumZ / windowSize;

    meanMagnitude = sqrt(meanX * meanX + meanY * meanY + meanZ * meanZ);
  }

  return meanMagnitude;
}

void adjustDataArrays(int targetSamples, int currentSamples, float* gX_data, float* gY_data, float* gZ_data, float* aX_data, float* aY_data, float* aZ_data, float* mX_data, float* mY_data, float* mZ_data) {
  if (currentSamples > targetSamples) {
    float decimationFactor = (float)currentSamples / targetSamples;
    int index = 0;
    for (int i = 0; i < targetSamples; i++) {
      int start = round(i * decimationFactor);
      int end = round((i + 1) * decimationFactor);
      float sumGX = 0, sumGY = 0, sumGZ = 0;
      float sumAX = 0, sumAY = 0, sumAZ = 0;
      float sumMX = 0, sumMY = 0, sumMZ = 0;
      for (int j = start; j < end; j++) {
        sumGX += gX_data[j];
        sumGY += gY_data[j];
        sumGZ += gZ_data[j];
        sumAX += aX_data[j];
        sumAY += aY_data[j];
        sumAZ += aZ_data[j];
        sumMX += mX_data[j];
        sumMY += mY_data[j];
        sumMZ += mZ_data[j];
      }
      gX_data[index] = sumGX / (end - start);
      gY_data[index] = sumGY / (end - start);
      gZ_data[index] = sumGZ / (end - start);
      aX_data[index] = sumAX / (end - start);
      aY_data[index] = sumAY / (end - start);
      aZ_data[index] = sumAZ / (end - start);
      mX_data[index] = sumMX / (end - start);
      mY_data[index] = sumMY / (end - start);
      mZ_data[index] = sumMZ / (end - start);
      index++;
    }
    samplesRead = targetSamples;
  } else if (currentSamples < targetSamples) {
    float interpolationFactor = (float)currentSamples / targetSamples;
    int index = 0;
    for (int i = 0; i < currentSamples - 1; i++) {
      gX_data[index] = gX_data[i];
      gY_data[index] = gY_data[i];
      gZ_data[index] = gZ_data[i];
      aX_data[index] = aX_data[i];
      aY_data[index] = aY_data[i];
      aZ_data[index] = aZ_data[i];
      mX_data[index] = mX_data[i];
      mY_data[index] = mY_data[i];
      mZ_data[index] = mZ_data[i];
      index++;
      for (int j = 1; j < interpolationFactor; j++) {
        float ratio = (float)j / interpolationFactor;
        gX_data[index] = gX_data[i] + (gX_data[i + 1] - gX_data[i]) * ratio;
        gY_data[index] = gY_data[i] + (gY_data[i + 1] - gY_data[i]) * ratio;
        gZ_data[index] = gZ_data[i] + (gZ_data[i + 1] - gZ_data[i]) * ratio;
        aX_data[index] = aX_data[i] + (aX_data[i + 1] - aX_data[i]) * ratio;
        aY_data[index] = aY_data[i] + (aY_data[i + 1] - aY_data[i]) * ratio;
        aZ_data[index] = aZ_data[i] + (aZ_data[i + 1] - aZ_data[i]) * ratio;
        mX_data[index] = mX_data[i] + (mX_data[i + 1] - mX_data[i]) * ratio;
        mY_data[index] = mY_data[i] + (mY_data[i + 1] - mY_data[i]) * ratio;
        mZ_data[index] = mZ_data[i] + (mZ_data[i + 1] - mZ_data[i]) * ratio;
        index++;
      }
    }
    gX_data[index] = gX_data[currentSamples - 1];
    gY_data[index] = gY_data[currentSamples - 1];
    gZ_data[index] = gZ_data[currentSamples - 1];
    aX_data[index] = aX_data[currentSamples - 1];
    aY_data[index] = aY_data[currentSamples - 1];
    aZ_data[index] = aZ_data[currentSamples - 1];
    mX_data[index] = mX_data[currentSamples - 1];
    mY_data[index] = mY_data[currentSamples - 1];
    mZ_data[index] = mZ_data[currentSamples - 1];
    samplesRead = targetSamples;
  }
}