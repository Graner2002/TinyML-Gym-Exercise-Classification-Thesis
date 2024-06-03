#include "Arduino_BMI270_BMM150.h"
#include <ArduinoBLE.h>

const float ACCELERATION_THRESHOLD = 0.3;
const float GYROSCOPE_THRESHOLD = 40;
const int NUM_SAMPLES = 60;
const int MAX_SAMPLES = 200;
const int WINDOW_SIZE = 25;
const float END_THRESHOLD = 25;

enum State {
  IDLE,
  SAMPLING,
  PRINTING
};

State state = IDLE;

int samplesRead = 0;

float aX_data[MAX_SAMPLES], aY_data[MAX_SAMPLES], aZ_data[MAX_SAMPLES];
float gX_data[MAX_SAMPLES], gY_data[MAX_SAMPLES], gZ_data[MAX_SAMPLES];
float mX_data[MAX_SAMPLES], mY_data[MAX_SAMPLES], mZ_data[MAX_SAMPLES];

float mX_prev = 0, mY_prev = 0, mZ_prev = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  if (!BLE.begin()) {
    Serial.println("Starting Bluetooth® Low Energy module failed!");
    while (1);
  }

  Serial.println("aX,aY,aZ,gX,gY,gZ,mX,mY,mZ");
}

void loop() {
  switch (state) {
    case IDLE:
      handleIdleState();
      break;
    case SAMPLING:
      handleSamplingState();
      break;
    case PRINTING:
      handlePrintingState();
      break;
  }
}

void handleIdleState() {
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float aX, aY, aZ, gX, gY, gZ;
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

void handleSamplingState() {
  if (samplesRead >= MAX_SAMPLES) {
    state = IDLE;
    return;
  }

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float aX, aY, aZ, gX, gY, gZ, mX, mY, mZ;
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

      mX_prev = mX;
      mY_prev = mY;
      mZ_prev = mZ;
    } else {
      mX_data[samplesRead] = mX_prev;
      mY_data[samplesRead] = mY_prev;
      mZ_data[samplesRead] = mZ_prev;
    }

    samplesRead++;

    if (samplesRead >= WINDOW_SIZE) {
      float filteredMagnitude = computeFilteredMagnitude(gX_data, gY_data, gZ_data, samplesRead, WINDOW_SIZE);
      if (filteredMagnitude < END_THRESHOLD) {
        if (samplesRead > 30) {
          state = PRINTING;
        } else {
          state = IDLE;
        }
      }
    }
  }
}

void handlePrintingState() {
  if (samplesRead != NUM_SAMPLES) {
    adjustDataArrays(NUM_SAMPLES, samplesRead, gX_data, gY_data, gZ_data, aX_data, aY_data, aZ_data, mX_data, mY_data, mZ_data);
  }

  for (int i = 0; i < NUM_SAMPLES; i++) {
    Serial.print(aX_data[i], 3); Serial.print(',');
    Serial.print(aY_data[i], 3); Serial.print(',');
    Serial.print(aZ_data[i], 3); Serial.print(',');
    Serial.print(gX_data[i], 3); Serial.print(',');
    Serial.print(gY_data[i], 3); Serial.print(',');
    Serial.print(gZ_data[i], 3); Serial.print(',');
    Serial.print(mX_data[i], 3); Serial.print(',');
    Serial.print(mY_data[i], 3); Serial.print(',');
    Serial.print(mZ_data[i], 3);
    Serial.println();
  }
  Serial.println();

  state = IDLE;
}

float computeFilteredMagnitude(float* gX_data, float* gY_data, float* gZ_data, int numSamples, int windowSize) {
  static float sumX = 0, sumY = 0, sumZ = 0;
  static float meanMagnitude = 0;

  if (numSamples == windowSize) {
    sumX = sumY = sumZ = 0;

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