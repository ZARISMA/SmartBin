#include <AccelStepper.h>
#include <Servo.h>

// Шаговый двигатель
#define STEP_PIN 12
#define DIR_PIN 11
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// Серво
Servo myServo;
int minAngle = 0;
int maxAngle = 180;
int stepAngle = 1;
int delayTime = 10;

// Состояние
bool busy = false;

// Позиции отсеков в шагах (подбираются вручную)
long positions[] = {0, 270*2, 540*2}; // пример для 3 отсеков
int numCompartments = 3;

void setup() {
  Serial.begin(9600);
  stepper.setMaxSpeed(500);    // подбирается под двигатель и груз
  stepper.setAcceleration(200);

  myServo.attach(4);

  Serial.println("Введите номер отсека (0, 1, 2):");
}

void loop() {
  if (!busy && Serial.available() > 0) {
    int compartment = Serial.parseInt(); // читаем номер отсека
    if (compartment >= 0 && compartment < numCompartments) {
      busy = true;

      // Вращаем двигатель к нужной позиции
      stepper.moveTo(positions[compartment]);
      while (stepper.distanceToGo() != 0) {
        stepper.run();
      }

      // Открываем серво
      for (int angle = minAngle; angle <= maxAngle; angle += stepAngle) {
        myServo.write(angle);
        delay(delayTime);
      }
      // Закрываем серво
      for (int angle = maxAngle; angle >= minAngle; angle -= stepAngle) {
        myServo.write(angle);
        delay(delayTime);
      }

      busy = false;
      Serial.println("Готово. Введите номер следующего отсека:");
    } else {
      Serial.println("Неверный номер отсека!");
      // очищаем буфер
      while (Serial.available() > 0) Serial.read();
    }
  }
}