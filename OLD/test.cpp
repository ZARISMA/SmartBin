#include <AccelStepper.h>
#include <Servo.h>

// --- Stepper Motor Pins ---
#define STEP_PIN 3
#define DIR_PIN 4
#define ENABLE_PIN 2   // connect to EN pin on driver

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// --- Servo Setup ---
Servo myServo;
int servoPin = 6;
int minAngle = 0;       // closed
int maxAngle = 120;     // open (adjust as needed)
int stepAngle = 1;
int delayTime = 10;     // ms between servo steps

// --- System State ---
bool busy = false;

// --- Bin positions in steps (adjust to your bins) ---
// Smaller step distances → less rotation
// Start small, test, then fine-tune
long positions[] = {0, 50, 100, 150};  
int numCompartments = 4;

// === Setup ===
void setup() {
  Serial.begin(9600);

  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, HIGH); // disable driver initially

  stepper.setMaxSpeed(400);
  stepper.setAcceleration(150);

  pinMode(servoPin, OUTPUT);
  digitalWrite(servoPin, LOW); // keep servo line quiet at startup

  Serial.println("SmartBin Ready!");
  Serial.println("Send: 0,1,2,3  OR  P,O,R,A");
  Serial.println("0/P=Plastic  1/O=Organic  2/R=Paper  3/A=Aluminum");
}

// === Helper: Move stepper to target position ===
void moveToPosition(long targetPosition) {
  digitalWrite(ENABLE_PIN, LOW); // enable motor
  stepper.moveTo(targetPosition);

  // This loop blocks until movement is done
  while (stepper.distanceToGo() != 0) {
    stepper.run();
  }

  digitalWrite(ENABLE_PIN, HIGH); // disable motor
}

// === Helper: Smooth servo motion (attach/detach each time) ===
void smoothServo(int fromAngle, int toAngle) {
  myServo.attach(servoPin);
  int stepDir = (toAngle > fromAngle) ? stepAngle : -stepAngle;

  for (int angle = fromAngle; angle != toAngle; angle += stepDir) {
    myServo.write(angle);
    delay(delayTime);
  }
  myServo.write(toAngle);
  delay(200);
  myServo.detach();
}

// === Main Loop ===
void loop() {
  if (!busy && Serial.available() > 0) {
    char input = Serial.read();
    while (Serial.available() > 0) Serial.read(); // clear buffer

    int compartment = -1;

    if (input == '0' || input == 'P') compartment = 0;
    else if (input == '1' || input == 'O') compartment = 1;
    else if (input == '2' || input == 'R') compartment = 2;
    else if (input == '3' || input == 'A') compartment = 3;

    if (compartment >= 0 && compartment < numCompartments) {
      busy = true;
      Serial.print("Sorting to section ");
      Serial.println(compartment);

      // ✅ Step 1: Make sure lid is closed
      Serial.println("Ensuring lid closed...");
      smoothServo(maxAngle, minAngle);

      // ✅ Step 2: Move to the target bin FIRST
      Serial.println("Moving to target bin...");
      moveToPosition(positions[compartment]);
      delay(300); // short delay to ensure motion fully settled

      // ✅ Step 3: Now open lid
      Serial.println("Opening lid...");
      smoothServo(minAngle, maxAngle);
      delay(700); // let item drop

      // ✅ Step 4: Close lid again
      Serial.println("Closing lid...");
      smoothServo(maxAngle, minAngle);
      delay(300);

      // ✅ Step 5: Return to home
      Serial.println("Returning home...");
      moveToPosition(positions[0]);
      delay(200);

      busy = false;
      Serial.println("Done. Ready for next item.");
    } else {
      Serial.print("Invalid input: ");
      Serial.println(input);
    }
  }
}



###################################################################
#include <AccelStepper.h>
#include <Servo.h>

// ---------- Stepper wiring ----------
#define STEP_PIN   3
#define DIR_PIN    4
#define ENABLE_PIN 2   // LOW = enabled on most A4988/DRV8825

// ---------- Stepper config ----------
#define FULL_STEPS_PER_REV 200        // 1.8° motor
#define MICROSTEPPING       4         // <<< LIKELY YOUR SETTING (fixes "rotates too much")
#define INVERT_DIR          false     // set true if direction is reversed

#define STEPS_PER_REV  (FULL_STEPS_PER_REV * MICROSTEPPING)
#define QUARTER_TURN   (STEPS_PER_REV / 4)
#define OFFSET_STEPS   0              // fine trim if bin 0 doesn't line up

// Motion tuning
#define MAX_SPEED_STEPS_S  2000
#define ACCEL_STEPS_S2     1200

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// ---------- Servo config ----------
Servo myServo;
const int servoPin    = 6;
const int minAngle    = 0;     // closed
const int maxAngle    = 120;   // open
const int stepAngle   = 2;     // servo sweep step
const int delayTimeMs = 8;     // ms between servo writes
const int dropDelayMs = 700;   // wait for item to fall

// ---------- State ----------
volatile bool busy = false;
int lastBin = -1;
unsigned long lastFinishedMs = 0;
const unsigned long COOLDOWN_MS = 1200;  // ignore immediate duplicate commands

long binPos[4];

void computeBinPositions() {
  binPos[0] = OFFSET_STEPS + 0 * QUARTER_TURN;  //   0°
  binPos[1] = OFFSET_STEPS + 1 * QUARTER_TURN;  //  90°
  binPos[2] = OFFSET_STEPS + 2 * QUARTER_TURN;  // 180°
  binPos[3] = OFFSET_STEPS + 3 * QUARTER_TURN;  // 270°
}

// Robust sweep (no risk of infinite loop if angles aren’t multiples of stepAngle)
void smoothServo(int fromAngle, int toAngle) {
  myServo.attach(servoPin);
  int dir = (toAngle >= fromAngle) ? stepAngle : -stepAngle;
  for (int a = fromAngle; (dir > 0) ? (a <= toAngle) : (a >= toAngle); a += dir) {
    myServo.write(a);
    delay(delayTimeMs);
  }
  delay(180);
  myServo.detach();
}

void goToSteps(long target) {
  digitalWrite(ENABLE_PIN, LOW);             // hold torque
  stepper.runToNewPosition(target);          // blocking until done
  // keep enabled to hold position; set HIGH here if you need to release
}

void serviceBin(uint8_t bin) {
  busy = true;

  // 1) Rotate to target and STOP
  goToSteps(binPos[bin]);

  // 2) Open → wait → close
  smoothServo(minAngle, maxAngle);
  delay(dropDelayMs);
  smoothServo(maxAngle, minAngle);

  busy = false;
  lastBin = bin;
  lastFinishedMs = millis();
  Serial.println("Done.");
}

void setup() {
  Serial.begin(9600);

  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);

  stepper.setMaxSpeed(MAX_SPEED_STEPS_S);
  stepper.setAcceleration(ACCEL_STEPS_S2);
  stepper.setPinsInverted(INVERT_DIR, false, false);

  // Park servo closed at boot
  myServo.attach(servoPin);
  myServo.write(minAngle);
  delay(250);
  myServo.detach();

  computeBinPositions();

  Serial.println("SmartBin (one-shot) Ready");
  Serial.print("Microstepping: "); Serial.println(MICROSTEPPING);
  Serial.print("Quarter-turn steps: "); Serial.println(QUARTER_TURN);
  Serial.println("Send: 0,1,2,3  OR  P,O,R,A");
}

void loop() {
  if (busy) return;

  if (Serial.available() > 0) {
    char c = Serial.read();
    while (Serial.available() > 0) Serial.read(); // flush CR/LF etc.

    int bin = -1;
    if (c == '0' || c == 'P') bin = 0;        // Plastic
    else if (c == '1' || c == 'O') bin = 1;   // Organic
    else if (c == '2' || c == 'R') bin = 2;   // Paper
    else if (c == '3' || c == 'A') bin = 3;   // Aluminum

    if (bin >= 0 && bin < 4) {
      // ignore immediate duplicates
      if (bin == lastBin && (millis() - lastFinishedMs) < COOLDOWN_MS) {
        Serial.println("Duplicate ignored (cooldown).");
        return;
      }
      Serial.print("Rotating to bin "); Serial.println(bin);
      serviceBin((uint8_t)bin);
    } else {
      if (c != '\r' && c != '\n') {
        Serial.print("Invalid input: "); Serial.println(c);
      }
    }
  }
}
