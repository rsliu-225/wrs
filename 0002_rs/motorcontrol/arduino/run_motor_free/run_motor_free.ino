/* Example sketch to control a stepper motor with TB6600 stepper motor driver and Arduino without a library: continuous rotation. More info: https://www.makerguides.com */
// Define stepper motor connections:
#define dirPin 2
#define stepPin 3
#define enaPin 13

float wait = 5;
boolean setdir=LOW;
int inputData;

void setup() {
  // Declare pins as output:
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enaPin, OUTPUT);
  // Set the spinning direction CW/CCW:
  digitalWrite(enaPin, HIGH);
  Serial.begin(9600);
}

void revert() {
  setdir = !setdir;
}

void loop() {
  if (Serial.available() > 0) {
    inputData = Serial.read();
    switch (inputData) {
      case '0':
        digitalWrite(enaPin, HIGH);
        Serial.println("turn off");
        break;
      case '1':
        digitalWrite(enaPin, LOW);
        Serial.println("turn on");
        break;
      case '2':
        revert();
        digitalWrite(dirPin, setdir);
        break;
      default:
        break;
    }
  }

  digitalWrite(stepPin, HIGH);
  delay(wait);
  digitalWrite(stepPin, LOW);
  delay(wait);

}
