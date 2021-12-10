
/*
  Stepper Motor Test
  stepper-test01.ino
  Uses MA860H or similar Stepper Driver Unit
  Has speed control & reverse switch

  DroneBot Workshop 2019
  https://dronebotworkshop.com
*/

// Defin pins

#define dirPin 2
#define stepPin 3
#define enaPin 13
#define senPin 12

// Variables

boolean ena = LOW;
boolean setdir = LOW; // Set Direction


void setup() {
  pinMode (stepPin, OUTPUT);
  pinMode (dirPin, OUTPUT);
  pinMode (enaPin, OUTPUT);
  pinMode (senPin, INPUT);
    digitalWrite(enaPin, HIGH);
      digitalWrite(dirPin, HIGH);
  Serial.begin(9600);
}

void loop() {
  digitalWrite(enaPin, digitalRead(senPin));
  Serial.println(ena);
    digitalWrite(stepPin, HIGH);
    delay(3);
    digitalWrite(stepPin, LOW);
    delay(3);

}
