
#define specialByte 123
#define startMarker 124
#define endMarker 125
#define maxMessage 16
#define maxNumDigit 5
#define actionNum 3

byte bytesRecvd = 0;
byte dataSentNum = 0;
// the transmitted value of the number of bytes in the package i.e. the 2nd byte received
byte dataRecvCount = 0;

byte dataRecvd[maxMessage];
byte dataSend[maxMessage];
byte tempBuffer[maxMessage];

byte dataSendCount = 0; // the number of 'real' bytes to be sent to the PC
byte dataTotalSend = 0; // the number of bytes to send to PC taking account of encoded bytes

boolean inProgress = false;
boolean startFound = false;
boolean allReceived = false;
int actionSeq[actionNum][maxNumDigit];

//Motor
#define dirPin 2
#define stepPin 3
#define enaPin 13
#define senPin 12

float wait = 500;

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enaPin, OUTPUT);
  pinMode(senPin, INPUT);
  digitalWrite(enaPin, LOW);
  Serial.begin(9600);
  debugToPC("Arduino Ready from exp.ino");
  delay(500);
  blinkLED(5, 100);
}

void loop() {
  getSerialData();
  doAction();
}

int power(int num, int times) {
  int res = 1;
  for (int n = 0; n < times ; n++) {
    res = res * num;
  }
  return res;
}

int cvtArr2Int(int input[]) {
  int num = 0;
  for (int n = 0; n < maxNumDigit ; n++) {
    num += input[n] * power(10, maxNumDigit - 1 - n);
  }

  Serial.println(num);
  return num;
}

void doAction() {
  if (allReceived) {
    //    blinkLED(cvtArr2Int(actionSeq[0]), cvtArr2Int(actionSeq[1]));
    runMotor(cvtArr2Int(actionSeq[0]), cvtArr2Int(actionSeq[1]), cvtArr2Int(actionSeq[2]));
    dataSendCount = dataRecvCount;
    for (byte n = 0; n < dataRecvCount; n++) {
      dataSend[n] = dataRecvd[n];
    }
    dataToPC();
    delay(100);
    allReceived = false;
  }
}

void getSerialData() {
  if (Serial.available() > 0) {
    byte x = Serial.read();
    if (x == startMarker) {
      bytesRecvd = 0;
      inProgress = true;
    }

    if (inProgress) {
      tempBuffer[bytesRecvd] = x;
      bytesRecvd ++;
    }

    if (x == endMarker) {
      inProgress = false;
      allReceived = true;
      dataSentNum = tempBuffer[1];
      decodeHighBytes();
    }
  }
}

void encodeHighBytes() {
  // Copies to temBuffer[] all of the data in dataSend[]
  // and converts any bytes of 253 or more into a pair of bytes, 253 0, 253 1 or 253 2 as appropriate
  dataTotalSend = 0;
  for (byte n = 0; n < dataSendCount; n++) {
    if (dataSend[n] >= specialByte) {
      tempBuffer[dataTotalSend] = specialByte;
      dataTotalSend++;
      tempBuffer[dataTotalSend] = dataSend[n] - specialByte;
    }
    else {
      tempBuffer[dataTotalSend] = dataSend[n];
    }
    dataTotalSend++;
  }
}

void decodeHighBytes() {
  dataRecvCount = 0;
  int actionCnt = 0;
  int numCnt = 0;
  for (byte n = 2; n < bytesRecvd - 1 ; n++) {
    byte x = tempBuffer[n];
    if (x == specialByte) {
      n++;
      actionCnt++;
      numCnt = 0;
      x = x + tempBuffer[n];
    }
    else {
      actionSeq[actionCnt][numCnt] = (int)x - 48;
      numCnt++;
    }
    dataRecvd[dataRecvCount] = x;
    dataRecvCount ++;
  }
}

void blinkLED(int numBlinks, int numDelay) {
  for (byte n = 0; n < numBlinks; n ++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(numDelay);
    digitalWrite(LED_BUILTIN, LOW);
    delay(numDelay);
  }
  delay(1000);
}

void runMotor(int stepNum, int dir, int useSen) {
  int i = 0;
  if (dir == 0) {
    digitalWrite(dirPin, LOW);
    debugToPC("CLOCKWISE");
  }
  else {
    digitalWrite(dirPin, HIGH);
    debugToPC("COUNTERCLOCKWISE");
  }
  if (useSen == 1) {
    debugToPC("USESENSOR");
  }
  while (i < stepNum) {
    if (useSen == 1) {
      digitalWrite(enaPin, digitalRead(senPin));
      if (digitalRead(senPin) == HIGH) {
        break;
      }
    }
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(wait);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(wait);
    i++;
  }
  digitalWrite(enaPin, LOW);
}


void dataToPC() {
  // expects to find data in dataSend[]
  // uses encodeHighBytes() to copy data to tempBuffer
  // sends data to PC from tempBuffer
  encodeHighBytes();
  Serial.write(startMarker);
  Serial.write(dataSendCount);
  Serial.write(tempBuffer, dataTotalSend);
  Serial.write(endMarker);
}

void debugToPC(char arr[]) {
  byte nb = 0;
  Serial.write(startMarker);
  Serial.write(nb);
  Serial.print(arr);
  Serial.write(endMarker);
}

void debugToPC(byte num) {
  byte nb = 0;
  Serial.write(startMarker);
  Serial.write(nb);
  Serial.print(num);
  Serial.write(endMarker);
}
