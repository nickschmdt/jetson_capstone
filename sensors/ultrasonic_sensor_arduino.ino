#define TRIG_PIN 9                                                                                                                                                      PIN 9
#define ECHO_PIN 10
#define BUTTON_PIN 13  // Using the built-in pull-up resistor

long duration;
float distance;
bool lastButtonState = HIGH;

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // internal pull-up, pressed = LOW
}

void loop() {
  
  // --- Ultrasonic measurement ---
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  duration = pulseIn(ECHO_PIN, HIGH);
  distance = (duration * 0.0343) / 2.0;

  // --- Send regular distance data ---
  Serial.print("Distance:");
  Serial.print(distance);
  Serial.println(" cm");

  // --- Check button state change ---
  bool buttonState = digitalRead(BUTTON_PIN);

  if (lastButtonState == HIGH && buttonState == LOW) {
      Serial.print("BUTTON_PRESSED:");
      Serial.println(distance, 1); // send distance in cm
  }

  lastButtonState = buttonState;
  delay(200);
}