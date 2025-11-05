#include "inflate.h"

#define INFLATE_PIN 6

void inflate_init() {
  pinMode(INFLATE_PIN, OUTPUT);
  digitalWrite(INFLATE_PIN, LOW);
}

void trigger_inflate() {
  digitalWrite(INFLATE_PIN, HIGH);
  delay(200);
  digitalWrite(INFLATE_PIN, LOW);
}
