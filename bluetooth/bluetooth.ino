#include <ArduinoBLE.h>

// ====== UUIDs ======
#define SERVICE_UUID        "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define RX_UUID             "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define TX_UUID             "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// BLE Service
BLEService vestService(SERVICE_UUID);

// Flutter → Arduino
BLEStringCharacteristic rxChar(
  RX_UUID,
  BLEWrite,
  20
);

// Arduino → Flutter
BLEStringCharacteristic txChar(
  TX_UUID,
  BLENotify,
  20
);

void setup() {
  Serial.begin(115200);
  while (!Serial); // REQUIRED for Nano 33 BLE Sense

  Serial.println("Starting BLE...");

  if (!BLE.begin()) {
    Serial.println("BLE start failed");
    while (1);
  }

  Serial.println("BLE OK");

  BLE.setLocalName("Vest");
  BLE.setAdvertisedService(vestService);

  vestService.addCharacteristic(rxChar);
  vestService.addCharacteristic(txChar);
  BLE.addService(vestService);

  BLE.advertise();
  Serial.println("Advertising as Vest…");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to: ");
    Serial.println(central.address());

    while (central.connected()) {
      if (rxChar.written()) {
        String cmd = rxChar.value();

        Serial.print("Received from phone: ");
        Serial.println(cmd);

        txChar.writeValue("ACK: " + cmd);
      }
    }

    Serial.println("Disconnected.");
  }
}
