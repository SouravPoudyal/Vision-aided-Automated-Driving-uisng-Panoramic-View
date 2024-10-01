void mesDist(int *address) {
  // Measuring
  for (int i = 0; i < 5 ; i++) {
    Wire.beginTransmission(address[i]);
    Wire.write(byte(0x00));
    Wire.write(byte(0x51));
    Wire.endTransmission();
    delay(15);
  }
}
void getDist(int *address, int* Dist) {
  // Reading
  for (int i = 0; i < 5 ; i++) {
    uint8_t dist_H = 0;
    uint8_t dist_L = 0;
    Wire.beginTransmission(address[i]);
    Wire.write(byte(0x02));
    Wire.endTransmission();
    Wire.requestFrom(address[i], 2);
    dist_H = Wire.read();
    dist_L = Wire.read();
    Dist[i] = (dist_H << 8) | dist_L;

  }
}

void plotDist(int* Dist) {
  // Print distances for plotting
  Serial.print("Distances:");
  for (int i = 0; i < 5; i++) {
    Serial.print(Dist[i]);
    if (i < 4) {
      Serial.print(",");
    }
  }
  Serial.println();
}
