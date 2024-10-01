
// Function to Enable Motor Driver
void Motor_E() {
  Wire.beginTransmission(88);
  Wire.write(0x03);
  Wire.write(100); // Set motor enable value (adjust as needed)
  Wire.endTransmission();
}

// Function to Write the speed and the Direction to the motor controller
void Motor_V(int V, int Dir) {
  Wire.beginTransmission(88);
  Wire.write(0x02); // Command byte for speed
  Wire.write(V);    // Speed value
  Wire.endTransmission();

  Wire.beginTransmission(88);
  Wire.write(0x00); // Command byte for direction
  Wire.write(Dir);  // Direction value (1 - forward, 2 - Reverse, 0 - Stop)
  Wire.endTransmission();
}
