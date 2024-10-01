void readEncoder() {
  int b = digitalRead(EncoderB);
  int increment = 0;
  if(b>0) {
    // If B is high, increment forward
    increment = 1;
  }
  else {
    // Otherwise, increment backward
    increment = -1;
  }
  pos_i = pos_i + increment;

  // Compute velocity with method 2
  long currT = micros();
  float deltaT = ((float) (currT - prevT_i))/1.0e6;
  velocity_i = increment/deltaT;
  prevT_i = currT;
}
