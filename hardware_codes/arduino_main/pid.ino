
float pid(float vt) {
    int pos = 0;
    float velocity2 = 0;
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      pos = pos_i;
      velocity2 = velocity_i;
    }
  
    // time difference
    currT = micros();
    long et = currT - prevT;
    float deltaT = ((float) (currT - prevT))/( 1.0e6 );
    float velocity1 = (pos - posPrev)/deltaT;
    posPrev = pos;
    prevT = currT;
  
  
    // Convert count/s to RPM
    float v1 = velocity1/600.0*60.0;
    float v2 = velocity2/600.0*60.0;
  
    
    // Low-pass filter (25 Hz cutoff)
    
    v1Filt = 0.854*v1Filt + 0.0728*v1 + 0.0728*v1Prev;
    v1Prev = v1;
    v2Filt = 0.854*v2Filt + 0.0728*v2 + 0.0728*v2Prev;
    v2Prev = v2;
  
    
    //controlled signal
    float kp = 2.4;
    float kd = 0.16;
    float ki = 9.0;
    float e = vt - v1Filt;
    //float e = vt - v1;
    
    // derivative
    float dedt = (e-eprev)/(deltaT);
    
    
    float u = kp*e + ki*eintegral + kd*dedt;
  
    //motor speed and direction
    dir = 1;
    if (u<0){
      dir = 2;
    }
    pwr = (int) fabs(u);
    //anti windup
    if(pwr > 180){
      if (e < 0 && u < 0 || e > 0 && u > 0) {
          eintegral = 0.0;
          c_i = 1;
      }
      pwr = 180;
    }
    if (c_i == 0) {
    //integral
    eintegral = eintegral + e*deltaT;
    }
    else {
      c_i = 0;
    }

    // store previous error
    eprev = e;
    /*
    Serial.print(vt);
    Serial.print(" ");
    Serial.print(v1Filt);
    Serial.print(" ");
    Serial.print(deltaT);
    Serial.println();*/
    delay(5); 
    
   return pwr, dir;
}
