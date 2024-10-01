/*
Universität Siegen
Naturwissenschaftlich-Technische Fakultät
Department Elektrotechnik und Informatik
Lehrstuhl für Regelungs- und Steuerungstechnik

Master Thesis: Vision-aided Automated Driving using Panoramic View
Done By: Sourav Poudyal, 1607167
First Examiner: Prof. Dr.-Ing. habil. Michael Gerke
Supervisior: Dr.-Ing. Nasser Gyagenda
*/

#include <Wire.h>
#include <util/atomic.h> //  For the ATOMIC_BLOCK macro
#include <Servo.h>


// Variables for the encoder
#define EncoderA 2 // EncoderA - Pin 3 // interrupt pin
#define EncoderB 3 // EncoderB - Pin 4

// Motor driver address
#define MOTOR_DRIVER_ADDRESS 88; // ddress of the  motor driver
bool running = false;
bool running_1 = false;

// Variables for Servo control
#define servopin 9
Servo steering;

// Variables for storing data from ultrasonic sensors
int address[5] = {113, 114, 115, 116, 117};
int Dist[5] = {0, 0, 0, 0, 0};

const int LED_PIN = 7; //LED pin as pin 7

/*pid inputs*/
long prevT = 0;
int  posPrev = 0;

volatile int pos_i = 0;
volatile float velocity_i = 0;
volatile long prevT_i = 0;

float v1Filt = 0; 
float v1Prev = 0;
float v2Filt = 0;
float v2Prev = 0;

float eprev = 0;
float eintegral = 0;

//initialiying current time
long currT = 0;

//initialiying command
char command;
float vt;

//initialiying desired rpm
float sp = 30;


//initialiying pid outputs
int pwr = 0;
int dir = 1;

int c_i = 0; //antiwindup switch

void setup() {
    Serial.begin(9600);   // Initialize Serial communication
    Serial2.begin(9600);  // Initialize Serial1 communication
    Wire.begin();         // Initialize I2C communication
    Motor_E();            // Enable the motor driver

    // Attach Servo
    steering.attach(servopin);

    pinMode(LED_PIN, OUTPUT); //LED pin as an output
    
    pinMode(EncoderA, INPUT);
    pinMode(EncoderB, INPUT);
    attachInterrupt(digitalPinToInterrupt(EncoderA), readEncoder, RISING);
}

void loop() {

 if (Serial2.available() > 0) {
    command = Serial2.read();
    
    if (command == 'f' || command == 'b' || command == 'l' || command == 'r' || command == 's') { // Start command received
      running = true;
      Serial2.println("Code started");
    }
    else if (command == 'x') { // Stop command received
      
      digitalWrite(LED_PIN, LOW); // Turn off LED
      Motor_V(0, 0); 
      running = false;
      Serial2.println("Code stopped");
    }
  }
  if (running) {

  //int pwr = 100/3*micros()/1.0e6;
  //int dir = 1;
  
  // Set a target
  //float vt = 60*(sin(currT*2/1e6)>0);
  if (command == 'b') {
    vt= -sp;
    pwr, dir = pid(vt);
    // Run the motor forward
    Motor_V(pwr, dir);
  }
  else if (command == 'f')  {
    vt = sp;
    pwr, dir = pid(vt);
    // Run the motor forward
    Motor_V(pwr, dir);
  }
  else if (command == 'r')  {
    steering.write(120);
  }
  else if (command == 'l')  {
    steering.write(60);
  }
   else if (command == 's')  {
    steering.write(90);
  }
  
   
  // steering.write(130);
  
   if (Serial.available() > 0) {
      //steering
      // Read the input string until newline character
      String inputString = Serial.readStringUntil('\n');
      if (inputString.startsWith("on")) {
        // Extract brightness value
        String SteerString = inputString.substring(2);
        float SteerFloat = SteerString.toFloat();
    
        int steer =int(SteerFloat);
        steering.write(steer);
      }
   }
  delay(1); 
  }
}
