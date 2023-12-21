#include<Servo.h>
int servo=6;
int goc;

Servo myServo;

void setup()
{
  Serial.begin(9600);
  myServo.attach(servo);
  pinMode(2,OUTPUT);
}

void loop()
{	
  Serial.println("Enter data:");
  while (Serial.available() == 0) {}     //neu ko nhan dc dau vao thi cu lap while nay miet,wait for data available
  String teststr = Serial.readString();  //read until timeout
  teststr.trim();
  
  Serial.println(teststr);
  
  if (teststr=="1_L"){
    Serial.println("bat den");
    digitalWrite(2,1);
    delay(100);// dừng để chờ python xử lí
    
  }else if((teststr=="0_L")){
  	Serial.println("Tat den");
    digitalWrite(2,0);
    delay(100);
  }
  else if((teststr=="1_S")){
    myServo.write(180);
    goc=myServo.read();
    Serial.print("goc:");
    Serial.println(goc);
    delay(3000);
  }
  else if((teststr=="0_S")){
    myServo.write(0);
    goc=myServo.read();
    Serial.print("goc:");
    Serial.println(goc);
  	delay(3000);
	}
  
}