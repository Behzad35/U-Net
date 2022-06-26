CC = g++
CFLAGS  = -O3 -Wall
TARGET = main

all: main.cpp 
	$(CC) $(CFLAGS) main.cpp -o $(TARGET)
