BINARY=bin
CODEDIRS =. src
INCDIRS=. ./include/

CC = g++
OPT=-O0

LDFLAGS= -fopenmp
LDLIBS= -lm -lpthread -lX11 -ljpeg -lpng

# generate files that encode make rules for the .h dependencies
DEPFLAGS=-MP -MD

# automatically add the -I onto each include directory
CFLAGS=-Wall -Wextra -g -fopenmp -O3 -DNDEBUG -std=c++11 $(foreach D,$(INCDIRSI), -I$(D)) $(OPT) $(DEPFLAGS)

# for-style iteration and regex completion
CFILES=$(foreach D,$(CODEDIRS),$(wildcard $(D)/*.cpp))

OBJECTS=$(patsubst %.cpp,%.o,$(CFILES))
DEPFILES=$(patsubst %.cpp,%.d,$(CFILES))

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CC) -o $@ $(LDFLAGS) $(OBJECTS) $(LDLIBS)

%o:%.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -rf $(BINARY) $(OBJECTS) $(DEPFILES)
