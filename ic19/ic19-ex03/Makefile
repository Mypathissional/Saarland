CC=gcc
CCFLAGS=-msse -Wall
#CCFLAGS+= -g
CCFLAGS+=-O3
LDFLAGS=-lm 

all: quantisation

quantisation: ic19_quantisation.c Makefile
	$(CC) $(CCFLAGS) $(OBJECTS) ic19_quantisation.c -o ic19_quantisation $(LDFLAGS) 

%.o : %.c
	$(CC) $(CCFLAGS) -o $@ -c $<
clean:
	rm -f ic19_quantisation
	rm -f *.o *~
