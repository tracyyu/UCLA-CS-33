.PHONY: all run clean

SRC ?= localization.c
PROG ?= ./localization
IN_DATA ?= input.txt
VALID_DATA ?= correct.txt

FLAGS=-Wall -fopenmp -o $(PROG) -I.

ifdef GPROF
FLAGS+=-pg
else
FLAGS+=-O3
endif

LIBS=-lm

all:
	gcc $(FLAGS) $(SRC) main.c $(LIBS)
run:
	$(PROG) $(IN_DATA) $(VALID_DATA)
clean:
	rm -f core.* gmon.out $(PROG)
