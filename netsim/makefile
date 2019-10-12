PETSC_DIR = ~/petsc
PETSC_ARCH = darwin-c-opt
CC		 = clang 
CFLAGS		 = -O3 -Werror
 #-I$(PETSC_DIR)/include/ -I$(PETSC_DIR)/$(PETSC_ARCH)/include/

#LDFLAGS		 = -L $(PETSC_DIR)/$(PETSC_ARCH)/lib/ -lpetsc -lmpi

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

mm2petsc: data/mm2petsc.c
	$(CC) $(CFLAGS) -c data/mm2petsc.c $(PETSC_CC_INCLUDES) -o data/mm2petsc.o
	$(CC) data/mmio.o data/mm2petsc.o -L $(PETSC_DIR)/$(PETSC_ARCH)/lib/ -lpetsc -lmpi -o mm2petsc
	rm -rf data/*.o

