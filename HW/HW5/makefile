PETSC_ARCH = arch-linux2-c-debug
PETSC_DIR = /home/diffeoinvariant/petsc-3.12.0

CC = clang
CFLAGS = -std=c99 -O3 -Werror

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

.PHONY: HW5

HW5: hw5.c
	$(CC) $(CFLAGS) hw5.c -o HW5  $(PETSC_CC_INCLUDES) -L $(PETSC_DIR)/$(PETSC_ARCH)/lib/ -lpetsc -lmpi -lm -ldl
