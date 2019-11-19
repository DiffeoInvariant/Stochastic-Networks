PETSC_ARCH = arch-linux2-c-debug
PETSC_DIR = /home/diffeoinvariant/petsc-3.12.0

CFLAGS = -std=c99 -O3 -g -Wall -Werror

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules


KuramotoSolver: hw5.c
	$(CC) $(CFLAGS) hw5.c -o KuramotoSolver $(PETSC_CC_INCLUDES) -L $(PETSC_DIR)/$(PETSC_ARCH)/lib/ -lpetsc -lmpi -lm -ldl
