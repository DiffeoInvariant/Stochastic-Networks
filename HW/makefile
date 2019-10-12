PETSC_DIR = ~/petsc
PETSC_ARCH = arch-darwin-c-debug
CC		 = clang 
CFLAGS		 = -std=c99 -O3 -Werror
 #-I$(PETSC_DIR)/include/ -I$(PETSC_DIR)/$(PETSC_ARCH)/include/

#LDFLAGS		 = -L $(PETSC_DIR)/$(PETSC_ARCH)/lib/ -lpetsc -lmpi

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

.PHONY: HW3

HW3: HW3.c
	$(CC) $(CFLAGS) HW3.c $(PETSC_CC_INCLUDES) -L $(PETSC_DIR)/$(PETSC_ARCH)/lib/ -lpetsc -lmpi -o HW3 
