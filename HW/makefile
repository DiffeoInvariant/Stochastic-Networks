PETSC_ARCH = arch-linux2-c-debug
SLEPC_DIR = /home/diffeoinvariant/slepc-3.12.0
PETSC_DIR = /home/diffeoinvariant/petsc-3.12.0

CC		 = clang 
CFLAGS		 = -std=c99 -O3 -Werror
 #-I$(PETSC_DIR)/include/ -I$(PETSC_DIR)/$(PETSC_ARCH)/include/

#LDFLAGS		 = -L $(PETSC_DIR)/$(PETSC_ARCH)/lib/ -lpetsc -lmpi

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules
include $(SLEPC_DIR)/lib/slepc/conf/slepc_common

.PHONY: HW3, HW3Eigen

HW3Eigen: HW3Eigen.c
	$(CLINKER) -I/home/diffeoinvariant/slepc-3.12.0/include/slepceps.h -o HW3Eigen HW3Eigen.c $(SLEPC_EPS_LIB)

HW3: HW3.c
	$(CC) $(CFLAGS) HW3.c -o HW3  $(PETSC_CC_INCLUDES) -L $(PETSC_DIR)/$(PETSC_ARCH)/lib/ -lpetsc -lmpi -lm -ldl
