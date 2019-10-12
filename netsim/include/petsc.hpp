#ifndef NETSIM_PETSC_HPP
#define NETSIM_PETSC_HPP

extern "C"{
#include <petscmat.h>
#include <petscsys.h>
#include <mpi.h>
}
#include <string>
#include <memory>

namespace netsim
{
	inline namespace petsc
	{
		PetscErrorCode NetsimPetscInitialize(int& argc, char**& argv,
											const char file[]=PETSC_NULL, const char help[]=PETSC_NULL)
		{
			PetscFunctionBeginUser;
			PetscErrorCode ierr = PetscInitialize(&argc, &argv, file, help); CHKERRQ(ierr);
			PetscFunctionReturn(0);
		}

		//same as above, but sets PETSC_COMM_WORLD equal to comm (this is useful if you want to use 
		// both PETSc and PBGL in the same Netsim program
		PetscErrorCode NetsimPetscInitialize(MPI_Comm comm, int& argc, char**& argv,
											const char file[]=PETSC_NULL, const char help[]=PETSC_NULL)
		{
			PetscFunctionBeginUser;
					PETSC_COMM_WORLD = comm;

			PetscErrorCode ierr = PetscInitialize(&argc, &argv, file, help); CHKERRQ(ierr);
			PetscFunctionReturn(0);
		}



	}
}

#endif
