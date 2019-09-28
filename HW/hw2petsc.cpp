extern "C"
{
#include <petscmat.h>
#include <petscvec.h>
#include <petscsys.h>
}
#include <random>
#include <iostream>

PetscError MatSetErdosRenyi(Mat adjmat, PetscInt num_nodes, PetscScalar edge_prob)
{
	PetscFunctionBeginUser;
	PetscError ierr;
	ierr = MatCreate(PETSC_COMM_WORLD, &adjmat); CHKERRQ(ierr);
	ierr = MatSetSizes(adjmat, PETSC_DECIDE, num_nodes, num_nodes, num_nodes); CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(adjmat,0, NULL, (PetscInt)(num_nodes * edge_prob), NULL); CHKERRQ(ierr);
	ierr = MatSetUp(adjmat); CHKERRQ(ierr);

	PetscInt rStart, rEnd;
	ierr = MatGetOwnershipRange(adjmat, &rStart, &rEnd); CHKERRQ(ierr);

	PetscInt colID[num_nodes];
	PetscScalar values[num_nodes];

	PetscInt row, col;

	std::default_random_engine gen;
    std::uniform_real_distribution<double> dis(0.0,1.0);
	//fill rows randomly
	for(row = rStart, row < rEnd; ++row){
		PetscInt nFilled = 0;
		for(col = 0; col < num_nodes; ++col){
			if(dis(gen) < edge_prob){
				values[col] = 1;
				colID[nFilled] = col;
				nFilled++;
			}
		}
		/*array of nonzero col id's*/
        PetscInt nzCol[nFilled];
        PetscInt nzId;
        for(nzId = 0; nzId < nFilled; nzId++){
            nzCol[nzId] = colID[nzId];
        }
		//set this row
		ierr = MatSetValues(adjmat, 1, &row, nFilled, nzCol, values, INSERT_VALUES); CHKERRQ(ierr);
