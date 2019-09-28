#include <petscts.h> //PETSc time steppers
#include <petscmat.h>
/**
 *
 * We're solving the system of ODEs 
 *
 * x' = k * Ax - x - x^2
 *
 * for scalar k and adjacency matrix A.
 */

/*problem context struct*/
typedef struct _n_prob_info *PInfo;

struct _n_prob_info
{
	Mat A;
	PetscReal k;
};

/*function that computes F(x,t) for system X' = F(X,t) */
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
	PetscErrorCode     ierr;
	PInfo              prob = (PInfo)ctx;
	PetscScalar    	   *f;
	const PetscScalar  *x;
	PetscScalar 	   xval;
	PetscInt	   N, id;

	ierr = MatMult(prob->A, X, F);CHKERRQ(ierr);
	ierr = VecScale(F, prob->k);CHKERRQ(ierr);

	VecGetArrayRead(X, &x);
	VecGetArray(F, &f);
	VecGetSize(F, &N);

	for(id = 0; id < N; ++id){
		xval = x[id];
		f[id] -= xval - xval * xval;
	}

	VecRestoreArrayRead(X, &x);
	VecRestoreArray(F, &f);

	return(0);
}

/* Jacobian of RHS, dF/dX = k * A - (1-2x) * I */
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat Z, void* ctx)
{
	PetscErrorCode     ierr;
	PInfo              prob = (PInfo)ctx;
	Mat 		   AminusJ;
	Vec                IdMultVec;
	PetscInt           N;

	VecGetSize(X, &N);
	/* get vector of 1 - 2x */
	VecDuplicate(X, &IdMultVec);
	VecSet(IdMultVec, 1.0);
	ierr = VecAXPY(IdMultVec, -2.0, X);CHKERRQ(ierr);

	/* insert IdMultVec into the diagonal of AminusJ = (1-2x)* I */
	ierr = MatCreateAIJ(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, N, N, 1, NULL, 0, NULL, AminusJ);CHKERRQ(ierr);
	ierr = MatDiagonalSet(AminusJ, IdMultVec, INSERT_VALUES);CHKERRQ(ierr);
	/* compute Jacobian */
	MatDuplicate(prob->A, MAT_COPY_VALUES, &J);
	ierr = MatScale(J, prob->k);CHKERRQ(ierr);
	ierr = MatAXPY(J, -1.0, AminusJ, SAME_NONZERO_PATTERN);CHKERRQ(ierr);

	if(Z != J){
		MatAssemblyBegin(Z, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(Z, MAT_FINAL_ASSEMBLY);
	}
	MatDestroy(&AminusJ);
	VecDestroy(&IdMultVec);
	return(0);
}

/* Jacobian of RHS w.r.t. the parameter k, J_k = k * Ax */
static PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec X, Mat J, void* ctx)
{
	PetscErrorCode     ierr;
	PInfo              prob = (PInfo)ctx;
	PetscInt           N, cols[]={0}, id;
	Vec                ax;
	PetscScalar        *Jvals;

	ierr = MatMult(prob->A, X, ax);CHKERRQ(ierr);

	VecGetSize(X, &N);
	PetscInt rows[N];
	for(id = 0; id < N; ++id){
		rows[id] = id;
	}

	VecGetArrayRead(ax, &Jvals);

	MatSetValues(J, N,rows, 1, cols, Jvals, INSERT_VALUES);

	MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);


	VecRestoreArrayRead(ax, &Jvals);
	VecDestroy(ax);

	return(0);
}







