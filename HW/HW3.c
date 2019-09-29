static char help[] = "Homework 3 Problem 4 code. Solves x' = kAx - x^2 - x for scalar k and graph adjacency matrix A.\n\
		      Input parameters:\n\
		      -k : scalar parameter\n\
		      -monitor (bool) : monitor the solver's progress and print to console? Default PETSC_FALSE.\n\
		      -n (int) : number of nodes in the network.\n\
		      --filename (same as -f) : filename for the adjacency matrix.\n\n"

#include <petscts.h> //PETSc time steppers
#include <petscmat.h>
#include <mpi.h>
/**
 *
 * We're solving the system of ODEs 
 *
 * x' = k * Ax - x - x^2
 *
 * for scalar k and adjacency matrix A.
 */

/*problem context struct*/
typedef struct _n_prob_info *User;

struct _n_prob_info
{
	Mat A; /* adjacency matrix */
	PetscReal k; /* k factor in the above ODE */
	bool print = true; /*print problem progress/info as it's being solved?*/

	/*	long int max_timesteps = 1E6;*/
	long int num_timesteps = 0;
	PetscReal next_output; /*for adjoint stuff*/
	PetscReal tprev;
/*	long int  N;problem size*/
};

/*function that computes F(x,t) for system X' = F(X,t) */
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
	PetscErrorCode     ierr;
	User               prob = (User)ctx;
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
	User               prob = (User)ctx;
	Mat 		   AminusJ;
	Vec                IdMultVec;
	PetscInt           N;

	VecGetSize(X, &N);
	/* get vector of 1 - 2x */
	VecDuplicate(X, &IdMultVec);
	VecSet(IdMultVec, 1.0);
	ierr = VecAXPY(IdMultVec, -2.0, X);CHKERRQ(ierr);

	/* insert IdMultVec into the diagonal of AminusJ = (1-2x)* I */
	ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, 1, NULL, 0, NULL, AminusJ);CHKERRQ(ierr);
	ierr = MatDiagonalSet(AminusJ, IdMultVec, INSERT_VALUES);CHKERRQ(ierr);
	/* compute Jacobian */
	MatDuplicate(prob->A, MAT_COPY_VALUES, &J);
	ierr = MatScale(J, prob->k);CHKERRQ(ierr);
	ierr = MatAXPY(J, -1.0, AminusJ, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

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
	User               prob = (User)ctx;
	PetscInt           id, idStart, idEnd, idLen, cols[]={0};
	Vec                ax;

	
	ierr = MatMult(prob->A, X, ax);CHKERRQ(ierr);

	VecGetOwnershipRange(X, &idStart, &idEnd);
	idLen = idEnd - idStart;
	PetscInt rows[idLen];
	for(id = 0; id < idLen; ++id){
		rows[id] = id + idStart;
	}

	PetscScalar  Jvals[idLen][1];


	VecGetArrayRead(ax, &Jvals);

	MatSetValues(J, idLen,rows, 1, cols, &Jvals[0][0], INSERT_VALUES);

	MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);


	VecRestoreArrayRead(ax, &Jvals);
	VecDestroy(ax);

	return(0);
}


static PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal t, Vec X, void* ctx)
{
	PetscErrorCode 		ierr;
	PetscReal 		dt, tprev;      
	User      		prob = (User)ctx; 

	TSGetTimeStep(ts, &dt);
	TSGetPrevTime(ts, &tprev);
	/*
	tsteps = prob->num_timesteps;
	
	if(tsteps == 0){
		// initial condition has not been set, error! 
		return(-1); //I should figure out what the appropriate PETSc error code here is, but this should never happen anyway...
	}
	else if(tsteps >= prob->max_timesteps - 1){
		//same
		return(-1);
	}
	prob->tprev = tprev;

	prob->t[tsteps] = tprev + dt;
	
	prob->xs[tsteps] = X;

	prob->num_timesteps++;
	*/
	PetscPrintf(PETSC_COMM_WORLD, "[%.1f] %D TS %.6f\n", (double)user->next_output, step, (double)t);
	return(0);
}

PetscErrorCode ApplyInitialConditions(Vec x, PetscScalar* initial_values)
{
	/* NOTE: initial_values should contain ONLY the initial values
	 * for the part of x owned by _this_ processor*/
	PetscScalar *x_ptr;
	PetscInt    N, id;
	/* I think you could also do this with VecSetValues(),
	 * but I don't wanna get all the local indices and store them in a temp array*/
	VecGetArray(x, &x_ptr);
	VecGetLocalSize(x, &N);

	for(id = 0; id < N; ++id){
		x_ptr[id] = initial_values[id];
	}

	VecRestoreArray(x, &x_ptr);
	return(0);
}


int main(int argc, char** argv)
{
	TS             		ts; /* PETSc nonlinear solver/time-stepper*/            
	Vec            		x;             
	Mat            		J;   
	Mat            		Jp;
	PetscInt       		steps;
	PetscReal      		solve_time, time_length = 10.0;
	PetscReal               step_size=0.01;
	PetscBool      		flag, monitor = PETSC_FALSE;
	char                    filename[];
	PetscMPIInt    		size, rank;
	int                     N;
	struct _n_prob_info 	user;
	PetscErrorCode          ierr=0;
	
	PetscInitialize(&arcv, &argv, NULL, help);if(ierr) return ierr;
	
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if(rank == 0){
		PetscPrintf(PETSC_COMM_WORLD, "Program initialized with %d processes.\n", size);
	}

	PetscOptionsGetReal(NULL, NULL, "-k", &user.k, &flag);
	PetscOptionsGetBool(NULL, NULL, "-monitor",&monitor, &flag);
	PetscOptionsGetString(NULL, NULL, "-f",&filename, 100, &flag); /* 100 is max length of filename in characters, including null-terminator*/

	/*create matrices and vectors*/
	MatCreateAIJ(PETSC_COMM_WORLD, &J);
	MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, N, N);
	MatSetUp(J);
	MatCreateVecs(J, &x, NULL); /*sets up the vectors x in a parallel format that plays nicely with the Jacobian)*/

	MatCreateAIJ(PETSC_COMM_WORLD, &Jp);
	MatSetSizes(Jp, PETSC_DECIDE, PETSC_DECIDE, N, 1);
	MatSetUp(Jp);

	/* TODO: read in matrix user.A (adjacency matrix) from filename, set it up, etc */

	/* create TS context */
	TSCreate(PETSC_COMM_WORLD, &ts);
	TSSetType(ts, TSRK4); /* for now, just use RK4 integrator */
	TSSetMaxTime(ts, time_length);
	TSSetRHSFunction(ts, NULL, RHSFunction, &user);
	TSSetTimeStep(ts, step_size);

	/* set Jacobian for adjoint problem */
	TSSetRHSJacobian(ts, J, J, RHSJacobian, &user);

	TSSetExactFinalTime(ts, TS_EXACTFINALTIME_INTERPOLATE); /* if the TS goes over the allotted time_length, interpolate back to it*/
	TSSetProblemType(ts, TS_NONLINEAR);

	TSSaveTrajectory(ts);

	if(monitor){
		TSMonitorSet(ts, Monitor, &user, NULL);
	}

	/* TODO: apply random initial conditions to x */

	/* solve the forward model */
	TSSolve(ts, x);
	TSGetSolveTime(ts, &solve_time);
	TSGetStepNumber(ts, &steps);

	PetscPrintf(PETSC_COMM_WORLD, "k = %g, solver took %D steps, completed solve in time %d\n", (double)user.k, steps, (double)solve_time);

	VecView(x, PETSC_VIEWER_STDOUT_WORLD);

	/* TODO: adjoint solve */





	/* cleanup */
	MatDestroy(&J);
	MatDestroy(&Jp);
	VecDestroy(&x);
	TSDestroy(&ts);

	PetscFinalize();
	return ierr;
}

	
