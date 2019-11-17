static char help[] = "Homework 5 Problem 1 code, solves the Kuramoto model.\n\n";

#include <petscts.h>
#include <petscsys.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

struct _kuramoto_prob_data {
  PetscInt     N;
  Vec          omega;
  PetscScalar  K;
  PetscComplex r;

  PetscReal    *r_history;
  PetscInt     timestep;
  PetscInt     max_timesteps;
};

typedef struct _kuramoto_prob_data *User;

static PetscErrorCode OrderParameter(User ctx, Vec theta)
{
  const PetscScalar  *theta_data;
  /*PetscComplex r = 0. + 0. * PETSC_i;*/
  PetscReal    rr, ri, rmag;
  PetscErrorCode ierr;
  PetscInt n, id;
  PetscFunctionBeginUser;

  ierr = VecGetArrayRead(theta, &theta_data);CHKERRQ(ierr);
  ierr = VecGetLocalSize(theta, &n);CHKERRQ(ierr);
  rr = 0.;
  ri = 0.;
  for(id = 0; id < n; ++id){
    /* r = abs(sum(exp(i * theta_n))) */ 
    rr += PetscCosReal(theta_data[id]);
    ri += PetscSinReal(theta_data[id]);
  }
  ierr = VecRestoreArrayRead(theta, &theta_data);CHKERRQ(ierr);
  rmag = PetscSqrtReal(rr * rr + ri * ri) / ((PetscReal)(ctx->N));
  
  ctx->r = rmag;
  ctx->r_history[ctx->timestep] = rmag;
  PetscFunctionReturn(0);
}

/* F is the output parameter */
static PetscErrorCode KuramotoRHSFunction(TS ts, PetscReal t, Vec Theta, Vec F, void* ctx)
{
  /*
    RHS function for d(theta)/dt = omega - K*r*sin(theta) 
   */
  PetscErrorCode ierr;
  User           pctx = (User)ctx;
  PetscScalar    *f;
  PetscScalar    *theta;
  PetscInt       n, m, id;
  PetscFunctionBeginUser;

  ierr = OrderParameter(pctx, Theta);

  ierr = VecCopy(pctx->omega, F);
  /*PetscPrintf(PETSC_COMM_WORLD, "F (after copying omega):\n");
    VecView(F, PETSC_VIEWER_STDOUT_WORLD);*/
  ierr = VecGetLocalSize(Theta, &n);
  ierr = VecGetLocalSize(F, &m);
  if(m != n){
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "local sizes of F and theta must match.");
  }

  ierr = VecGetArray(Theta, &theta);
  ierr = VecGetArray(F, &f);

  for(id = 0; id < n; ++id){
    f[id] -= pctx->K * pctx->r * PetscSinReal(theta[id]);
  }

  ierr = VecRestoreArray(Theta, &theta);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  pctx->timestep += 1;
  
  PetscFunctionReturn(0);
}


static PetscErrorCode UniformSample(PetscReal *x, PetscReal low, PetscReal high)
{
  PetscFunctionBeginUser;
        *x = (double)rand() / nextafter((double)RAND_MAX, DBL_MAX);

	if(low != 0.0){
		*x += low;
	}
	if(high - low != 1.0){
		*x *= (high - low);
	}

  PetscFunctionReturn(0);
}


static PetscErrorCode BoxMullerTransform(PetscReal u1, PetscReal u2, PetscReal* z1, PetscReal*z2)
{
  PetscReal prefactor;
  PetscFunctionBeginUser;

  prefactor = PetscSqrtReal(-2 * PetscLogReal(u1));
  
  *z1 = prefactor * PetscCosReal(2 * PETSC_PI * u2);
  *z2 = prefactor * PetscSinReal(2 * PETSC_PI * u2);

  PetscFunctionReturn(0);
}

static PetscErrorCode StandardNormalSample(PetscReal* samples, PetscInt num_samples)
{
  PetscInt id;
  PetscInt orig_samples = num_samples;
  if(num_samples % 2 != 0) { ++num_samples; }
  PetscReal uniform_samples[num_samples];
  PetscReal u, z1, z2;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  for(id = 0; id < num_samples; ++id){
    ierr = UniformSample(&u, 0., 1.);CHKERRQ(ierr);
    uniform_samples[id] = u;
  }

  for(id = 0; id < orig_samples - 1; id += 2){
    ierr = BoxMullerTransform(uniform_samples[id], uniform_samples[id+1], &z1, &z2);
    samples[id] = z1;
    samples[id+1] = z2;
  }

  if(orig_samples != num_samples){
    /* if the number of samples requested is odd, fill the last entry */
    ierr = BoxMullerTransform(uniform_samples[num_samples-2], uniform_samples[num_samples-1], &z1, &z2);
    samples[num_samples - 1] = z1;
  }

  PetscFunctionReturn(0);
}
    
  

int main(int argc, char** argv)
{
  PetscErrorCode             ierr=0;
  TS                         ts;
  struct _kuramoto_prob_data ctx;
  Vec                        theta;
  PetscMPIInt                size, rank;
  PetscInt                   nlocal, id;
  PetscBool                  flag, wflag;
  char                       filename[100];
  FILE                       *outfile;
  PetscReal                  *omega;
  PetscReal                  step_size=0.1;

  PetscInitialize(&argc, &argv, NULL, help); if(ierr) return ierr;

  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscOptionsGetInt(NULL, NULL, "-N", &ctx.N, &flag);
  if(!flag){
    PetscOptionsGetInt(NULL, NULL, "-n", &ctx.N, &flag);
  }

  PetscOptionsGetReal(NULL, NULL, "-K", &ctx.K, &flag);
  if(!flag){
    PetscOptionsGetReal(NULL, NULL, "-k", &ctx.K, &flag);
  }

  if(!rank){
    PetscPrintf(PETSC_COMM_WORLD, "Solving Kuramoto model with N=%d, K=%f.\n", ctx.N, ctx.K);
  }

  PetscOptionsGetString(NULL, NULL, "-f", filename, 100, &wflag);
  if(!wflag){
    PetscOptionsGetString(NULL, NULL, "--filename", filename, 100, &wflag);
  }

  PetscOptionsGetInt(NULL, NULL, "-ts_max_steps", &ctx.max_timesteps, &flag);

  ctx.r_history = malloc(ctx.max_timesteps * sizeof(PetscReal));
  ctx.timestep = 0;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, ctx.N, &theta);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, ctx.N, &ctx.omega);CHKERRQ(ierr);
  
  ierr = VecGetLocalSize(ctx.omega, &nlocal);CHKERRQ(ierr);
  /* array for holding standard normal samples */
  srand(time(NULL));
  PetscReal omega_samples[nlocal];
  ierr = StandardNormalSample(omega_samples, nlocal);CHKERRQ(ierr);

  /* apply initial condition */
  ierr = VecGetArray(ctx.omega, &omega);CHKERRQ(ierr);
  for(id = 0; id < nlocal; ++id){
    omega[id] = omega_samples[id];
  }

  ierr = VecRestoreArray(ctx.omega, &omega);

  TSCreate(PETSC_COMM_WORLD, &ts); 
  TSSetRHSFunction(ts, NULL, KuramotoRHSFunction, &ctx);
  TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
  TSSetProblemType(ts, TS_NONLINEAR);
  
  TSSetFromOptions(ts);
  
  TSSolve(ts, theta);

  if(wflag && !rank){
    /* write results to file formatted as N, K, <r> (where <r> is the time average of r)*/
    PetscReal rbar = 0.;
    PetscReal rv;
    for(id = 0; id < ctx.timestep; ++id){
      rv = ctx.r_history[id];
      /*PetscPrintf(PETSC_COMM_WORLD, "rv = %f\n", rv);*/
      rbar += rv;
    }
    rbar /= (PetscReal)ctx.timestep;

    outfile = fopen(filename, "a");
    ierr = PetscFPrintf(PETSC_COMM_WORLD, outfile, "%D, %f, %f\n", ctx.N, ctx.K, rbar);CHKERRQ(ierr);
    ierr = fclose(outfile);
  }


  VecDestroy(&theta);
  VecDestroy(&ctx.omega);
  free(ctx.r_history);
  TSDestroy(&ts);
  return ierr;
}