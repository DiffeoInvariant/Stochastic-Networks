/*#include "/home/diffeoinvariant/slepc-3.12.0/include/slepceps.h"*/
#include <slepceps.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv)
{
  Mat         A;
  Vec         ure, uim, vre, vim;
  Eps         eps;
  EPSType     eps_t;
  PetscReal   err, tol, re, im;
  PetscInt    num_eval, max_iter, num_iter;
  char        filename[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  PetscBool   flag;

  SlepcInitialize(&argc, &argv, (char*)0, NULL);if(ierr) return ierr;

  PetscOptionsGetString(NULL, NULL, "--filename", filename, PETSC_MAX_PATH_LEN, &flag);

  if(!flag)
    PetscOptionsGetString(NULL, NULL, "-f", filename, PETSC_MAX_PATH_LEN, &flag);

  if(!flag)
    SETETTQ(PETSC_COMM_WORLD, 1, "Must provide a matrix with the option --filename or -f");

  /* read matrix */
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetFromOptions(A);
  MatLoad(A, viewer);
  PetscViewerDestroy(&viewer);

  /* set up eigenproblem and solver */
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, A, NULL);
  EPSSetFromOptions(eps);

  /* solve system and print/write info to terminal/file */
  ierr = EPSSolve(eps);CHKERRQ(ierr);

  EPSGetIterationNumber(eps, &num_iter);

  EPSGetEigenpair(eps, 0, &re, &im, ure, uim);
  EPSComputeError(eps, 0, EPS_RELATIVE_ERROR, &err);

  PetscPrintf(PETSC_COMM_WORLD, "Eigenproblem solved in %d iterations.\n\
                                 Largest eigenvalue: %10f%+10fi .\n\
                                 Relative error: %12g.\n",(int)num_iter, (double)re, (double)im, (double)err);
  EPSDestroy(&eps);
  MatDestroy(&A);
  VecDestroy(&ure);
  VecDestroy(&uim);

  SlepcFinalize();
  return ierr;
}
