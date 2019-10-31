#ifndef SIS_MODEL_H
#define SIS_MODEL_H

#include <petscdmnetwork.h>


/* problem-specific context */
struct _p_UserCtx_SIS
{
  PetscScalar beta;
  PetscScalar gamma;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));

typedef struct _p_UserCtx_SIS UserCtx_SIS;

struct _p_VERTEX_SIS
{
  PetscInt    id;/* Node ID */
  PetscScalar ps; /* probability of being susceptible*/
  PetscScalar px; /* probability of being infected */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));

typedef struct _p_VERTEX_SIS *VERTEX_SIS;

extern PetscErrorCode SISSetContextFromOptions(UserCtx_SIS*);
extern PetscErrorCode SIS
  




#endif /* SIS_MODEL_H */
