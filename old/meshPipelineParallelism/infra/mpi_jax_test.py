from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f'hello from {rank} of {comm}')

@jax.jit
def foo(arr):
   arr = arr + rank
   arr_sum, _ = mpi4jax.allreduce(arr, op=MPI.SUM, comm=comm)
   return arr_sum

a = jnp.zeros((3, 3))
result = foo(a)

if rank == 0:
   print(result)