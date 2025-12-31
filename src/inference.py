import jax
import jax.numpy as jnp
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([5.0, 6.0, 7.0, 8.0])
z = x + y

print(z)

x = jnp.array([1.0, 2.0, 3.0, 4.0])
y = jnp.array([5.0, 6.0, 7.0, 8.0])
z = x + y

print(z)