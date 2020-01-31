
import numpy as np

def random_argmax(vector):
  """Helper function to select argmax at random... not just first one."""
  index = np.random.choice(np.where(vector == vector.max())[0])
  return index