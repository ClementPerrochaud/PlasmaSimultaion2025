import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from numba import njit
from scipy.stats import maxwell



###### - Main structure of the program - ###########################################################################################################
####################################################################################################################################################

class EM_field:
  def __init__(self, grid_shape, grid_spacing, epsilon_0=1, mu_0=1):
    self.field = np.zeros(grid_shape + (3,))
    self.grid_shape = grid_shape
    self.grid_spacing = grid_spacing
    self.domain_size = np.array(grid_shape) * grid_spacing
    self.mu_0 = mu_0    
    self.epsilon_0 = epsilon_0

  def trilinear_interpolate(self, positions):
      fractional_index = positions / self.grid_spacing
      base_index = np.floor(fractional_index).astype(int)

      offset = fractional_index - base_index

      base_index %= (np.array(self.grid_shape) - 1)

      ox, oy, oz = offset[:, 0], offset[:, 1], offset[:, 2]
      corner_offsets = np.array([
          [0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [1, 1, 0],
          [0, 0, 1],
          [1, 0, 1],
          [0, 1, 1],
          [1, 1, 1],
      ])

      weights = np.stack([
          (1 - ox) * (1 - oy) * (1 - oz),
          ox       * (1 - oy) * (1 - oz),
          (1 - ox) * oy       * (1 - oz),
          ox       * oy       * (1 - oz),
          (1 - ox) * (1 - oy) * oz,
          ox       * (1 - oy) * oz,
          (1 - ox) * oy       * oz,
          ox       * oy       * oz,
      ], axis=1)

      interpolation = np.zeros((positions.shape[0], 3))
      for i, offset_vec in enumerate(corner_offsets):
        corner_index = (base_index + offset_vec) % self.grid_shape
        field_vals = self.field[
          corner_index[:, 0],
          corner_index[:, 1],
          corner_index[:, 2]
        ]
        interpolation += weights[:, i, None] * field_vals

      return interpolation

class Magnetic_Field(EM_field):
  def __init__(self, grid_shape, grid_spacing, epsilon_0=1, mu_0=1):
    super().__init__(grid_shape, grid_spacing, epsilon_0, mu_0)
    self.current_density = np.zeros(grid_shape + (3,))
    self.vector_potential = np.zeros(grid_shape + (3,))

  def compute_current_density(self, particles):
    self.current_density.fill(0.0)

    grid_pos = particles['position'] / self.grid_spacing
    i0 = np.floor(grid_pos).astype(int)
    d = grid_pos - i0

    i0 = i0 % self.grid_shape
    i1 = (i0 + 1) % self.grid_shape

    qv = particles['charge'][:, None] * particles['v_half']

    weights = [
        ((1 - d[:, 0]) * (1 - d[:, 1]) * (1 - d[:, 2]), i0[:, 0], i0[:, 1], i0[:, 2]),
        (d[:, 0] * (1 - d[:, 1]) * (1 - d[:, 2]),       i1[:, 0], i0[:, 1], i0[:, 2]),
        ((1 - d[:, 0]) * d[:, 1] * (1 - d[:, 2]),       i0[:, 0], i1[:, 1], i0[:, 2]),
        (d[:, 0] * d[:, 1] * (1 - d[:, 2]),             i1[:, 0], i1[:, 1], i0[:, 2]),
        ((1 - d[:, 0]) * (1 - d[:, 1]) * d[:, 2],       i0[:, 0], i0[:, 1], i1[:, 2]),
        (d[:, 0] * (1 - d[:, 1]) * d[:, 2],             i1[:, 0], i0[:, 1], i1[:, 2]),
        ((1 - d[:, 0]) * d[:, 1] * d[:, 2],             i0[:, 0], i1[:, 1], i1[:, 2]),
        (d[:, 0] * d[:, 1] * d[:, 2],                   i1[:, 0], i1[:, 1], i1[:, 2]),
    ]

    for w, ix, iy, iz in weights:
        np.add.at(self.current_density, (ix, iy, iz, slice(None)), qv * w[:, None])

    self.current_density -= np.mean(self.current_density, axis=(0,1,2), keepdims=True)

  def MaxwellFaraday(self, E_field, dt):
    Ex, Ey, Ez = E_field.field[..., 0], E_field.field[..., 1], E_field.field[..., 2]

    curl_x = np.gradient(Ez, self.grid_spacing, axis=1) - np.gradient(Ey, self.grid_spacing, axis=2)
    curl_y = np.gradient(Ex, self.grid_spacing, axis=2) - np.gradient(Ez, self.grid_spacing, axis=0)
    curl_z = np.gradient(Ey, self.grid_spacing, axis=0) - np.gradient(Ex, self.grid_spacing, axis=1)

    self.field[..., 0] -= dt * curl_x
    self.field[..., 1] -= dt * curl_y
    self.field[..., 2] -= dt * curl_z

  def Ampère_Poisson_Solver(self, particles, max_iterations: int = 1000, tolerance: float = 1e-3):
    dx2 = self.grid_spacing ** 2
    self.compute_current_density(particles=particles)

    for iteration in range(max_iterations):
      reference_potential = np.copy(self.vector_potential)

      self.vector_potential[1:-1, 1:-1, 1:-1] = (
        self.vector_potential[:-2, 1:-1, 1:-1, :] +  self.vector_potential[2:, 1:-1, 1:-1, :] + self.vector_potential[1:-1, :-2, 1:-1, :] + self.vector_potential[1:-1, 2:, 1:-1, :] +
        self.vector_potential[1:-1, 1:-1, :-2, :] + self.vector_potential[1:-1, 1:-1, 2:, :] + dx2*self.current_density[1:-1, 1:-1, 1:-1, :]*self.mu_0
      )/6

      self.vector_potential[0, :, :, :]  = self.vector_potential[1, :, :, :]
      self.vector_potential[-1, :, :, :] = self.vector_potential[-2, :, :, :]
      self.vector_potential[:, 0, :, :]  = self.vector_potential[:, 1, :, :]
      self.vector_potential[:, -1, :, :] = self.vector_potential[:, -2, :, :]
      self.vector_potential[:, :, 0, :]  = self.vector_potential[:, :, 1, :]
      self.vector_potential[:, :, -1, :] = self.vector_potential[:, :, -2, :]

      if np.max(np.abs(reference_potential - self.vector_potential)) < tolerance:
        break
    
    self.Field_from_vPotential()

  def Field_from_vPotential(self):
    self.field[..., 0] = np.gradient(self.vector_potential[..., 2], axis=1) - np.gradient(self.vector_potential[..., 1], axis=2)
    self.field[..., 1] = np.gradient(self.vector_potential[..., 0], axis=2) - np.gradient(self.vector_potential[..., 2], axis=0)
    self.field[..., 2] = np.gradient(self.vector_potential[..., 1], axis=0) - np.gradient(self.vector_potential[..., 0], axis=1)

class Electric_Field(EM_field):
  def __init__(self, grid_shape, grid_spacing, epsilon_0=1, mu_0=1):
    super().__init__(grid_shape, grid_spacing, epsilon_0, mu_0)
    self.potential = np.zeros(grid_shape)
    self.charge_density = np.zeros(grid_shape)
    
  def compute_charge_density(self, particles):
    self.charge_density.fill(0.0)

    grid_pos = particles['position'] / self.grid_spacing
    i0 = np.floor(grid_pos).astype(int)
    d = grid_pos - i0

    i0 = i0 % self.grid_shape
    i1 = (i0 + 1) % self.grid_shape

    # 8 corner contributions (CIC in 3D)
    weights = [
        ((1 - d[:, 0]) * (1 - d[:, 1]) * (1 - d[:, 2]), i0[:, 0], i0[:, 1], i0[:, 2]),
        (d[:, 0] * (1 - d[:, 1]) * (1 - d[:, 2]),       i1[:, 0], i0[:, 1], i0[:, 2]),
        ((1 - d[:, 0]) * d[:, 1] * (1 - d[:, 2]),       i0[:, 0], i1[:, 1], i0[:, 2]),
        (d[:, 0] * d[:, 1] * (1 - d[:, 2]),             i1[:, 0], i1[:, 1], i0[:, 2]),
        ((1 - d[:, 0]) * (1 - d[:, 1]) * d[:, 2],       i0[:, 0], i0[:, 1], i1[:, 2]),
        (d[:, 0] * (1 - d[:, 1]) * d[:, 2],             i1[:, 0], i0[:, 1], i1[:, 2]),
        ((1 - d[:, 0]) * d[:, 1] * d[:, 2],             i0[:, 0], i1[:, 1], i1[:, 2]),
        (d[:, 0] * d[:, 1] * d[:, 2],                   i1[:, 0], i1[:, 1], i1[:, 2]),
    ]

    q = particles['charge']
    for w, ix, iy, iz in weights:
        np.add.at(self.charge_density, (ix, iy, iz), q * w)

    self.charge_density -= np.mean(self.charge_density)

  def Poisson_Solver(self, particles, max_iterations: int = 1000, tolerance: float = 1e-3):
    dx2 = self.grid_spacing ** 2
    self.compute_charge_density(particles=particles)

    for iteration in range(max_iterations):
      reference_potential = np.copy(self.potential)
      
      self.potential[1:-1, 1:-1, 1:-1] = (
        self.potential[:-2, 1:-1, 1:-1] + self.potential[2:, 1:-1, 1:-1] + self.potential[1:-1, :-2, 1:-1] + self.potential[1:-1, 2:, 1:-1] + 
        self.potential[1:-1, 1:-1, :-2] + self.potential[1:-1, 1:-1, 2:] + dx2*self.charge_density[1:-1, 1:-1, 1:-1]/self.epsilon_0
      )/6

      self.potential[0, :, :]  = self.potential[1, :, :]
      self.potential[-1, :, :] = self.potential[-2, :, :]
      self.potential[:, 0, :]  = self.potential[:, 1, :]
      self.potential[:, -1, :] = self.potential[:, -2, :]
      self.potential[:, :, 0]  = self.potential[:, :, 1]
      self.potential[:, :, -1] = self.potential[:, :, -2]

      if np.max(np.abs(reference_potential - self.potential)) < tolerance:
        break
    self.Field_from_Potential()
  
  def MaxwellAmpère(self, B_field, dt):
    Bx, By, Bz = B_field.field[..., 0], B_field.field[..., 1], B_field.field[..., 2]

    curl_x = np.gradient(Bz, self.grid_spacing, axis=1) - np.gradient(By, self.grid_spacing, axis=2)
    curl_y = np.gradient(Bx, self.grid_spacing, axis=2) - np.gradient(Bz, self.grid_spacing, axis=0)
    curl_z = np.gradient(By, self.grid_spacing, axis=0) - np.gradient(Bx, self.grid_spacing, axis=1)

    curlB = np.stack((curl_x, curl_y, curl_z), axis=3)
    dE = (curlB - self.mu_0 * B_field.current_density) / self.epsilon_0

    self.field += dt * dE


  def Field_from_Potential(self):
    self.field[..., 0], self.field[..., 1], self.field[..., 2] = np.gradient(self.potential)
    self.field = - self.field

###### - Simulation loops, initialisations and forces - ############################################################################################
####################################################################################################################################################


# Simple Lorentz force that I've tried to vectorise as best I can with some heavy googling and searching
@njit
def lorentz_force(charges, masses, v_half, E_at_positions, B_at_positions):
    return (charges[:, None] / masses[:, None]) * (
        E_at_positions + np.cross(v_half, B_at_positions)
    )

def Lorenz_force(particles, E_field, B_field, B_dip_grid=None):
    pos = particles['position']
    E_at = E_field.trilinear_interpolate(pos)
    B_at = B_field.trilinear_interpolate(pos)

    if B_dip_grid is not None:
        B_dip_at = trilinear_interpolate_field(B_dip_grid, pos, E_field.grid_spacing, E_field.grid_shape)
        B_at += B_dip_at

    return lorentz_force(
        particles['charge'], particles['mass'],
        particles['v_half'], E_at, B_at
    )

# Interpolation for the added field for the dipole experiment 
def trilinear_interpolate_field(field, positions, grid_spacing, grid_shape):
    fractional_index = positions / grid_spacing
    base_index = np.floor(fractional_index).astype(int)
    offset = fractional_index - base_index
    base_index %= (np.array(grid_shape) - 1)

    ox, oy, oz = offset[:, 0], offset[:, 1], offset[:, 2]
    corner_offsets = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ])

    weights = np.stack([
        (1 - ox) * (1 - oy) * (1 - oz),
        ox       * (1 - oy) * (1 - oz),
        (1 - ox) * oy       * (1 - oz),
        ox       * oy       * (1 - oz),
        (1 - ox) * (1 - oy) * oz,
        ox       * (1 - oy) * oz,
        (1 - ox) * oy       * oz,
        ox       * oy       * oz,
    ], axis=1)

    interp = np.zeros((positions.shape[0], 3))
    for i, offset_vec in enumerate(corner_offsets):
        corner = (base_index + offset_vec) % grid_shape
        values = field[corner[:, 0], corner[:, 1], corner[:, 2]]
        interp += weights[:, i, None] * values

    return interp

# Ended up being too lazy to do multiple masses and multiple charges
def generate_particles(number):
  particles = np.zeros((int(number),), dtype=[
        ('position', float, 3),
        ('velocity', float, 3),
        ('v_half', float, 3), 
        ('charge', float),
        ('mass', float),
      ])
  return particles

# Gives a certain temperature to a plasma fluid at equilibrium, we won't see much in the system but checking that it stays consistent is good
def initiate_particles_MaxwellBoltzmann(E_field, B_field, particles, T, charge=1.0, mass=1.0, kB=1.0):
    sigma = np.sqrt(kB * T / mass)
    N = particles.shape[0]

    particles['position'] = np.random.uniform(low=0.0, high=1.0, size=(N, 3)) * E_field.domain_size

    particles['velocity'] = np.random.normal(loc=0.0, scale=sigma, size=(N, 3))

    particles['mass'].fill(mass)

    charges = np.full(N, -charge)
    charges[:N // 2] = +charge
    np.random.shuffle(charges)
    particles['charge'] = charges

    return particles

# This barely works, in truth, too large a number of particles and it just spins wildly to infinities
def initiate_particles_crossbeams(E_field, B_field, particles, opposite_charges=True, speed=1.0, charge=1.0, mass=1.0, kB=1.0):
  N = particles.shape[0]
  assert N % 6 == 0, "Number of particles must be divisible by 6"
  per_beam = N // 6
  domain = E_field.domain_size

  beams = [
      (0, 1, 0.0),        # -x beam
      (0, -1, domain[0]), # +x beam
      (1, 1, 0.0),        # -y beam
      (1, -1, domain[1]), # +y beam
      (2, 1, 0.0),        # -z beam
      (2, -1, domain[2]), # +z beam
  ]

  for i, (axis, direction, edge) in enumerate(beams):
      idx = slice(i * per_beam, (i + 1) * per_beam)
      pos = np.zeros((per_beam, 3))

      # Fixed center in perpendicular axes
      center = domain / 2.0
      for a in range(3):
          if a != axis:
              pos[:, a] = center[a]

      # Randomize only along beam axis, confined to one half
      if direction > 0:
          pos[:, axis] = np.random.uniform(0.0, 0.5 * domain[axis], per_beam)
      else:
          pos[:, axis] = np.random.uniform(0.5 * domain[axis], domain[axis], per_beam)

      particles['position'][idx] = pos

      velocity = np.zeros(3)
      velocity[axis] = -direction * speed  # toward center
      particles['velocity'][idx] = velocity
      particles['mass'][idx] = mass
      particles['charge'][idx] = charge * (1 if (i % 2 == 0 or not opposite_charges) else -1)

  return particles

# Simple leapfrog with half steps
def leapfrog_update_particles(particles, E_field, B_field, dt):
    positions = particles['position']
    v_half = particles['v_half']

    accelerations = Lorenz_force(particles, E_field, B_field)

    v_half += accelerations * dt
    positions += v_half * dt

    box_size = np.array(E_field.grid_shape) * E_field.grid_spacing
    positions %= box_size

    particles['position'] = positions
    particles['v_half'] = v_half
    particles['velocity'] = v_half  

# We need a dt small enough to not cross multiple cells at once
def compute_adaptive_dt(particles, grid_spacing, safety_factor=0.2):
    max_velocity = np.max(np.linalg.norm(particles['velocity'], axis=1))
    if max_velocity == 0:
        return np.inf 
    return safety_factor * grid_spacing / max_velocity

# Simulation loop, track idx is the number of particles we plot on the 3D animation, because large numbers of particles just renders every illegible
def run_pic_simulation(particles, E_field, B_field, total_time, initial_dt, safety_factor=0.5):
    time = 0.0
    dt = initial_dt
    position_history = []
    N_part = particles.shape[0]
    track_idx = np.random.choice(len(particles['position']), size=N_part, replace=False)
    times = []
    kinetic_energies = []
    electric_energies = []
    magnetic_energies = []
    phase_snapshots = {}

    E_field.Poisson_Solver(particles)
    B_field.Ampère_Poisson_Solver(particles)

    E0 = Lorenz_force(particles, E_field, B_field)
    a0 = (particles['charge'][:, None] / particles['mass'][:, None]) * E0
    particles['v_half'] = particles['velocity'] - 0.5 * dt * a0
    phase_snapshots['initial'] = particles[['position', 'v_half']].copy()

    while time < total_time:
        max_v = np.max(np.linalg.norm(particles['v_half'], axis=1))
        if max_v > 0:
            dt = min(dt, safety_factor * E_field.grid_spacing / max_v)

        E_field.Poisson_Solver(particles)
        B_field.Ampère_Poisson_Solver(particles)

        E_field.MaxwellAmpère(B_field, dt)
        B_field.MaxwellFaraday(E_field, dt)

        leapfrog_update_particles(particles, E_field, B_field, dt)

        times.append(time)
        kinetic_energies.append(compute_kinetic_energy(particles))
        electric_energies.append(compute_electric_energy(E_field))
        magnetic_energies.append(compute_magnetic_energy(B_field))

        time += dt
        if int(time / total_time ) % 100 == 0:
            position_history.append(particles['position'][track_idx].copy())
            TE = kinetic_energies[-1]+magnetic_energies[-1]+electric_energies[-1]
            print(f"t = {time:.3f} | KE = {kinetic_energies[-1]:.3e} | ME = {magnetic_energies[-1]:.3e} | EE = {electric_energies[-1]:.3e} | TE = {TE:.3f} | dt = {dt:.2e}")
        if abs(time - total_time / 2) < dt:
          phase_snapshots['mid'] = particles[['position', 'v_half']].copy()
    
    phase_snapshots['final'] = particles[['position', 'v_half']].copy()
    return np.array(times), np.array(kinetic_energies), np.array(electric_energies), np.array(magnetic_energies), position_history, track_idx, phase_snapshots


###### - Addition of a spinning dipole in the center of the cube - #################################################################################
####################################################################################################################################################


def rotate_vector(vector, axis, angle):
    vector = np.asarray(vector, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)

    cos = np.cos(angle)
    sin = np.sin(angle)

    cross = np.cross(axis, vector)
    dot = np.dot(axis, vector)

    return vector * cos + cross * sin + axis * dot * (1 - cos)

def dipole_moment(t, moment, axis, omega = 1e-3):
   angle = omega * t
   return rotate_vector(moment, axis, angle)

def B_from_dipole(r, E_field, moment, mu_0):
   r = r - 0.5 * np.array(E_field.domain_size)
   r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
   r_hat = r / r_norm
   r_dot_m = np.sum(r_hat * moment, axis=-1, keepdims=True)
   return mu_0 / (4 * np.pi) * (3 * r_hat * r_dot_m - moment) / (r_norm**3 + 1e-12)

def run_pic_simulation_dipole(
    particles, E_field, B_field,
    total_time, initial_dt, moment = 10,
    safety_factor=0.5, mu_0=1.0
):
    time = 0.0
    dt = initial_dt
    track_idx = np.random.choice(len(particles), size=len(particles), replace=False)
    position_history = []
    times = []
    kinetic_energies = []
    electric_energies = []
    magnetic_energies = []
    phase_snapshots = {}

    m0 = moment * np.array([0.05, 0.05, 0.90])
    omega = 5 * 2/5 * np.pi
    axis_1 = np.array([0, 0, 1])

    grid_shape = E_field.grid_shape
    grid_spacing = E_field.grid_spacing
    grid = np.indices(grid_shape).transpose(1, 2, 3, 0) * grid_spacing
    dipole_center = np.array(E_field.domain_size) / 2
    grid_r = grid - dipole_center

    E_field.Poisson_Solver(particles)
    B_field.Ampère_Poisson_Solver(particles)

    E0 = Lorenz_force(particles, E_field, B_field)
    a0 = (particles['charge'][:, None] / particles['mass'][:, None]) * E0
    particles['v_half'] = particles['velocity'] - 0.5 * dt * a0

    phase_snapshots['initial'] = particles[['position', 'v_half']].copy()


    center = np.array(E_field.domain_size) / 2
    ring_idx = track_ring_particles(particles, center, radius=2.0, count=20)
    ring_trajectories = []

    while time < total_time:
        max_v = np.max(np.linalg.norm(particles['v_half'], axis=1))
        if max_v > 0:
            dt = min(dt, safety_factor * grid_spacing / max_v)

        E_field.Poisson_Solver(particles)
        B_field.Ampère_Poisson_Solver(particles)

        m_vec = dipole_moment(time, m0, axis_1, omega)
        B_dip = B_from_dipole(grid_r, E_field, m_vec, mu_0=mu_0)

        E_field.MaxwellAmpère(B_field, dt)
        B_field.MaxwellFaraday(E_field, dt)

        force = Lorenz_force(particles, E_field, B_field, B_dip)
        a = (particles['charge'][:, None] / particles['mass'][:, None]) * force
        particles['v_half'] += a * dt
        particles['position'] += particles['v_half'] * dt
        box_size = np.array(E_field.grid_shape) * grid_spacing
        particles['position'] %= box_size
        particles['velocity'] = particles['v_half']

        times.append(time)
        ring_trajectories.append(particles['position'][ring_idx].copy())
        kinetic_energies.append(compute_kinetic_energy(particles))
        electric_energies.append(compute_electric_energy(E_field))
        magnetic_energies.append(compute_magnetic_energy(B_field))

        if int(time / total_time) % 100 == 0:
            position_history.append(particles['position'][track_idx].copy())
            TE = kinetic_energies[-1] + electric_energies[-1] + magnetic_energies[-1]
            #print(f"t = {time:.3f} | KE = {kinetic_energies[-1]:.3e} | ME = {magnetic_energies[-1]:.3e} | EE = {electric_energies[-1]:.3e} | TE = {TE:.3f} | dt = {dt:.2e}")

        if abs(time - total_time / 2) < dt:
            phase_snapshots['mid'] = particles[['position', 'v_half']].copy()

        time += dt

    phase_snapshots['final'] = particles[['position', 'v_half']].copy()

    return (
        np.array(times),
        np.array(kinetic_energies),
        np.array(electric_energies),
        np.array(magnetic_energies),
        position_history,
        track_idx,
        phase_snapshots,
        ring_trajectories
    )

def track_ring_particles(particles, center, radius, count=20):
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    ring_indices = []

    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]

        distances = np.linalg.norm(particles['position'] - np.array([x, y, z]), axis=1)
        idx = np.argmin(distances)
        ring_indices.append(idx)

    return np.array(ring_indices)

###### - Tracking the constants - ##################################################################################################################
####################################################################################################################################################

def compute_kinetic_energy(particles):
    masses = particles['mass']
    velocities = particles['v_half']
    speeds_squared = np.sum(velocities ** 2, axis=1)
    kinetic_energy = 0.5 * np.sum(masses * speeds_squared)
    return kinetic_energy

def compute_electric_energy(E_field):
    E_squared = np.sum(E_field.field ** 2, axis=3)
    cell_volume = E_field.grid_spacing ** 3
    electric_energy = 0.5 * E_field.epsilon_0 * np.sum(E_squared) * cell_volume
    return electric_energy

def compute_magnetic_energy(B_field):
    B_squared = np.sum(B_field.field ** 2, axis=3)
    cell_volume = B_field.grid_spacing ** 3
    magnetic_energy = 0.5 * np.sum(B_squared) * cell_volume / B_field.mu_0
    return magnetic_energy


###### - Plotting and visuals - ####################################################################################################################
####################################################################################################################################################

def plot_energy(times, kinetic, electric, magnetic):
    total = kinetic + electric + magnetic
    plt.figure(figsize=(8, 5))
    plt.plot(times, kinetic, label='Kinetic Energy', color="goldenrod")
    plt.plot(times, electric, label='Electric Field Energy', color="royalblue")
    plt.plot(times, magnetic, label='Magnetic Field Energy', color="mediumseagreen")
    plt.plot(times, total, label='Total Energy', color='black')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('EM field energy and total kinetic over runtime')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Wanted to see if the dipole thing was spinning particles around, but it really doesn't seem to do it.
# To see it, add a variable to replace the "_" return in the simulation loop and feed it to the function. Doesn't give cool rings though.
# Can use it to plot individual particle motions and verify that we are in fact not just in a perfect gas type of situation.
def plot_ring_trajectories(ring_trajectories):
    ring_trajectories = np.array(ring_trajectories)
    T, N, _ = ring_trajectories.shape

    plt.figure(figsize=(6, 6))
    for i in range(N):
        x = ring_trajectories[:, i, 0]
        y = ring_trajectories[:, i, 1]
        plt.plot(x, y, label=f'p{i}', linewidth=1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ring Particle Trajectories (x–y)')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def animate_particles_3D(position_history, box_size, particles, track_idx, skip=1):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, box_size[0])
    ax.set_ylim(0, box_size[1])
    ax.set_zlim(0, box_size[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Assign colors by charge
    tracked_charges = particles['charge'][track_idx]
    colors = np.where(tracked_charges > 0, 'red', 'blue')

    # Initial scatter with correct color
    dummy = np.zeros_like(tracked_charges)
    scat = ax.scatter(dummy, dummy, dummy, s=2, c=colors)

    def init():
        scat._offsets3d = ([], [], [])
        return scat,

    def update(frame):
        pos = position_history[frame]
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        scat._offsets3d = (x, y, z)
        return scat,

    ani = FuncAnimation(fig, update, frames=range(0, len(position_history), skip), init_func=init, blit=False)
    plt.tight_layout()
    plt.show()
    return ani

def plot_phase_space_snapshots(phase_snapshots):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex='col', sharey='row')
    stages = ['initial', 'mid', 'final']
    labels = ['x', 'y', 'z']
    velocity_labels = ['vx', 'vy', 'vz']
    column_titles = ['Initial', 'Midway', 'Final']

    for col, stage in enumerate(stages):
        snapshot = phase_snapshots[stage]
        pos = snapshot['position']
        vel = snapshot['v_half']

        for row in range(3):
            ax = axes[row, col]
            ax.scatter(pos[:, row], vel[:, row], s=1, alpha=0.5)

            if col == 0:
                ax.set_ylabel(velocity_labels[row])
            ax.set_xlabel(labels[row])

            ax.grid(True)

    for col in range(3):
        axes[2, col].set_title(column_titles[col], y=-0.35)

    plt.suptitle("Phase Space Snapshots: Position vs Velocity Components")
    plt.tight_layout()
    plt.show()

def plot_velocity_histograms(phase_snapshots, bins=50):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    stages = ['initial', 'mid', 'final']
    column_titles = ['Initial', 'Midway', 'Final']

    for i, stage in enumerate(stages):
        snapshot = phase_snapshots[stage]
        velocities = snapshot['v_half']
        v_magnitude = np.linalg.norm(velocities, axis=1)

        ax = axes[i]
        counts, bins_edges, _ = ax.hist(
            v_magnitude, bins=bins, density=True, alpha=0.6, color='teal'
        )

        params = maxwell.fit(v_magnitude, floc=0)
        x = np.linspace(0, np.max(v_magnitude), 200)
        pdf = maxwell.pdf(x, *params)
        ax.plot(x, pdf, 'r--', label=f'T ≈ {params[1]**2:.2f}', color="darkviolet")

        ax.set_title(f"{column_titles[i]} Velocities")
        ax.set_xlabel("|v|")
        ax.set_ylabel("Counts")
        ax.grid(True)
        ax.legend()

    plt.suptitle("Maxwell-Boltzmann fit for velocity distribution")
    plt.tight_layout()
    plt.show()

def plot_velocity_distributions_by_charge(phase_snapshots, particles, track_idx, bins=50):
    stages = ['initial', 'mid', 'final']
    colors = {'initial': 'blue', 'mid': 'orange', 'final': 'red'}

    tracked_charges = particles['charge'][track_idx]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    charge_labels = ['Positive Charges', 'Negative Charges']

    for charge_sign, ax in zip([1, -1], axes):
        for stage in stages:
            snap = phase_snapshots[stage]
            velocities = snap['v_half']

            mask = tracked_charges > 0 if charge_sign > 0 else tracked_charges < 0
            v_mag = np.linalg.norm(velocities[mask], axis=1)

            hist_vals, bin_edges = np.histogram(v_mag, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            ax.vlines(bin_centers, 0, hist_vals, color=colors[stage], alpha=1, lw=1)

            params = maxwell.fit(v_mag, floc=0)
            x = np.linspace(0, np.max(v_mag), 200)
            fit = maxwell.pdf(x, *params)
            ax.plot(x, fit, color=colors[stage], lw=2,
                    label=f"{stage} (T ≈ {params[1]**2:.2f})")

        ax.set_title(charge_labels[charge_sign < 0])
        ax.set_xlabel('|v|')
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel('Counts')
    plt.suptitle('Velocity Magnitude Distributions by Charge')
    plt.tight_layout()
    plt.show()

####################################################################################################################################################
####################################################################################################################################################
'''
All the below parameters are what is needed to run the code, I would recommend not straying from the current values.
The mu_0 choice is physically incorrect, but almost no influence from magnetic forces can be observed if scaled properly.
To run, simply uncomment the correct initialisation you wish to see, with the parameters you want (reminding you that it is VERY unstable).
The plotting functions relevant to our system have been typed out, simply comment or uncomment them as you see fit, some aren't very helpful and some are pretty disappointing/useless
(the charge distribution in particular).

'''


# --- Physical constants ---
epsilon_0 = 1.0
mu_0 = 1.0    # if we wanted to be accurate we'd put 1e-17 for the relative strength of the fields to be right
kB = 1.0
q = 1.0
m = 1.0

# --- Simulation Parameters (dimensionless) ---

N = 90                      # N^3 has to be ~~ num_particles otherwise we get massive drifts in energy
grid_shape = (N, N, N)
grid_spacing = 1.0          # changing it creates a ton of instabilities, probably packing too densely the cells
num_particles = 5004        # has to be divisible by 6 for initialisation ~5000
temperature = 10            # only used in the Boltzmann initialisation, typical range is [0.1, 10] but 10 is starting to be unstable
total_time = 10.0            # ~number of steps in the interations, can lead to a VERY long loop if not in line with the system's size (small system + many particles = very small dt)
initial_dt = 0.005          # starting time step, will be adjusted by the program if too large


# --- Initialize simulation objects ---

E_field = Electric_Field(grid_shape, grid_spacing, epsilon_0=epsilon_0, mu_0=mu_0)
B_field = Magnetic_Field(grid_shape, grid_spacing, epsilon_0=epsilon_0, mu_0=mu_0)
particles = generate_particles(num_particles)
#particles = initiate_particles_MaxwellBoltzmann(E_field, B_field, particles, temperature, charge=-q, mass=m, kB=kB)
particles = initiate_particles_crossbeams(E_field, B_field, particles)


# --- Run simulation ---

times, ke, ee, me, position_history, track_idx, phase_snapshots = run_pic_simulation(particles, E_field, B_field, total_time, initial_dt)
#times, ke, ee, me, position_history, track_idx, phase_snapshots, _ = run_pic_simulation_dipole(particles, E_field, B_field, total_time, initial_dt, moment=moment)

initial_total = ke[0] + ee[0] + me[0]
final_total = ke[-1] + ee[-1] + me[-1]
drift_percent = 100 * (final_total - initial_total) / initial_total
print(f"Total energy drift: {drift_percent:.2f}%")

#print("\n\n\n\n\n" + values)
#plot_energy(times, ke, ee, me)
#plot_phase_space_snapshots(phase_snapshots)
#plot_velocity_histograms(phase_snapshots)
#plot_velocity_distributions_by_charge(phase_snapshots, particles, track_idx)
ani = animate_particles_3D(position_history, E_field.domain_size, particles, track_idx, skip=1)
ani.save("crossing_beams.mp4", writer="ffmpeg", fps=60) # This very regularly crashes but I stole it off the internet and am way too lazy to figure out why so, euh, be careful it correctly saves?