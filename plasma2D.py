import pygame # <- <- <- !! you may need to pip install pygame
import sys
import numpy as np
import matplotlib.cm as colormap
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator


''' PLAN :
    - FUNCTIONS:         definitions of the functions used in this program
    - PARAMETERS:        definitions of the parameters of the 2D simulation                               <- <- <- <- help()
    - PYGAME & MAINLOOP: pygame initialisation, mainloop in which you can modify the functions called
'''


# ___________________________________________________________________________________________________________________
# ___ FUNCTIONS _____________________________________________________________________________________________________


### stuff ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

class Struct: # could be useful (wasn't)
    def __init__(self, **kwargs): 
        for key in kwargs: self.__dict__[key] = kwargs[key]

rotate  = pygame.transform.rotate # alias for epic functions which are too long
rescale = pygame.transform.scale
norm    = np.linalg.norm

def GridInterpol(Positions, Grid, GridX, GridY): # basicly like the np.interp but for 2D scalar/vector grid (field) with periodic boundaries
    extGridX = np.append(GridX, GridX[-1]+GridX[1]-GridX[0])  
    extGridY = np.append(GridY, GridY[-1]+GridY[1]-GridY[0])
    if Grid.ndim == 2: extGrid = np.pad(Grid,((0,1),(0,1)),      mode="wrap") # scalar field
    else:              extGrid = np.pad(Grid,((0,1),(0,1),(0,0)),mode="wrap") # vector field
    return RegularGridInterpolator((extGridX, extGridY), extGrid, bounds_error=False, fill_value=None)(Positions)

@np.vectorize(signature="(2),(n,n),()->()")
def __simpleScalarGridInterpol(Positions, Grid, dl): return Grid[*(Positions//dl).astype(int)]

@np.vectorize(signature="(2),(n,n,2),()->(2)")
def __simpleVectorGridInterpol(Positions, Grid, dl): return Grid[*(Positions//dl).astype(int)]

def simpleGridInterpol(Positions, Grid, dl): # this one assumes discrete positions (same as the Grid)
    if Grid.ndim == 2: return __simpleScalarGridInterpol(Positions, Grid, dl)
    else:              return __simpleVectorGridInterpol(Positions, Grid, dl)

def getGrad(Grid, dl): # like "np.stack(np.gradient(Grid, dl, dl), axis=2)" but with periodic boundaries 
    dGdx = (np.roll(Grid, -1, axis=0) - np.roll(Grid, 1, axis=0))/(2*dl)
    dGdy = (np.roll(Grid, -1, axis=1) - np.roll(Grid, 1, axis=1))/(2*dl)
    return np.stack((dGdx, dGdy), axis=2)

@np.vectorize(signature="(2),(2)->()")
def fakeVectorProduct(u,v):
    return u[0]*v[1] - u[1]*v[0]

def getFakeRotationnal(Grid, dl):
    return (np.roll(Grid, 1, axis=0)[:,:,1] - np.roll(Grid, -1, axis=0)[:,:,1] - np.roll(Grid, 1, axis=1)[:,:,0] + np.roll(Grid, -1, axis=1)[:,:,0])/(2*dl)



### physics ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

# integrators

def Euler_update(Positions, Velocities, Accelerations, dt): # need Accelerations array
    Positions  += Velocities*dt
    Velocities += Accelerations*dt

def LeapFrog_update(Positions, Velocities, Accelerations0, Accelerations1, dt): # need both Accelerations i and i-1 array
    Positions  += Velocities*dt + .5*Accelerations0*dt**2
    Velocities += .5*(Accelerations0 + Accelerations1)*dt

# Old version, does work for this version, but in fact RK4 is not very suited for this program.
#def RK4_update(Positions, Velocities, Charges, Masses, GridX, GridY, gL, gN, e0=1, iter=100, otherAccelerations=None, to_return=[]): # doesn't need Accelerations array!
#    '''to_return = ["Accelerations", "Density", "Potential", "E_field"] or "dict"'''
#    if otherAccelerations == None: otherAccelerations = np.zeros_like(Positions)
#    acc = lambda: compute_All(Positions, Charges, Masses, GridX, GridY, gL, gN, e0, iter) + otherAccelerations
#    k1v, k1a = Velocities,             acc()
#    k2v, k2a = Velocities + .5*dt*k1v, acc() + .5*dt*k1a
#    k3v, k3a = Velocities + .5*dt*k2v, acc() + .5*dt*k2a; computed = compute_All(Positions, Charges, Masses, GridX, GridY, gL, gN, e0, iter, to_return="dict"); acc4 = computed["Accelerations"] + otherAccelerations
#    k4v, k4a = Velocities +    dt*k3v, acc4  +    dt*k3a  # I didn't want it to break the nicely aligned RK4 algorithm, sorry for the length
#    Positions  += dt/6*(k1v + 2*k2v + 2*k3v + k4v)
#    Velocities += dt/6*(k1a + 2*k2a + 2*k3a + k4a)
#    if to_return == "dict": return computed
#    if len(to_return) != 0: return [computed[key] for key in to_return] if len(to_return) > 1 else computed[to_return[0]]


# electric computation 

def compute_Density(Positions, Charges, gL, gN): # we simply add up charges
    mean = np.sum(Charges)/gN**2
    Density = np.zeros([gN,gN])
    indices = (Positions//(gL/gN)).astype(int)
    np.add.at(Density, (indices[:, 0],indices[:, 1]), Charges)
    return Density - mean

def compute_Potential(Density, dl, e0=1, iter=100): # iterative algorithm
    V = np.zeros([gN,gN])
    for _ in range(iter):
        V = 1/4*(dl**2*Density/e0 + np.roll(V,1,0) + np.roll(V,-1,0) + np.roll(V,1,1) + np.roll(V,-1,1))
    return V

def compute_Potential_Fourier(Density, gN, dl, e0=1): # FFT
    K = np.fft.fftfreq(gN,d=dl)*2*np.pi # get \vec{k}
    K2 = K[:, None]**2 + K[None, :]**2
    K2[0,0] = 1e-12 # no /0
    VF = np.fft.fft2(Density)/(e0*K2) # -> F
    return np.fft.ifft2(VF).real      # -> F-1

def compute_Electric_field(Potential, dl):
    return -getGrad(Potential, dl)

def compute_Electric_Accelerations(Positions, E_field, dl, Masses, Charges):
    return simpleGridInterpol(Positions, E_field, dl) * Charges[:, np.newaxis]/Masses[:, np.newaxis] # a*q/m


# magnetic computation 

def compute_Current(Positions, Velocities, Charges, gL, gN): # J=sum(qv)
    Current = np.zeros([gN,gN,2])
    indices = (Positions//(gL/gN)).astype(int)
    np.add.at(Current, (indices[:, 0],indices[:, 1]), Velocities*Charges[:, np.newaxis])
    return Current

def compute_VectorPotential(Current, dl, mu0=1, iter=100): # basicly like for electric potential
    Ax, Ay = np.zeros([gN, gN]), np.zeros([gN, gN])
    for _ in range(iter):
        Ax = 1/4*(dl**2*Current[:,:,0]*mu0 + np.roll(Ax,1,0) + np.roll(Ax,-1,0) + np.roll(Ax,1,1) + np.roll(Ax,-1,1))
        Ay = 1/4*(dl**2*Current[:,:,1]*mu0 + np.roll(Ay,1,0) + np.roll(Ay,-1,0) + np.roll(Ay,1,1) + np.roll(Ay,-1,1))
    return np.stack([Ax,Ay],axis=2)

def compute_VectorPotential_Fourier(Current, gN, dl, mu0=1): # basicly like for electric potential as well
    Ax, Ay = np.zeros([gN, gN]), np.zeros([gN, gN])
    K = np.fft.fftfreq(gN,d=dl)*2*np.pi
    K2 = K[:, None]**2 + K[None, :]**2
    K2[0,0] = 1e-12
    AxF = np.fft.fft2(Current[:,:,0])*(mu0/K2)
    AyF = np.fft.fft2(Current[:,:,1])*(mu0/K2)
    Ax = np.fft.ifft2(AxF).real
    Ay = np.fft.ifft2(AyF).real
    return np.stack([Ax,Ay],axis=2)

def compute_Magnetic_field(VectorPotential, dl):
    return getFakeRotationnal(VectorPotential, dl)

def compute_Magnetic_Accelerations(Positions, Velocities, B_field, dl, Masses, Charges):
    return Velocities[:,[1,0]]*[1,-1] * simpleGridInterpol(Positions, B_field, dl)[:, np.newaxis] * Charges[:, np.newaxis]/Masses[:, np.newaxis] # v^B*q/m


# maxi best of

def compute_All(Positions, Charges, Masses, gL, gN, e0=1, mu0=1, iter=100, Fourier_potential=False, magnetism=False, other_Accelerations=[], to_return=["Accelerations"]):
    '''to_return = ["Accelerations", "Density", "Potential", "E_field"; 
                    if magnetism : "Current", "Vector_Potential", "B_field", "E_accelerations", "B_accelerations"] 
                    or just "dict" '''
    computed = {"Current":np.empty((gN,gN,2)), "Vector_Potential":np.empty((gN,gN,2)), 
                "B_field":np.empty((gN,gN)), "B_accelerations":np.empty((gN,gN,2))}
    
    # Electric part
    computed["Density"] = compute_Density(Positions, Charges, gL, gN)
    if Fourier_potential : computed["Potential"] = compute_Potential_Fourier(computed["Density"], gN, gL/gN, e0)
    else:                  computed["Potential"] = compute_Potential(computed["Density"], gL/gN, e0, iter)
    computed["E_field"] = compute_Electric_field(computed["Potential"], gL/gN)
    computed["E_accelerations"] = compute_Electric_Accelerations(Positions, computed["E_field"], gL/gN, Masses, Charges)

    # Magnetic part
    if magnetism :
        computed["Current"] = compute_Current(Positions, Velocities, Charges, gL, gN)
        if Fourier_potential : computed["Vector_Potential"] = compute_VectorPotential_Fourier(computed["Current"], gN, gL/gN, mu0)
        else :                 computed["Vector_Potential"] = compute_VectorPotential(computed["Current"], gL/gN, mu0, iter)
        computed["B_field"] = compute_Magnetic_field(computed["Vector_Potential"], gL/gN)
        computed["B_accelerations"] = compute_Magnetic_Accelerations(Positions, Velocities, computed["B_field"], gL/gN, Masses, Charges)
    
    # Acceleration result
        computed["Accelerations"] = computed["E_accelerations"] + computed["B_accelerations"]
    else : computed["Accelerations"] = computed["E_accelerations"]
    for acc in other_Accelerations: computed["Accelerations"] += acc

    if to_return=="dict": return computed
    return [computed[key] for key in to_return] if len(to_return) > 1 else computed[to_return[0]]


# initialisation

def RandomInit(pN, gL, max_init_velocity=0):
    '''-> Positions, Velocities'''
    Positions  =    np.random.random([pN,2])*gL
    Velocities = (2*np.random.random([pN,2])-1)*max_init_velocity
    return Positions, Velocities

def GaussianInit(pN, gL, sigma_velocity=0.1):
    '''-> Positions, Velocities'''
    Positions  = np.random.random([pN,2])*gL
    Velocities = np.random.normal(0,sigma_velocity,[pN,2])
    return Positions, Velocities

def PlasmaBallInit(pN, gL, R=0.25, sigma_velocity=0.1):
    '''-> Positions, Velocities'''
    Angles, Radii = np.random.uniform(0,2*np.pi,pN), R*np.sqrt(np.random.uniform(0,1,pN))
    X, Y = gL/2+Radii*np.cos(Angles), gL/2+Radii*np.sin(Angles)
    Positions  = np.column_stack((X,Y))
    Velocities = np.random.normal(0,sigma_velocity,[pN,2])
    return Positions, Velocities

def TwoCurrentsInit(pN, gL, v=0.5):
    '''-> Positions, Velocities'''
    Positions  = np.concatenate([ np.random.random([pN//2,2])*gL/2, (1 + np.random.random([pN-pN//2,2]))*gL/2])
    Velocities = np.array([[0,v]]*(pN//2) + [[0,-v]]*(pN-pN//2))
    return Positions, Velocities


# other epic things

def periodic_bounardy_condition(Positions, gL): # don't forget this one
    Positions %= gL

def get_energies(dl, Velocities, Masses, Positions, Potential, Magnetic_field=None):
    '''-> (kinetic, potential) energies'''
    cin = np.sum(1/2*Masses*norm(Velocities,axis=1))
    potE = np.sum(simpleGridInterpol(Positions, Potential, dl)) # compute the potential field at particle positions
    if not Magnetic_field is None:
        potB = np.sum(simpleGridInterpol(Positions, Magnetic_field, dl)**2)
        return cin, potE + potB
    return cin, potE



### draw ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

def draw_surface(screen, surface, screen_Position): # to draw surfaces at their center coordinates
    center_fix = np.array(surface.get_size())/2
    try: screen.blit(surface, screen_Position-center_fix)
    except:
        for p in screen_Position:
            screen.blit(surface, p-center_fix)
    
draw_electrons = draw_surface # in fact it is the same so just an alias is ok

def draw_Grid(screen, Grid, cm, maxGrid=None, minGrid=None): # for scalar fields
    if maxGrid == None: maxGrid = np.max(Grid)
    if minGrid == None: minGrid = np.min(Grid)
    Grid_RGB = (cm(((Grid-minGrid)/(maxGrid-minGrid)))[:, :, :3]*255).astype(np.uint8)
    surface = pygame.surfarray.make_surface(Grid_RGB)
    surface = rescale(surface, screen.get_size())
    screen.blit(surface, [0,0])

def draw_vector_Grid1(screen, vector_Grid): # draw vector field orientation as a color wheel!
    #Grid_norm = norm(vector_Grid,axis=2)
    #Grid_normed = Grid_norm/np.max(Grid_norm)
    Gx, Gy = vector_Grid[..., 0], vector_Grid[..., 1]
    H = (np.arctan2(Gy, Gx)+np.pi)/(2*np.pi)
    S, V = np.ones_like(H)*.5, np.ones_like(H) # editable here
    Grid_RGB = (mcolors.hsv_to_rgb(np.stack((H,S,V),axis=2))*255).astype(np.uint8)
    surface = pygame.surfarray.make_surface(Grid_RGB)
    surface = pygame.transform.scale(surface, screen.get_size())
    screen.blit(surface, [0,0])

def draw_vector_Grid2(screen, vector_Grid, arrow_surface): # draw tiny arrows but makes only sens for very small grids
    Gx, Gy = vector_Grid[..., 0], vector_Grid[..., 1]
    Angle = np.arctan2(Gy, Gx)
    surface = pygame.Surface(screen.get_size()).convert_alpha(); surface.fill([0,0,0,0])
    screen_size = screen.get_size()[0]
    gN = len(vector_Grid)
    for i in range(gN):
        for j in range(gN):
            draw_surface(screen, rotate(arrow_surface, np.degrees(Angle[i,j])), [(.5+i)/gN*screen_size,(.5+j)/gN*screen_size])
    screen.blit(surface, [0,0]) 
            


# ___________________________________________________________________________________________________________________
# ___ PARAMETERS ____________________________________________________________________________________________________


def help(): print('''
    Right below you can change the simulation parameters
    (number, charge and mass of particles; size of the grid; physical constants; display parameters...).

    'Positions' & 'Velocities' arrays(pN,2) can be initialized by the following functions :
    - RandomInit(pN, gL, maxVelocity) ie. linear repartition of velocities;
    - GaussianInit(pL, gL, standard deviation);
    - PlasmaBallInit(pN, gL, gL/4) ie. a plasma ball...;
    - TwoCurrentsInit(pN, gL, v) ie. funky initialisation with two high density regions passing by;
    or you can do it yourself!

    In the section "physics" of "PYGAME & MAIN LOOP" you can the algorithm sequence.
    The function compute_All() compile all the physics computations functions, you can tweak some parameters :
    - Fourier_potential = True/False ie. chose between the two types of potential derivation (if False see iter=...);
    - magnetism = True/False ie. enable/disable magnetic interactions;
    - to_return = list of strings of what does the function returns
                  -> "Accelerations", "Density", "Potential", "E_field" ("Current", "Vector_Potential", "B_field", "E_accelerations", "B_accelerations")
                  or just "dict" to return the full dict.
    
    Then you should use the integrator :
    - Euler_update(Positions, Velocities, Accelerations, dt) (with the parameters previously computed);
    - LeapFrog_update(Positions, Velocities, Accelerations0, Accelerations, dt) with 'Accelerations0' the previous 'Accelerations' array you should save;
    
    Don't forget periodic_bounardy_condition(Positions, gL) after that!
    
    After than in the section "draw" you can tweat the display part of this code :
    - draw_Grid(screen, background_scalar_field, colormap.viridis, maxGrid, minGrid) display the indicated field
      (here "scalar" but you could use the functions draw_vector_Grid1() the same way for vector field orientation)
      (and see the definition of background_scalar_field which lead to max/minGrid which cap the colormap).
    - draw_electrons(screen, electron_surface, screen_size*Positions/gL) display the tiny electrons surfaces :)
    
    Then right after that you finally have the function that compute the energies of the system. ''')


np.random.seed(10)

### particles
pN = 10000 # number
pm = 1    # mass
pq = 1    # charge
### grid
gN = 500  # cell number
gL = 1    # total width & height
### physic simulation
e0  = 0.01
mu0 = 1
time_speed = 0.3
### pygame parameters
screen_size = 1000
fps = 144

Positions, Velocities = PlasmaBallInit(pN, gL, gL/4)# GaussianInit(pN, gL, 0.05) #PlasmaBallInit(pN, gL, gL/4)
Masses  = np.zeros(pN) + pm
Charges = np.zeros(pN) + pq

GridX = np.array([(.5+i)*gL/gN for i in range(gN)])
GridY = np.array([(.5+j)*gL/gN for j in range(gN)])

# see the "physics" part of "PYGAME & MAIN LOOP" juste after


# ___________________________________________________________________________________________________________________
# ___ PYGAME & MAIN LOOP ____________________________________________________________________________________________

pygame.init()
screen = pygame.display.set_mode([screen_size, screen_size])
pygame.display.set_caption("Plasma")
pygame.display.set_icon(pygame.image.load('assets/big_electron.png'))
clock = pygame.time.Clock()

electron_surface = pygame.image.load("assets/electron.png").convert_alpha()
#arrow_surface    = pygame.image.load("assets/arrow32.png").convert_alpha()
#arrow_surface.set_alpha(100)
#electron_surface.set_alpha(50)

t = 0
maxGrid = 0
minGrid = 0
#Accelerations0 = np.zeros([pN,2]) # for LeapFrog
cin_nrg, pot_nrg = [], []
times = []
fpss  = []



# main loop ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

i_range = 1000
i = 0
while i!=i_range:
    dt = time_speed*0.03 # time_speed*clock.get_time()/1000
    t += dt
    fpss.append(clock.get_fps())
    for event in pygame.event.get() :
        if event.type == pygame.QUIT : 
            pygame.quit();sys.exit()


    ### physics ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    # most important part!!!!!
    Potential, E_field, Accelerations = compute_All(Positions, Charges, Masses, gL, gN, e0, mu0, 
                                                    iter=1000, Fourier_potential=True, magnetism=False,
                                                    to_return=["Potential", "E_field", "Accelerations"]) # "Accelerations", "Density", "Potential", "E_field" ("Current", "Vector_Potential", "B_field", "E_accelerations", "B_accelerations")
    Euler_update(Positions, Velocities, Accelerations, dt)

    #LeapFrog_update(Positions, Velocities, Accelerations0, Accelerations, dt)
    #Accelerations0 = Accelerations
    
    periodic_bounardy_condition(Positions, gL)


    ### draw ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    background_scalar_field = Potential #np.linalg.norm(E_field, axis=2)

    maxGrid = max(maxGrid, np.max(background_scalar_field))
    minGrid = min(minGrid, np.min(background_scalar_field))

    #screen.fill("white")
    draw_Grid(screen, background_scalar_field, colormap.viridis, maxGrid, minGrid)
    #draw_vector_Grid1(screen, E_field)
    draw_electrons(screen, electron_surface, screen_size*Positions/gL)
    #pygame.draw.circle(screen, "black", screen_size*Positions[0]/gL, 4)  # to follow a single particle

    cin, pot = get_energies(gL/gN, Velocities, Masses, Positions, Potential) # save energy values
    cin_nrg.append(cin); pot_nrg.append(pot); times.append(t)
    if (i+1)%(i_range//100)==1: print(f"{i}  -  {round(clock.get_fps(),1)}fps  -  total_nrg = {round((cin+pot)/pN,5)}J  -  mean_v = {norm(np.sum(Velocities,axis=0)):.2e}m.s-1")

    pygame.display.update()
    clock.tick(fps)
    i += 1


#cin_nrg = np.array(cin_nrg) # save energy values
#pot_nrg = np.array(pot_nrg)
#np.savez("data.npz", Times=times, Kinetic=cin_nrg, Potential=pot_nrg, FPS=fpss)