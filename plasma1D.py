import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import scipy.constants as cst 
import time
from scipy.interpolate import RegularGridInterpolator, interp1d
# ======== CLASSES ==============
class make_grid : # Just because
    def __init__(self, dimension, box_length, number_of_cell) :
        self.dimension = int(dimension)
        self.box_length = int(box_length)
        self.number_of_cell = int(number_of_cell)

    def get_resolution(self) : 
        resolution = self.box_length/self.number_of_cell
        return resolution
        
# ======== PARAMETERS FOR SIMULATION ==========
N_particule = 1e3 #Nombre de particules
N_cell = 300 #Nombre de cellules
dim = 1 #dimension de la boîte
box_length = 1.0 #Longueur de la boîte
eps0 = 1.0 #cst.epsilon_0
G = 1.0 #cst.G
m = 1.0 #masse
q = np.ones(int(N_particule)) #charge
PIC_nstep = 2000

# ======== PIC LOOP =============
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
def Pic_loop(U,internal_field, q, m, phase_space, method, initial_condition, E0 = 0.0) : 
    #Create variables
    xp = U[0]
    vp = U[1]
    K = []

    #initial conditions & safety check
    if initial_condition == 'opposite velocities' : 
        vp[int(len(vp)/2):] = - vp[:int(len(vp)/2)] 
        charges = q.copy()
    if initial_condition == 'opposite charges':
        if internal_field == 'G' : 
            print('m < 0 : Change the internal field to G !')
            return None
        else : 
            charges = q.copy()
            charges[:int(N_particule)//2] = +1.0
            charges[int(N_particule)//2:] = -1.0
    else:
        charges = q.copy()
    
    if internal_field == 'G' : print('E0 was removed') #We consider neutral mass that can't be affected by an external field

    #PIC-LOOP
    dt = 0.01 #for integration methods
    step_times = [] #to count the time it takes to converge to the solutions
    start = time.time()

    for step in range (PIC_nstep) : 
        if dim == 1 : 

            print(step*100/PIC_nstep, '%')
            grid1d = make_grid(dim, box_length, N_cell)
            dx = grid1d.get_resolution()  
            
            #safety check : check that particle don't cross more than 1 cell
            good_dt = []
            for v in vp : 
                if np.abs(dx/v) < dt  : 
                    good_dt.append(np.abs(dx/v))
                    dt = min(good_dt) 
                else : pass
            
              
            #step 1 : density of the particle
            density, edges_x = np.histogram(xp, bins = N_cell, range=(0, box_length), weights = charges)
            charge_density = density/dx                              
            charge_density -= np.mean(charge_density)  

            #step 2 : Potential V 
            if internal_field == 'E' : field_cst = -1/eps0 
            if internal_field == 'G' : field_cst = +4*np.pi*G   #charges becomes the mass 
            rho_k = np.fft.fft(charge_density)
            kx = 2*np.pi*np.fft.fftfreq(N_cell, dx)
            kx[0] = 1e-12  #avoid division by 0 for the first term
            V_k = rho_k * field_cst / (kx**2)
            V_complex = np.fft.ifft(V_k)
            V = np.real(V_complex)

            #step3 : Electric field
            Field_grid = -np.gradient(V, dx)

            #step4 : Interpolate at each particule positions taking into account the boundary problem by periodicity
            x_centers = 0.5 * (edges_x[:-1] + edges_x[1:])
            x_centers_extended = np.concatenate(([x_centers[-1] - box_length],  x_centers,[x_centers[0] + box_length]))
            Fieldx_extended = np.concatenate(([Field_grid[-1]],Field_grid,[Field_grid[0]])) 
            Fieldx_interp = interp1d(x_centers_extended, Fieldx_extended, kind='cubic',  fill_value='extrapolate', bounds_error=False)
            xp_wrapped = xp % box_length
            Field_particule = Fieldx_interp(xp_wrapped)   
            if internal_field == 'E': Field_particule += E0 #add external electric field only in the electric case

            #step 5 : Update velocity and position of each particle 
            step_times.append(time.time() - start) #store the time taken for the chosen method
            if method == 'Euler' : 
                vp += (charges/m) * Field_particule * dt
                xp += vp*dt
                xp %= box_length
                
            if method == 'kick-drift-kick' : 
                v_mid = vp + 0.5*(charges/m) * Field_particule * dt
                xp += v_mid*dt
                xp %= box_length
                newField_particule = Fieldx_interp(xp)
                if internal_field == 'E' : newField_particule += E0
                vp = v_mid + 0.5*(charges/m) * newField_particule * dt

            if method == 'Verlet':
                xp += vp * dt + 0.5 * (charges/m) * Field_particule * dt**2
                xp %= box_length
                newField_particule = Fieldx_interp(xp % box_length)
                if internal_field == 'E' : newField_particule += E0
                vp += 0.5 * (charges/m) * (Field_particule + newField_particule) * dt

                
            #velocity dispersion and kinetic energy
            Kinetic_energy = 0.5*m*vp**2
            K.append(np.mean(Kinetic_energy))
              
            
            #==== PHASE SPACE ANIMATION  =====
            if (phase_space == True) & (step%20 == 0):
                axs[0].cla()
                axs[1].cla()

                if initial_condition in ['opposite velocities', 'opposite charges'] :
                    axs[0].scatter(xp[:int(len(xp)//2)], vp[:int(len(xp)//2)], color='orange', marker='.', s=1, alpha = 0.8) 
                    axs[0].scatter(xp[int(len(xp)//2):], vp[int(len(xp)//2):], color='blue', marker='.', s=1, alpha = 0.8) 
                else : 
                    axs[0].scatter(xp, vp, marker = '.', s=1)

                if internal_field == 'E' : 
                    fig.suptitle(f"Phase Space - {initial_condition} instabilities - $E_0$ = {E0}", fontsize=18)
                else : 
                    fig.suptitle(f"Phase Space - {initial_condition} instabilities", fontsize=18)

                axs[0].set_xlabel("x")
                axs[0].set_ylabel("v")
                axs[0].set_xlim(0, box_length)

                axs[1].plot(x_centers, Field_grid/np.max(np.abs(Field_grid)), color = 'black', label = 'Normalized field $\phi$(x)')
                axs[1].step(x_centers, density/np.max(np.abs(density)), color = 'black', alpha = 0.2, label ='Normalized density of particle')
                axs[1].set_xlabel("x")
                axs[1].set_xlim(0, box_length)
                axs[1].set_ylabel("$\phi / \phi_0$")
                axs[1].legend()
                #if (phase_space==True) & (step ==0):   fig.savefig(f"/home/alexisrobert/Master/Plasma_simulation/PSti_{initial_condition}_{internal_field}_Eext{E0}.pdf")
                
                plt.tight_layout()
                plt.pause(0.01) 
    #if phase_space==True :   fig.savefig(f"/home/alexisrobert/Master/Plasma_simulation/PStf_{initial_condition}_{internal_field}_Eext{E0}.pdf")  
    print('PIC-LOOP is finally over.')      
    return np.array(K), np.array(step_times)       

#========= SET SIMULATION & PLOT ===========
#launch simulation in 1D
'''Method availabe for the update of positions are : 
        - Euler, kick-drift-kick, Verlet

    Initial conditions available are (at t = 0): 
        - opposite velocities : create two streams with oppositie velocities.
        - opposite charges : create two streams with opposite charges.

    Interaction available are : 
        - E : electric field
        - G : gravitational field 

    Add external electric field : 
        - last argument of the pic_loop function, E0 = value of the electric field''' 

positions = np.array([np.random.rand(int(N_particule))*box_length for _ in range(dim)])
velocities =  np.array([np.random.uniform(0.8, 1.0, int(N_particule)) for _ in range(dim)]) 
#velocities = np.array([np.zeros(int(N_particule)) for _ in range(dim)])

initial_condition = 'opposite velocities'
E0 = 5000.0
internal_field = 'E'

def plot(plot_phase_space, compare_method) : 
    if plot_phase_space == True : 
        Pic_loop(np.array([positions[0], velocities[0] ]),internal_field, q, m, True, 'kick-drift-kick', initial_condition, E0 = E0)
        #Pic_loop(np.array([positions[0], velocities[0]]),'E', q, m, True, 'kick-drift-kick', 'opposite charge')

    if compare_method == True : 
        Kmean_lf, times_lf = Pic_loop(np.array([positions[0], velocities[0]]),internal_field, q, m, False, 'kick-drift-kick', initial_condition) 
        Kmean_e, times_e = Pic_loop(np.array([positions[0], velocities[0]]),internal_field, q, m, False, 'Euler', initial_condition)
        Kmean_v, times_v = Pic_loop(np.array([positions[0], velocities[0]]),internal_field, q, m, False, 'Verlet', initial_condition) 
        steps = np.arange(0, PIC_nstep)
        plt.figure()
        plt.grid()
        plt.plot(steps, Kmean_lf, color = 'green', label = 'kick-drift-kick')
        plt.plot(steps, Kmean_e, color = 'blue' , label = 'Euler')
        plt.plot(steps, Kmean_v + 2.0, color = 'red' , label = 'Verlet + 2.0')
        plt.ylabel('<K>')
        plt.xlabel('steps')
        if internal_field == 'E' : 
            plt.title(f'{initial_condition} - $E_0$ = {E0}')
        else : 
            plt.title(f'{initial_condition}')
        plt.legend()
        #plt.savefig(f"/home/alexisrobert/Master/Plasma_simulation/KE_{initial_condition}_{internal_field}_Ext{E0}.pdf")

        plt.figure()
        plt.grid()
        plt.plot(times_lf, steps, color='green', label='kick-drift-kick')
        plt.plot(times_e, steps, color='blue', label='Euler')
        plt.plot(times_v, steps, color='red', label='Verlet')
        plt.ylabel('Steps', fontsize = '16')
        plt.xlabel('Cumulative time (s)', fontsize = '16')
        plt.title(f'Computation time: {initial_condition} - $E_0$ = {E0}',fontsize = '16')
        plt.legend()
        #plt.savefig(f"/home/alexisrobert/Master/Plasma_simulation/Time_{initial_condition}_{internal_field}_Ext{E0}.pdf")

    
plot(plot_phase_space = True , compare_method = True) #compare_method take some time to plot
plt.show()





    



