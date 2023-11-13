"""
Problem file for AVF FSI simulation
"""
import numpy as np

from dolfin import *
from turtleFSI.problems import *


# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet normals n('+') within interior boundaries (dS). For 3D mesh the value should be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    E_s_val_artery = 1E6                                                 # artery Young modulus (elasticity) [Pa]
    E_s_val_vein = 1E6                                                   # vein Young modulus (elasticity) [Pa]
    nu_s_val = 0.45                                                      # Poisson ratio (compressibility)
    mu_s_val_artery = E_s_val_artery/(2*(1+nu_s_val))                    # artery Shear modulus
    mu_s_val_vein = E_s_val_vein/(2*(1+nu_s_val))                        # vein Shear modulus
    lambda_s_val_artery = nu_s_val*2.*mu_s_val_artery/(1. - 2.*nu_s_val) # artery Solid 1rst Lamé coef. [Pa]
    lambda_s_val_vein = nu_s_val*2.*mu_s_val_vein/(1. - 2.*nu_s_val)     # vein Solid 1rst Lamé coef. [Pa]


    default_variables.update(dict( 
        T=3,                                              # Simulation end time (3 cardiac cycles)
        dt=0.0001,                                        # Time step size
        theta=0.501,                                      # Theta scheme (implicit/explicit time stepping)
        atol=1e-7,                                        # Absolute tolerance in the Newton solver
        rtol=1e-7,                                        # Relative tolerance in the Newton solver
        inlet_id1=3,                                      # inlet 1 id (PA)
        inlet_id2=2,                                      # inlet 2 id (DA)
        outlet_id1=4,                                     # outlet id (V)
        rigid_id=[11,1011],                               # "rigid wall" id for the fluid and mesh problem
        fsi_id=[22,1022],                                 # fsi interface
        outlet_s_id=44,                                   # solid outlet id (to be defined with sphere if you want to set different BC for the solid outlet)
        outer_id=[33,1033],                               # outer wall surface
        ds_s_id=[33,1033],                                # ID of solid external boundary(for Robin BC)
        Um1=0.629741588640293,                            # inlet 1 flow velocity [m/s](not needed for pulsatile)
        Um2=0.060856464239494,                            # inlet 2 flow velocity [m/s](not needed for pulsatile)
        p_val= 1333.0,                                    # inner pressure for initialisation [Pa] (not needed for pulsatile)
        vel_t_ramp= 0.2,                                  # time for velocity ramp 
        p_t_ramp_start = 0.05,                            # pressure ramp start time
        p_t_ramp_end = 0.2,                               # pressure ramp end time
        rho_f=1.025E3,                                    # Fluid density [kg/m3]
        mu_f=3.5E-3,                                      # Fluid dynamic viscosity [Pa.s]
        rho_s=[1.0E3,1.0E3],                              # Solid density [kg/m3]
        mu_s=[mu_s_val_artery,mu_s_val_vein],             # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,                                    # Solid Poisson ratio [-]
        lambda_s=[lambda_s_val_artery,lambda_s_val_vein], # Solid 1rst Lamé coef. [Pa]
        material_model="MooneyRivlin",                    # Material model
        gravity=None,                                     # Gravitational force [m/s**2]
	    robin_bc = True,                                  # Robin BC
        k_s = 1E5,                                        # elastic response necesary for RobinBC
        c_s = 1E1,                                        # viscoelastic response necesary for RobinBC 
        dx_f_id=1,                                        # ID of marker in the fluid domain
        dx_s_id=[2,1002],                                 # ID of marker in the solid domain
	    solid_properties = [{"dx_s_id":2, "material_model":"MooneyRivlin", "rho_s":1.0E3, "mu_s":mu_s_val_artery, "lambda_s":lambda_s_val_artery, "C01":0.03e6,"C10":0.0,"C11":2.2e6},
                            {"dx_s_id":1002, "material_model":"MooneyRivlin", "rho_s":1.0E3, "mu_s":mu_s_val_vein, "lambda_s":lambda_s_val_vein, "C01":0.003e6,"C10":0.0,"C11":0.538e6}],
        extrapolation="laplace",            # laplace, elastic, biharmonic, no-extrapolation
        extrapolation_sub_type="constant",  # constant, small_constant, volume, volume_change, bc1, bc2
        recompute=30,
        recompute_tstep=10,                 # Number of time steps before recompute Jacobian. 
        save_step=1,                        # Save frequency of files for visualisation
        folder="avf_results",                       # Folder where the results will be stored
        checkpoint_step=50,                 # checkpoint frequency
        compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)
   ))
    
    return default_variables


def get_mesh_domain_and_boundaries(mesh_path, fsi_id, rigid_id, outer_id, **namespace):
    #Import mesh file
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    # Define the region outside the sphere and assign with rigid id
    sph_x = 0.33642
    sph_y = 0.0873934
    sph_z = 0.0369964
    sph_rad = 0.02
    print("The coordinates of the sphere are [{}, {}, {}] of radius:{}".format(sph_x, sph_y, sph_z, sph_rad))

    i = 0
    for submesh_facet in facets(mesh):
        idx_facet = boundaries.array()[i]
        if idx_facet == fsi_id[0]:
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id[0] # changed "fsi" idx to "rigid wall" idx
        if idx_facet == fsi_id[1]:
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id[1]  # changed "fsi" idx to "rigid wall" idx
        if idx_facet == outer_id[0]:
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id[0]  # changed "outer" idx to "rigid wall" idx
        if idx_facet == outer_id[1]:
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id[1]  # changed "outer" idx to "rigid wall" idx
        i += 1


    return mesh, domains, boundaries


# Define velocity inlet 1 (PA) parabolic profile
class VelIn1Para(UserExpression):
    def __init__(self, t, dt, vel_t_ramp, n, dsi, mesh, interp_PA, **kwargs):
        self.t = t
        self.dt = dt
        self.t_ramp = vel_t_ramp
        self.interp_PA = interp_PA
        self.number = int(self.t/self.dt)
        self.n = n # normal direction
        self.dsi = dsi # surface integral element
        self.d = mesh.geometry().dim()
        self.x = SpatialCoordinate(mesh)
        # Compute area of boundary tesselation by integrating 1.0 over all facets
        self.A = assemble(Constant(1.0, name="one")*self.dsi)
        # Compute barycenter by integrating x components over all facets
        self.c = [assemble(self.x[i]*self.dsi) / self.A for i in range(self.d)]
        # Compute radius by taking max radius of boundary points
        self.r = np.sqrt(self.A / np.pi)
        super().__init__(**kwargs)


    def update(self, t):
        self.t = t
        if self.number +1 < len(self.interp_PA):
                 self.number = int(self.t/self.dt)


    def eval(self, value, x):
        # Define the parabola
        r2 = (x[0]-self.c[0])**2 + (x[1]-self.c[1])**2 + (x[2]-self.c[2])**2  # radius**2
        fact_r = 1 - (r2/self.r**2)

        #Define the velocity ramp
        if (self.t < self.t_ramp) and (self.t_ramp > 0.0):
            fact = self.interp_PA[self.number] * (-0.5*np.cos((pi/(self.t_ramp))*(self.t)) + 0.5)   # Velocity initialisation with sigmoid
            value[0] = -self.n[0] *fact_r * fact
            value[1] = -self.n[1] *fact_r * fact
            value[2] = -self.n[2] *fact_r * fact
        else:
            value[0] = -self.n[0] * (self.interp_PA[self.number]) *fact_r  # *self.t  # x-values
            value[1] = -self.n[1] * (self.interp_PA[self.number]) *fact_r  # *self.t. # y-values
            value[2] = -self.n[2] * (self.interp_PA[self.number]) *fact_r  # *self.t. # z-values


    def value_shape(self):
        return (3,)

    
# Define velocity inlet 2 (DA) parabolic profile
class VelIn2Para(UserExpression):
    def __init__(self, t, dt, vel_t_ramp, n, dsi, mesh, interp_DA, **kwargs):
        self.t = t
        self.dt = dt
        self.t_ramp = vel_t_ramp
        self.interp_DA = interp_DA
        self.number = int(self.t/self.dt)
        self.n = n # normal direction
        self.dsi = dsi # surface integral element
        self.d = mesh.geometry().dim()
        self.x = SpatialCoordinate(mesh)
        # Compute area of boundary tesselation by integrating 1.0 over all facets
        self.A = assemble(Constant(1.0, name="one")*self.dsi)
        # Compute barycenter by integrating x components over all facets
        self.c = [assemble(self.x[i]*self.dsi) / self.A for i in range(self.d)]
        # Compute radius by taking max radius of boundary points
        self.r = np.sqrt(self.A / np.pi)
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        if self.number +1 < len(self.interp_DA):
                 self.number = int(self.t/self.dt)  
        
    def eval(self, value, x):
        # Define the parabola
        r2 = (x[0]-self.c[0])**2 + (x[1]-self.c[1])**2 + (x[2]-self.c[2])**2  # radius**2
        fact_r = 1 - (r2/self.r**2)

        #Define the velocity ramp
        if self.t < self.t_ramp and (self.t_ramp > 0.0):
            fact = self.interp_DA[self.number]* (-0.5*np.cos((pi/(self.t_ramp))*(self.t)) + 0.5)   # Velocity initialisation with sigmoid
            value[0] = -self.n[0] *fact_r * fact
            value[1] = -self.n[1] *fact_r * fact
            value[2] = -self.n[2] *fact_r * fact
        else:
            value[0] = -self.n[0] * (self.interp_DA[self.number]) *fact_r  # *self.t  # x-values
            value[1] = -self.n[1] * (self.interp_DA[self.number]) *fact_r  # *self.t. # y-values
            value[2] = -self.n[2] * (self.interp_DA[self.number]) *fact_r  # *self.t. # z-values

    def value_shape(self):
        return (3,)
    

# Define the pressure profile
class InnerP(UserExpression):
    def __init__(self, t, dt, p_t_ramp_start, p_t_ramp_end, interp_P, **kwargs):
        self.t = t
        self.dt = dt
        self.interp_P = interp_P
        self.number = int(self.t/self.dt)
        self.p_t_ramp_start = p_t_ramp_start
        self.p_t_ramp_end = p_t_ramp_end     
        super().__init__(**kwargs)


    def update(self, t):
        self.t = t
        if self.number +1 < len(self.interp_P):
                 self.number = int(self.t/self.dt)  


    def eval(self, value, x):
        if self.t < self.p_t_ramp_start:
            value[0] = 0.0
        elif self.t < self.p_t_ramp_end:
            value[0] = self.interp_P[self.number]*(-0.5*np.cos((pi/(self.p_t_ramp_end - self.p_t_ramp_start))*(self.t - self.p_t_ramp_start)) + 0.5) # Pressure initialisation with sigmoid
        else:
            value[0] = self.interp_P[self.number]


    def value_shape(self):
        return ()


# Define boundary conditions
def create_bcs(dvp_, DVP, mesh, boundaries, domains, T, dt, mu_f, fsi_id, outlet_id1, inlet_id1, inlet_id2, outlet_s_id, rigid_id, psi, F_solid_linear, Um1, Um2, vel_t_ramp, p_t_ramp_start, p_t_ramp_end, p_deg,  v_deg, **namespace):

    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    # Fluid velocity BCs
    dsi1 = ds(inlet_id1, domain=mesh, subdomain_data=boundaries)
    dsi2 = ds(inlet_id2, domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni1 = np.array([assemble(n[i]*dsi1) for i in range(ndim)])
    ni2 = np.array([assemble(n[i]*dsi2) for i in range(ndim)])
    n_len1 = np.sqrt(sum([ni1[i]**2 for i in range(ndim)])) 
    n_len2 = np.sqrt(sum([ni2[i]**2 for i in range(ndim)]))  
    normal1 = ni1/n_len1
    normal2 = ni2/n_len2

    patient_data = np.loadtxt("hpg03_3d_BC_p_v_DEF_3c.csv", skiprows=1, delimiter=",", usecols=(0,1,2))
    v_PA = patient_data[:,0]
    v_DA = patient_data[:,1]
    PV = patient_data[:,2]
    
    len_v = len(v_PA)
    t_v = np.arange(len(v_PA))
    num_t = int(T/dt) #30.000 timesteps = 3s (T) / 0.0001s (dt)
    tnew = np.linspace (0, len_v, num=num_t) 

    interp_DA = np.array(np.interp(tnew,t_v, v_DA)) 
    interp_PA = np.array(np.interp(tnew,t_v, v_PA))
    interp_P = np.array(np.interp(tnew,t_v, PV)) #pressure interpolation (velocity and pressure waveforms must be syncronized)
    
    # Parabolic profile
    u_inflow_exp1 = VelIn1Para(t=0.0, dt=dt, vel_t_ramp=vel_t_ramp, n=normal1, dsi=dsi1, mesh=mesh, interp_PA=interp_PA, degree=v_deg)     
    u_inlet1 = DirichletBC(DVP.sub(1), u_inflow_exp1, boundaries, inlet_id1)   # Impose the pulsatile parabolic inlet velocity at the inlet1
    u_inflow_exp2 = VelIn2Para(t=0.0, dt=dt, vel_t_ramp=vel_t_ramp, n=normal2, dsi=dsi2, mesh=mesh, interp_DA=interp_DA, degree=v_deg)     
    u_inlet2 = DirichletBC(DVP.sub(1), u_inflow_exp2, boundaries, inlet_id2)   # Impose the parabolic inlet velocity at the inlet2
    u_inlet_s1 = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, rigid_id[0]) #Impose velocity 0 for first rigid solid region
    u_inlet_s2 = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, rigid_id[1]) #Impose velocity 0 for second rigid solid region

    # Solid Displacement BCs
    d_inlet1 = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_id1)
    d_inlet2 = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_id2)
    d_inlet_s1 = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, rigid_id[0]) #Impose displacement 0 for first rigid solid region
    d_inlet_s2 = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, rigid_id[1]) #Impose displacement 0 for second rigid solid region

    # Assign InnerP on the reference domain (FSI interface)
    p_out_bc_val = InnerP(t=0.0, dt=dt, interp_P=interp_P, p_t_ramp_start=p_t_ramp_start, p_t_ramp_end=p_t_ramp_end, degree=p_deg)
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    F_solid_linear += p_out_bc_val * inner(n('+'), psi('+'))*dSS(fsi_id[0]) + p_out_bc_val * inner(n('+'), psi('+'))*dSS(fsi_id[1])

    # Assemble Dirichlet boundary conditions
    bcs = [u_inlet1, u_inlet2, u_inlet_s1, u_inlet_s2, d_inlet1, d_inlet2, d_inlet_s1, d_inlet_s2]

    return dict(bcs=bcs, u_inflow_exp1=u_inflow_exp1, u_inflow_exp2=u_inflow_exp2, p_out_bc_val=p_out_bc_val,
                F_solid_linear=F_solid_linear)


def pre_solve(t, u_inflow_exp1, u_inflow_exp2, p_out_bc_val, **namespace):
    # Update the time variable used for the inlet boundary condition
    u_inflow_exp1.update(t)
    u_inflow_exp2.update(t)
    p_out_bc_val.update(t)
    return dict(u_inflow_exp1=u_inflow_exp1, u_inflow_exp2=u_inflow_exp2,p_out_bc_val=p_out_bc_val)