import numpy as np
from few.trajectory.ode import KerrEccEqFlux_nex, KerrEccEqFlux
from few.trajectory.ode.flux import _pdot_PN,_edot_PN
from few.utils.mappings.kerrecceq import (_kerrecceq_flux_forward_map_nex, DELTAPMIN_REGIONC, DELTAPMAX, _uwyz_of_apex_kernel,_UWYZ_of_apex_kernel,
ALPHA_FLUX,BETA_FLUX,_uwyz_of_apex_kernel_nex,_kerrecceq_flux_forward_map,apex_of_uwyz_nex)
from few.utils.geodesic import get_separatrix, get_fundamental_frequencies


rhs_nex_pex = KerrEccEqFlux_nex(flux_output_convention="pex")
rhs_few_pex = KerrEccEqFlux()

def _PN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2) ** 1.5
    Edot = 32.0 / 5.0 * p ** (-5) * oneme2 * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    Ldot = 32.0 / 5.0 * p ** (-7 / 2) * oneme2 * (1 + 7.0 / 8.0 * e**2)
    return Edot, Ldot


#Function to evaluate fluxes on a range of p values for both models
def compare(ee,aa):

    m1 = 1e6  
    m2 = 1e2
    x=1
    delta_pmax = 1

    rhs_nex_pex.add_fixed_parameters(m1, m2, aa)
    rhs_few_pex.add_fixed_parameters(m1, m2, aa)

    pLSO = get_separatrix(aa,ee,x)
    p = np.linspace(pLSO + DELTAPMIN_REGIONC , pLSO + delta_pmax,10000)
    
    pdots1 = []
    pdots2 = []
    edots1 = []
    edots2 = []
    for pp in p:
        
    
        pdot1, edot1, xIdot1, Omega_phi1, Omega_theta1, Omega_r1 = rhs_nex_pex([pp, ee, x])
        pdot2, edot2, xIdot2, Omega_phi2, Omega_theta2, Omega_r2 = rhs_few_pex([pp, ee, x])
    
        pdots1.append(pdot1)
        pdots2.append(pdot2)
        edots1.append(edot1)
        edots2.append(edot2)

    return np.asarray(pdots1), np.asarray(pdots2), np.asarray(edots1), np.asarray(edots2), p, pLSO



ELQ = KerrEccEqFlux(flux_output_convention = "ELQ")
pex = KerrEccEqFlux(flux_output_convention = "pex")

ELQ_nex = KerrEccEqFlux_nex(flux_output_convention = "ELQ")
pex_nex = KerrEccEqFlux_nex(flux_output_convention = "pex")


def kerrecceqA(a,p,e,x):  # valid on p < plso +9.001
    pisco = get_separatrix(a,np.zeros_like(a),x)
    pLSO = get_separatrix(a,e,x)
    u,w,y,z = _uwyz_of_apex_kernel(a,p,e,x,pLSO)

    pn_term_p = 1#_pdot_PN(p,e,pisco,pLSO)
    pn_term_e = 1#_edot_PN(p,e,pisco,pLSO)
    
    
    Edot_interp = ELQ.Edot_interp_A
    Ldot_interp = ELQ.Ldot_interp_A
    pdot_interp = pex.pdot_interp_A
    edot_interp = pex.edot_interp_A

    return Edot_interp(u,w,z), Ldot_interp(u,w,z), -pdot_interp(u,w,z)*pn_term_p, -edot_interp(u,w,z)*pn_term_e,[u,w,y,z],pLSO


def nex(a,p,e,x):
    
    pisco = get_separatrix(a,np.zeros_like(a),x)
    pLSO = get_separatrix(a,e,x)
    u,w,y,z = _uwyz_of_apex_kernel_nex(a,p,e,x,pLSO)
    
    pn_term_p = 1#_pdot_PN(p,e,pisco,pLSO)
    pn_term_e = 1#_edot_PN(p,e,pisco,pLSO)
    
    Edot_interps = ELQ_nex.Edot_interp_A
    Ldot_interps = ELQ_nex.Ldot_interp_A
    pdot_interps = pex_nex.pdot_interp_A
    edot_interps = pex_nex.edot_interp_A

    return Edot_interps(u,w,z), Ldot_interps(u,w,z), -pdot_interps(u,w,z)*pn_term_p, -edot_interps(u,w,z)*pn_term_e,[u,w,y,z]   


def drE(r,a):
    num = r**2 - 3*a**2 + 8*a *np.sqrt(r) - 6*r
    den = 2*r**(7/4) * (r**(3/2)-3*np.sqrt(r)+2*a)**(3/2)
    return num/den

def nex_pedot(a,p,e,x): #Returns the actual rhs used to make traj
    psep = get_separatrix(a,e,x)
    pisco = get_separatrix(a,np.zeros_like(a),x)
    u,w,y,z = _uwyz_of_apex_kernel_nex(a,p,e,x,psep)

    pn_term_p = _pdot_PN(p,e,pisco,psep)
    pn_term_e = _edot_PN(p,e,pisco,psep)

    #turn = drE(p,a)

    pdot_interps = pex_nex.pdot_interp_A
    edot_interps = pex_nex.edot_interp_A

    return -pdot_interps(u,w,z)*pn_term_p , -edot_interps(u,w,z)*pn_term_e 
    
def few_pedot(a,p,e,x): #Returns the actual rhs used to make traj
    psep = get_separatrix(a,e,x)
    pisco = get_separatrix(a,np.zeros_like(a),x)
    u,w,y,z = _uwyz_of_apex_kernel(a,p,e,x,psep,ALPHA_FLUX,BETA_FLUX)

    pn_term_p = _pdot_PN(p,e,pisco,psep)
    pn_term_e = _edot_PN(p,e,pisco,psep)

    pdot_interps = pex.pdot_interp_A
    edot_interps = pex.edot_interp_A

    return -pdot_interps(u,w,z)*pn_term_p , -edot_interps(u,w,z)*pn_term_e 

def nex_ELdot(a,p,e,x): #Returns the actual rhs used to make traj
    psep = get_separatrix(a,e,x)
    pisco = get_separatrix(a,np.zeros_like(a),x)
    u,w,y,z = _uwyz_of_apex_kernel_nex(a,p,e,x,psep)

    EdotPN, LdotPN = _PN_alt(p, e)

    Edot_interps = ELQ_nex.Edot_interp_A
    Ldot_interps = ELQ_nex.Ldot_interp_A

    return -Edot_interps(u,w,z)*EdotPN , -Ldot_interps(u,w,z)*LdotPN

def nex_ELdot_uwz(u,w,z):
    u,w,y,z  = np.asarray(u),np.asarray(w),np.asarray(1),np.asarray(z)
    a,p,e,x = apex_of_uwyz_nex(u,w,y,z)
    return nex_ELdot(a,p,e,x)

def few_ELdot(a,p,e,x): #Returns the actual rhs used to make traj
    psep = get_separatrix(a,e,x)
    pisco = get_separatrix(a,np.zeros_like(a),x)
    u,w,y,z = _uwyz_of_apex_kernel(a,p,e,x,psep,ALPHA_FLUX,BETA_FLUX)

    EdotPN, LdotPN = _PN_alt(p, e)

    Edot_interps = ELQ.Edot_interp_A
    Ldot_interps = ELQ.Ldot_interp_A

    return -Edot_interps(u,w,z)*EdotPN , -Ldot_interps(u,w,z)*LdotPN
    