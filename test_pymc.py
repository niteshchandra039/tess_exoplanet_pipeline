import pymc as pm
import exoplanet as xo
import numpy as np
import celerite2.pymc as celerite2_pm
from celerite2.pymc import terms as c2terms
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
time_arr = np.linspace(120.0, 147.0, 200)
period_true = 6.2678
t0_true = 122.34
phase = ((time_arr - t0_true) / period_true) % 1.0
phase[phase > 0.5] -= 1.0
in_transit = np.abs(phase) < (0.1 / period_true)
flux_arr = np.ones_like(time_arr)
flux_arr[in_transit] -= 0.005
flux_arr += np.random.normal(0, 1e-4, len(time_arr))
flux_err_arr = np.ones_like(time_arr) * 1e-4

print("Building model...")
with pm.Model() as model:
    rho_star = pm.TruncatedNormal("rho_star", mu=1.4, sigma=0.1, lower=0.0, initval=1.4)
    period = pm.Normal("period", mu=6.2678, sigma=0.03, initval=6.2678)
    t0 = pm.Normal("t0", mu=122.34, sigma=0.05, initval=122.34)
    log_rp = pm.Uniform("log_rp", lower=np.log(0.001), upper=np.log(0.5), initval=np.log(0.07))
    rp_r_star = pm.Deterministic("rp_r_star", pm.math.exp(log_rp))
    b = pm.Uniform("b", lower=0.0, upper=1.0 + rp_r_star, initval=0.1)
    
    q1 = pm.Uniform("q1", lower=0.0, upper=1.0, initval=0.3)
    q2 = pm.Uniform("q2", lower=0.0, upper=1.0, initval=0.3)
    sqrt_q1 = pm.math.sqrt(q1)
    u1 = pm.Deterministic("u1", 2.0 * sqrt_q1 * q2)
    u2 = pm.Deterministic("u2", sqrt_q1 * (1.0 - 2.0 * q2))
    
    mean_flux = pm.Normal("mean_flux", mu=0.0, sigma=0.01, shape=1, initval=[0.0])
    log_jitter = pm.Normal("log_jitter", mu=-6.0, sigma=2.0, initval=-6.0)
    
    orbit = xo.orbits.KeplerianOrbit(
        period=period,
        t0=t0,
        b=b,
        rho_star=rho_star,
        ror=rp_r_star
    )
    
    star_lc = xo.LimbDarkLightCurve(u1, u2)
    light_curves = star_lc.get_light_curve(orbit=orbit, r=rp_r_star, t=time_arr)
    transit_model = pm.Deterministic("transit_model", pm.math.sum(light_curves, axis=-1) + mean_flux[0] + 1.0)
    
    log_sigma_gp = pm.Normal("log_sigma_gp", mu=-3.0, sigma=2.0, initval=-3.0)
    log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(5.0), sigma=2.0, initval=np.log(5.0))
    sigma_gp = pm.Deterministic("sigma_gp", pm.math.exp(log_sigma_gp))
    rho_gp = pm.Deterministic("rho_gp", pm.math.exp(log_rho_gp))
    
    term = c2terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=1.0 / np.sqrt(2.0))
    gp = celerite2_pm.GaussianProcess(term, mean=transit_model)
    gp.compute(time_arr, diag=flux_err_arr**2 + pm.math.exp(2.0 * log_jitter), quiet=True)
    gp.marginal("obs", observed=flux_arr)
    
    print("Testing pm.find_MAP...")
    map_soln = pm.find_MAP()
    print("MAP found successfully!")
    
    print("Testing pm.sample with init='adapt_diag'...")
    trace = pm.sample(
        draws=10,
        tune=10,
        chains=1,
        init="adapt_diag",
        initvals=map_soln,
        return_inferencedata=True,
        progressbar=False
    )
    print("MCMC Sample completed successfully!")
