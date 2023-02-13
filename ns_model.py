import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from scipy.optimize import root
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d, UnivariateSpline


class NeutronStar:

    def __init__(self, df1, df2, df3, df4, df5, ns_inputs, verbose=True):
        rho_center, proton_frac = ns_inputs.values()
        self.EoS_f = self.get_EOS(df1)
        self.in_data = [df1, df2, df3, df4, df5]
        (self.profile, self.density, self.enc_mass) = self.solve_TOV(rho_center)
        (self.dens_, self.latt_) = self.calc_internal(df2, proton_frac)
        (self.mass_, self.moi_) = self.get_mass_MoI()
        print(self.profile) if verbose else None
        print(self.moi_) if verbose else None
        self.J_name = None

    @staticmethod
    def get_EOS(df):
        """
        Interpolation of numerical data to generate equation of state
        :param df: data for equation of state
        :return: functional forms for pressure(density) and density(pressure)
        """

        def EoS_pressure(rho):
            # equation of state in outer crust, rho < 1e9 kg/m3
            kr = 2241876752.766294  # continuity
            knr = kr * 1e9 ** (-1 / 3)
            knd = knr * 1e7 ** (2 / 3)

            conds = [(7.9e3 < rho) & (rho <= 1e7),
                     (1e7 < rho) & (rho <= 1e9)]

            vals = [(lambda x: knd * x),    # ideal gas law
                    (lambda x: knr * x ** (5 / 3))]     # non-relativistic degenerate electrons

            return np.piecewise(rho, conds, vals)

        # get equation of state in Si units
        p_conversion = (u.MeV * u.fm ** -3).to(u.Pa)
        rho_conversion = (u.MeV * u.fm ** -3 / const.c ** 2).to(u.kg / u.m ** 3).value
        dens_ = df.density * rho_conversion
        press_ = df.pressure * p_conversion

        # combine EoS of core/inner-crust to outer-crust
        ds_cr = np.geomspace(7.9e3, dens_.min() - 10, 50)
        pr_cr = EoS_pressure(ds_cr)
        press_ = np.sort(pd.concat([pd.Series(pr_cr), press_]).values)
        dens_ = np.sort(pd.concat([pd.Series(ds_cr), dens_]).values)

        get_pressure = UnivariateSpline(dens_, press_, s=1, k=1)  # pressure(density)
        get_density = UnivariateSpline(press_, dens_, s=1, k=1)  # density(pressure)

        return get_pressure, get_density

    def solve_TOV(self, rho_center):
        """
        Tolman-Oppenheimer-Volkoff equation solver
        :param rho_center: central density / 2.7e17kg/m3
        :return: radial profile for a neutron star, density(r) and enclosed_mass(r)
        """

        def dm_dr(r, rho):
            return 4 * np.pi * r ** 2 * rho

        def dP_dr(r, G, c, m, rho, P):
            term1 = -G * rho * m / r ** 2
            term2 = 1 + P / (rho * c ** 2)
            term3 = 1 + (4 * np.pi * P * r ** 3) / (m * c ** 2)
            term4 = (1 - (2 * G * m) / (c ** 2 * r)) ** -1
            return term1 * term2 * term3 * term4

        def dSdr(r, S, G, c):
            m, P = S
            rho = self.EoS_f[1](P)
            return [dm_dr(r, rho),
                    dP_dr(r, G, c, m, rho, P)]

        def init_mass(r, rho):  # nucleation sphere
            return (4 / 3) * np.pi * r ** 3 * rho

        G_ = const.G.value
        c_ = const.c.value
        r0 = 1e-6   # nucleation sphere radius
        rho_cent = 2.7e17 * rho_center          # central density
        pre_cent = self.EoS_f[0](rho_cent)      # corresponding central pressure
        r_ = np.geomspace(r0, 14e3, 100000)     # integration range [m]
        ans = solve_ivp(dSdr, t_span=(np.min(r_), np.max(r_)), t_eval=r_, y0=[init_mass(r0, rho_cent), pre_cent],
                        args=(G_, c_), method='LSODA', rtol=1e-5, atol=1e-8)
        sols = ans.y

        density = self.EoS_f[1](sols[1])
        m_dist = sols[0][~np.isnan(sols[0])] * u.kg.to(u.Msun)

        profile = {'surface': r_[density <= 7.9e3][0],  # radius of neutron star
                   'crust': r_[density <= 4.3e14][0],   # distance from center to inner/outer crust boundary
                   'core': r_[density <= 2.63e17][0],   # radius of core
                   'mass': m_dist.max()}                # mass of neutron star

        def interp(y_data):     # generate interpolation for density(r) and enclosed_mass(r)
            x_data = r_[r_ < profile['surface']]
            y_data = y_data[r_ < profile['surface']]
            interp_ = interp1d(x_data, y_data, kind='cubic', fill_value=0, bounds_error=False)
            return interp_

        return profile, interp(density), interp(m_dist)

    def calc_internal(self, df, proton_frac):
        """
        Calculate the density distribution for each component in a neutron star.
        :param df: input data for equilibrium nucleus in neutron drip regime
        :param proton_frac: fraction of core that contains superconducting protons
        :return: density(r), atomic_number(r) and lattice_density(r) for each component in a neutron star
        """
        rho_ = df.density * (u.g / u.cm ** 3).to(u.kg / u.m ** 3)
        conv = (u.fm ** -3).to(u.m ** -3)
        n = df.k ** 3 / (1.5 * np.pi ** 2)      # bulk neutron number density
        nn = df.kn ** 3 / (1.5 * np.pi ** 2)    # neutron super-fluid number density
        nN = df.nN * 1e-6                       # lattice nuclei number density

        neutron_density = (1 - df.A * nN / n) * nn * const.m_n.value
        nuclei_density = (df.A - df.Z) * nN * const.m_n.value + df.Z * nN * const.m_p.value

        # interpolations
        ngas_ = UnivariateSpline(rho_, neutron_density * conv)
        lattice_ = UnivariateSpline(rho_, nuclei_density * conv)
        lim_ = root(lambda x: lattice_(x) - x, np.array([2.7e17])).x[0]  # find optimal radius of nuclear saturation

        def charged_density(rho):   # density(r) for all components in a neutron star that are charged
            conds = [rho <= 7.9e3,
                     (7.9e3 < rho) & (rho <= 4.3e14),
                     (4.3e14 < rho) & (rho <= lim_),
                     rho > lim_]
            vals = [(lambda x: x * 0),
                    (lambda x: x),
                    (lambda x: lattice_(x)),
                    (lambda x: x * proton_frac)]
            return np.piecewise(rho, conds, vals)

        def inner_core(r): return (self.density(r) < lim_) & (self.density(r) >= 4.3e14)
        density_frac = {'n_gas': lambda r: ngas_(self.density(r)) * inner_core(r),
                        'lattice': lambda r: lattice_(self.density(r)) * inner_core(r),
                        'charged': lambda r: charged_density(self.density(r)),
                        'n_core': lambda r: self.density(r) * (1 - proton_frac) * (self.density(r) > lim_)}

        # equilibrium nucleus data for outer crust, required for star-quake simulations
        outer_crust = np.array([[8.02e6, 26, 30, 1404.05], [2.71e8, 28, 34, 449.48], [1.33e9, 28, 36, 266.97],
                                [1.50e9, 28, 38, 259.26], [3.09e9, 36, 50, 222.66], [1.06e10, 34, 50, 146.56],
                                [2.79e10, 32, 50, 105.23], [6.07e10, 30, 50, 80.58], [8.46e10, 30, 52, 72.77],
                                [9.67e10, 46, 82, 80.77], [1.47e11, 44, 82, 69.81], [2.11e11, 42, 82, 61.71],
                                [2.89e11, 40, 82, 55.22], [3.97e11, 38, 82, 49.37], [4.27e11, 36, 82, 47.92]])

        rho_oc, Z_oc, N_oc, R_cell = outer_crust.T  # [density, proton_number, neutron_number, lattice_unit_cell]
        rho_x = pd.concat([pd.Series(rho_oc * (u.g / u.cm ** 3).to(u.kg / u.m ** 3)), rho_])

        z_y = pd.concat([pd.Series(Z_oc), df.Z])
        z_interp = UnivariateSpline(rho_x, z_y, s=1, k=1)

        nN_oc = (4 / 3 * np.pi * R_cell ** 3) ** -1
        nN_y = pd.concat([pd.Series(nN_oc * conv), nN * conv])
        nN_interp = UnivariateSpline(rho_x, nN_y, s=1, k=1)

        lattice_interps = {'Z': lambda x: z_interp(self.density(x)),    # atomic_number(radius)
                           'nN': lambda x: nN_interp(self.density(x))}  # nuclei_density(radius)

        return density_frac, lattice_interps

    def get_mass_MoI(self):
        """
        Calculate mass and moment of inertia of each component
        :return: mass, moment of inertia
        """

        def get_mass(low, high, density):
            M = 4 * np.pi * quad(lambda x: x ** 2 * density(x), low, high, limit=200)[0]
            return M * u.kg.to(u.Msun)

        def get_MoI(low, high, density):
            MoI = (8 / 3) * np.pi * quad(lambda x: x ** 4 * density(x), low, high, limit=200)[0]
            return MoI

        center = 1e-6
        surface, crust, core = self.profile['surface'], self.profile['crust'], self.profile['core']

        superfluid_mass = get_mass(core, crust, self.dens_['n_gas'])
        superfluid_moi = get_MoI(core, crust, self.dens_['n_gas'])

        charged_mass = get_mass(center, surface, self.dens_['charged'])
        charged_moi = get_MoI(center, surface, self.dens_['charged'])

        n_core_mass = get_mass(center, core, self.dens_['n_core'])
        n_core_moi = get_MoI(center, core, self.dens_['n_core'])

        total_mass = get_mass(center, surface, self.density)
        total_moi = get_MoI(center, surface, self.density)

        mass_ = {'n_gas': superfluid_mass, 'charged': charged_mass,
                 'n_core': n_core_mass, 'total': total_mass}
        moi_ = {'n_gas': superfluid_moi, 'charged': charged_moi,
                'n_core': n_core_moi, 'total': total_moi}

        return mass_, moi_

    def get_pulsar_data(self, J_name):
        """
        Get angular frequency, spin-down rate and glitch data for target pulsar
        :param J_name: Name of pulsar
        :return: Delta-Omega, Omega, Omega_dot
        """
        self.J_name = J_name
        df_glitch = self.in_data[2]
        df_pulsar = self.in_data[3]

        # rotational frequencies of pulsar + glitch data
        F0, F1 = df_pulsar[['F0', 'F1']][df_pulsar.JNAME == 'J' + J_name].values[0].astype(float)
        glitches = df_glitch[df_glitch.J_name == J_name]

        # find max glitch size
        dff, dff1 = glitches.iloc[glitches['dF/F'].astype(float).argmax()][['dF/F', 'dF1/F1']].astype(float).values
        dF, dF1 = dff * 1e-9 * F0, dff1 * 1e-3 * F1
        dO, dO1 = 2 * np.pi * dF, 2 * np.pi * dF1

        return dO, 2 * np.pi * F0, 2 * np.pi * F1

    def plot_ns(self, graph):
        if graph == 'eos':  # plot equation of state
            fig, ax = plt.subplots(figsize=(12, 8))
            dens_r = np.geomspace(7e3, 1e19, 1000)
            ax.plot(dens_r, self.EoS_f[0](dens_r), 'r-')
            for rho_i in [7.9e3, 1e7, 1e9, 1.2e10, 4.3e14, 2.7e17]:
                ax.axvline(rho_i, c='black', linestyle='--')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('density [kg m^-3]')
            ax.set_ylabel('pressure [pa]')
            ax.set_title(r'EoS, P = P($\rho$)')
            ax.set_ylim([1e12, 1e36])

        if graph == 'profile':  # plot density distributions
            fig, ax = plt.subplots(figsize=(16, 8))
            r = np.linspace(1e-6, 12e3, 1000)
            ax.plot(r / 1000, self.density(r), 'k--', label='total density')
            ax.plot(r / 1000, self.dens_['n_gas'](r), 'r', label='neutron gas')
            ax.plot(r / 1000, self.dens_['n_core'](r), c='orange', label='core neutrons')
            ax.plot(r / 1000, self.dens_['charged'](r), 'g', label='charged component')
            ax.set_ylabel('density [kg/m^3]')
            ax.set_xlabel('radius [km]')
            ax.axhline(0, c='k')
            ax.legend()

    def plot_sim(self, data, graph, comp='all', figsize=(16, 9), ev_=100, start_year=0):
        """
        Plot results of simulation
        :param data: simulation data
        :param graph: which graph to plot
        :param comp: which component
        :param figsize: plot size
        :param ev_: resolution of plots
        :param start_year: start of sim
        :return:
        """
        params_ = list(data.get('params')[-1].values())
        sg = data.get('sg')
        dt_ = params_[0][1]
        start_year = Time(sg.get('t0', start_year), format='mjd').jyear

        def plot_(axis, charge, core, neutron, labels=('charged', 'core', 'neutrons'),
                  cols=('blue', 'red', 'orange')):

            t_, g_t_ = [data.get(key) for key in ['t', 'g_t']]
            t_r = Time(t_[::ev_] * u.s.to(u.year) + start_year, format='jyear').datetime
            axis.plot(t_r, charge[::ev_], label=labels[0], c=cols[0]) if comp in ['all', 'charge', 'reduced'] else None
            axis.plot(t_r, core[::ev_], label=labels[1], c=cols[1]) if comp in ['all', 'core'] else None
            axis.plot(t_r, neutron[::ev_], label=labels[2], c=cols[2]) if comp in ['all', 'neutron'] else None
            m_, r_ = self.profile["mass"], self.profile["surface"] / 1000
            axis.set_title(f'# of glitches: {len(g_t_)}, mass: {m_:.2f} M_sun, radius: {r_: .2f} km')
            axis.set_xlabel(f'Time, dt={dt_}s')
            axis.legend()
            plt.tight_layout()

        sim_key = {'omega': ['Oc', 'Ol', 'On', r'$\Omega$ [rad/s]'],    # plot
                   'd_omega': ['dOc', 'dOl', 'dOn', r'$\dot{\Omega}$ [rad/s^2]'],
                   'MoI': ['Ic', 'Il', 'In', 'Moment of Inertia [kg m^2]'],
                   'eps': ['eps_c', 'eps_l', 'eps_n', 'oblateness']}

        if graph in sim_key.keys():
            fig, ax = plt.subplots(figsize=figsize)
            y_label = sim_key[graph].pop(-1)
            xc, xl, xn = [data.get(key) for key in sim_key.get(graph)]
            plot_(ax, xc, xl, xn)
            ax.set_ylabel(y_label)

        elif graph == 'gw_strain':
            fig, ax = plt.subplots(figsize=figsize)
            hc, hl, hn = self.get_gw_strain(data)
            plot_(ax, hc, hl, hn)
            ax.set_ylabel('GW strain')
            ax.set_yscale('log')

        elif graph == 'stress':
            fig, ax = plt.subplots(figsize=figsize)
            t_arr, eps_c, eps0_c = [data.get(key) for key in ['t', 'eps_c', 'eps0_c']]
            t_range = Time(t_arr[::ev_] * u.s.to(u.year) + start_year, format='jyear').datetime
            ax.plot(t_range, eps0_c[::ev_] - eps_c[::ev_], 'b')
            ax.set_ylabel('stress/shear')
            ax.set_xlabel(f'Time, dt={dt_}s')

        elif graph == 'glitch':
            dO_max, omega_i, omega_dot_i = self.get_pulsar_data(self.J_name)
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
            g_t, g_dO, g_ddO = [data.get(key) for key in ['g_t', 'g_dO', 'g_ddO']]
            t_range = Time(g_t * u.s.to(u.year) + start_year, format='jyear').datetime
            ax1.plot(t_range, g_dO, 'b.')
            ax2.plot(t_range, abs(g_ddO), 'r.')
            ax1.set_title(fr'# of glitches: {len(g_t)}, $\Omega$: {omega_i:.3f}, $\Omega_d$: {omega_dot_i:.3e}')
            ax1.set_ylabel(r'$\Delta\Omega$')
            ax2.set_ylabel(r'$\Delta\dot{\Omega}$')
            ax2.set_xlabel(f'Time')
            ax1.set_yscale('log')
            ax2.set_yscale('log')

        elif graph == 'radiation':
            fig, ax = plt.subplots(figsize=figsize)
            keys = ['t', 'Oc', 'eps_c', 'Ic', 'params']
            t_arr, Omega_c, eps_c, Ic, params_ = [data.get(key) for key in keys]
            k, n = list(params_[-1].values())[3], list(params_[-1].values())[4]
            G, c = const.G.value, const.c.value

            dO_em_c = - k * Omega_c ** n
            dO_gw_c = -32 / 5 * G / c ** 5 * Ic * (3 / 2 * eps_c) ** 2 * Omega_c ** 5

            t_range = Time(t_arr[::ev_] * u.s.to(u.year) + start_year, format='jyear').datetime
            ax.plot(t_range, np.abs(dO_em_c[::ev_]), c='blue', label='EM-braking')
            ax.plot(t_range, np.abs(dO_gw_c[::ev_]), c='red', label='GR waves')
            ax.set_ylabel(r'$-\dot{\Omega}$ [rad/s^2]')
            ax.set_xlabel(f'Time, dt={dt_}s')
            ax.set_yscale('log')
            ax.legend()

        elif graph == 'dOcrit':
            fig, ax = plt.subplots(figsize=figsize)
            keys = ['t', 'Oc', 'On', 'dO_crit', 'Ic', 'Il', 'In', 'params']
            t_arr, Omega_c, Omega_n, dO_crit, Ic, Il, Ing, params_ = [data.get(key) for key in keys]
            t_range = Time(t_arr[::ev_] * u.s.to(u.year) + start_year, format='jyear').datetime
            ax.plot(t_range, Omega_n[::ev_] - Omega_c[::ev_], 'b', label='sim')
            ax.plot(t_range, dO_crit[::ev_], 'r--', label='critical-sep')
            ax.set_title(fr'$\Delta\Omega$ = {np.mean(dO_crit):.7f}')
            ax.set_ylabel(r'$\Omega_n - \Omega_c$')
            ax.set_xlabel(f'Time, dt={dt_}s')
            ax.legend()

    def get_gw_strain(self, data):
        """
        Calculate gravitational wave strain emitted from spinning neutron star
        :param data: simulation data
        :return: GR strain for each component
        """
        # Omega, Moment_of_inertia, oblateness
        Oc, Ic, eps_c = [data.get(key) for key in ['Oc', 'Ic', 'eps_c']]
        Ol, Il, eps_l = [data.get(key) for key in ['Ol', 'Il', 'eps_l']]
        On, Ing, eps_n = [data.get(key) for key in ['On', 'In', 'eps_n']]
        d = float(self.in_data[4].DIST[self.in_data[4].JNAME == 'J' + self.J_name].values[0]) * u.kpc.to(u.m)

        def gw_strain(omega, I_, eps_, dist):   # gravitational quadrupole
            G = const.G.value
            c = const.c.value
            return G/c**4 * I_/dist * (3/2 * eps_) * (2 * omega) ** 2

        return gw_strain(Oc, Ic, eps_c, d), gw_strain(Ol, Il, eps_l, d), gw_strain(On, Ing, eps_n, d)

    def get_AB(self):
        """
        A: Gravitational potential energy
        B: Elastic energy stored in crust
        :return: A, B
        """
        G = const.G.value
        eps0 = const.eps0.value
        e = const.e.value

        R_surface = self.profile['surface']
        R_core = self.profile['core']
        R_center = 1e-6

        def w0_integrand(x): return self.enc_mass(x) * self.density(x) * x
        w0 = -G * 4 * np.pi * quad(w0_integrand, R_center, R_surface, limit=200)[0]
        A = abs(w0) / 5 * u.Msun.to(u.kg)

        def B_integrand(x): return self.latt_['Z'](x) ** 2 * self.latt_['nN'](x) ** (4 / 3) * x ** 2
        B = 2 * e ** 2 / eps0 * quad(B_integrand, R_core, R_surface, limit=200)[0]

        return A, B

    def get_residuals(self, data, toas, r_lim, plot=False, ax=None, res=False):
        """
        Weighted least squares for glitch simulation
        :param data: Simulation results
        :param toas: Pulsar timing data
        :param r_lim: residual cut-off
        :param plot: show plot
        :param ax: plot axis
        :param res: plot resolution
        :return: W-RSS
        """

        t_i = data['sg']['t0'] + data['t'] * u.s.to(u.day)
        sim_interp = interp1d(t_i, data['dOc'])
        toas_ = toas[2][abs(toas[3]) < r_lim]
        dOmega = toas[1][abs(toas[3]) < r_lim]*2*np.pi
        weights = toas[3][abs(toas[3]) < r_lim]
        t_max, t_min = np.max(t_i),  np.min(t_i)
        dOmega = dOmega[(toas_ > t_min) & (toas_ < t_max)]
        t_test = toas_[(toas_ > t_min) & (toas_ < t_max)]
        weights = weights[(toas_ > t_min) & (toas_ < t_max)]
        residuals = (dOmega-sim_interp(t_test))**2 if not res else ((dOmega-sim_interp(t_test))/weights)**2
        if ax:
            label = f'M={self.profile["mass"]:.2f} M_sun, r={self.profile["surface"]/1000:.2f} km'
            ax.plot(t_test, sim_interp(t_test), label=label, lw=1.5)
            ax.plot(t_test, dOmega, 'k.', markersize=5, mew=0.1)
        elif plot:
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 12))
            ax1.plot(t_test, sim_interp(t_test), 'b--')
            ax1.plot(t_test, dOmega, 'g.')
            ax1.set_ylabel('omega_dot')
            ax1.set_xlabel('mjd')
            ax1.set_ylim([-2.34e-9, -2.31e-9])
            ax2.plot(t_test, residuals)
            ax2.set_ylabel('residuals')
            ax2.set_xlabel('mjd')
            ax2.set_yscale('log')

        return np.sum(residuals)

    @staticmethod
    def sim_functions(params):
        """
        Functions required for simulation
        :param params: input parameters
        :return: required functions
        """
        A, B, Ic_0, Il_0, In_0, dO_max, omega_i, shear_, single_glitch, multi_glitch, inputs = params
        t_param, stress_amp, sq_param, nc_param = inputs.values()

        def get_vals(t):    # get simulation parameters for single glitch analysis
            t0, tg, d0, O, k_, n_, Tl_, Tn_, source = single_glitch.values()
            ind_ = 0 if t < tg else 1
            return k_[ind_], n_[ind_], Tl_[ind_], Tn_[ind_]

        def MoI_(I0, eps, oblateness_on=False):  # moment of inertia
            return I0 * (1 + eps*oblateness_on)

        def get_eps_0(omega, I0):   # reference oblateness
            return omega ** 2 * I0 / (4*A)

        def eps_(omega, I0, eps_0):     # actual oblateness
            term1 = omega ** 2 * I0 / (4 * (A + B))
            term2 = B * eps_0 / (A + B)
            return term1 + term2

        def get_eps_I(omega_c, omega_l, omega_n, ref_eps_0):    # oblateness and MOI for each component
            eps_ci = eps_(omega_c, Ic_0, ref_eps_0)
            eps_li = get_eps_0(omega_l, Il_0)
            eps_ni = get_eps_0(omega_n, In_0)
            Ic = MoI_(Ic_0, eps_ci)
            Il = MoI_(Il_0, eps_li)
            Ing = MoI_(In_0, eps_ni)
            return eps_ci, eps_li, eps_ni, Ic, Il, Ing

        def get_stress_crit():  # calculate critical stress for a given glitch size
            Ic = MoI_(Ic_0, get_eps_0(omega_i, Ic_0))
            term1 = A / B * shear_ * Ic / Ic_0 * 1 / omega_i
            term2 = Ic_0 ** 2 * omega_i ** 2 / (2 * A * Ic)
            return term1 * (1 + term2) * dO_max * stress_amp
        stress_crit = get_stress_crit()

        def stress_(omega, I0, eps_0):  # calculate stress
            return shear_ * (eps_0 - eps_(omega, I0, eps_0))

        def dO_star_quake(omega_c, ref_eps):    # calculate glitch jump from star-quake
            Ic = MoI_(Ic_0, eps_(omega_c, Ic_0, ref_eps))
            term1 = B/A * stress_crit / shear_ * Ic_0/Ic * omega_c
            term2 = Ic_0 ** 2 * omega_c ** 2 / (2 * A * Ic)
            return term1 / (1 + term2)

        def L_transfer_sq(Oc, Ol, On, dO):      # angular momentum transfer for star-quake
            N = (1-np.random.power(sq_param[0], 1)[0]) * sq_param[1]
            new_eps_0 = get_eps_0(Oc + dO, Ic_0)
            eps_ci, eps_li, eps_ni, Ic, Il, Ing = get_eps_I(Oc, Ol, On, new_eps_0)
            dO = (On - Oc) / (1 + Ic/Ing)
            return dO * N, -Ic / Ing * dO * N

        def L_transfer_ns(Oc, Ol, On, eps_0):    # angular momentum transfer for vortex un-pinning event
            N = np.random.uniform(low=nc_param[0], high=nc_param[1])
            eps_ci, eps_li, eps_ni, Ic, Il, Ing = get_eps_I(Oc, Ol, On, eps_0)
            dO = (On - Oc) / (1 + Ic / Ing)
            return dO * N, -Ic / Ing * dO * N

        def dLc(Ic, omega_c, omega_l, omega_n, eps_c, k, n, Tl, Tn, oblateness_on=False):
            # rate of change of angular momentum for charged component
            G, c = const.G.value, const.c.value
            term1 = - Ic / Tl * (omega_c - omega_l)
            term2 = -Ic/Tn * (omega_c - omega_n)
            term3 = -Ic * k * omega_c ** n
            term4 = -32 / 5 * G / c ** 5 * Ic ** 2 * (3 / 2 * eps_c) ** 2 * omega_c ** 5
            return term1 + term2 + term3 + term4*oblateness_on

        def dLl(Ic, Il, omega_c, omega_l, eps_l, Tl, oblateness_on=False):
            # rate of change of angular momentum for core
            G, c = const.G.value, const.c.value
            term1 = Ic / Tl * (omega_c - omega_l)
            term2 = -32 / 5 * G / c ** 5 * Il ** 2 * (3 / 2 * eps_l) ** 2 * omega_l ** 5
            return term1 + term2*oblateness_on

        def dLn(Ic, Ing, omega_c, omega_n, eps_n, Tn, oblateness_on=False):
            # rate of change of angular momentum for neutron super fluid
            G, c = const.G.value, const.c.value
            term1 = Ic / Tn * (omega_c - omega_n)
            term2 = -32 / 5 * G / c ** 5 * Ing ** 2 * (3 / 2 * eps_n) ** 2 * omega_n ** 5
            return term1 + term2*oblateness_on

        def dSdt(t, S, ref_eps_0, k, n, Tl, Tn, oblateness_on=False):  # rate of change of angular frequency at time t
            omega_c, omega_l, omega_n = S
            if oblateness_on:
                eps_ci, eps_li, eps_ni, Ic, Il, Ing = get_eps_I(omega_c, omega_l, omega_n, ref_eps_0)
            else:
                eps_ci, eps_li, eps_ni, Ic, Il, Ing = 0, 0, 0,  Ic_0, Il_0, In_0
            return [dLc(Ic_0, omega_c, omega_l, omega_n, eps_ci, k, n, Tl, Tn) / (3*Ic - 2*Ic_0),
                    dLl(Ic_0, Il_0, omega_c, omega_l, eps_li, Tl) / (3*Il - 2*Il_0),
                    dLn(Ic_0, In_0, omega_c, omega_n, eps_ni, Tn) / (3*Ing - 2*In_0)]

        def ddO_c(omega_c, omega_l, omega_n, eps_0_i, eps_0_f, k, n, Tl, Tn,
                  dOc=0, dOl=0, dOn=0):  # jump in spin-down rate
            ddO_i = dSdt(0, [omega_c, omega_l, omega_n], eps_0_i, k[0], n[0], Tl[0], Tn[0])
            dd0_f = dSdt(0, [omega_c + dOc, omega_l + dOl, omega_n + dOn],  eps_0_f, k[1], n[1], Tl[1], Tn[1])
            return dd0_f[0] - ddO_i[0]

        def event1(t, x, *args):    # star-quake trigger event
            return stress_(x[0], Ic_0, args[0]) - stress_crit

        def event2(t, x, *args):    # vortex un-pinning trigger event
            eps_ci, eps_li, eps_ni, Ic, Il, Ing = get_eps_I(x[0], x[1], x[2], args[0])
            dO_crit = (1 + Ic/Ing) * dO_max * nc_param[2]
            return x[2] - x[0] - dO_crit

        def event3(t, x, *args):    # trigger single glitch
            return single_glitch.get('t0', 0) + t * u.s.to(u.day) - single_glitch.get('tg', 0)

        def sim_(t, y0, ref_eps_0, k, n, Tl, Tn, glitch_on):    # neutron star time inter-glitch evolution simulator
            event1.terminal = glitch_on[0]
            event2.terminal = glitch_on[1]
            event3.terminal = glitch_on[2]
            sols = solve_ivp(dSdt, t_span=(t.min(), t.max()), t_eval=t, y0=y0, args=[ref_eps_0, k, n, Tl, Tn],
                             dense_output=True, method='LSODA', events=(event1, event2, event3),
                             rtol=1e-8, atol=1e-11)
            return sols

        def return_vals(args):      # collect all simulated data
            t_arr, Oc, Ol, On, eps_0, vals_, glitches = map(np.array, args)
            k_, n_, Tl_, Tn_ = np.array(vals_).T
            l_i = len(t_arr[t_arr <= (single_glitch['tg'] - single_glitch['t0']) * u.day.to(u.s)])
            l_f = len(t_arr[t_arr > (single_glitch['tg'] - single_glitch['t0']) * u.day.to(u.s)])
            k_sim = np.append(np.ones(l_i) * k_[0], np.ones(l_f) * k_[1])
            n_sim = np.append(np.ones(l_i) * n_[0], np.ones(l_f) * n_[1])
            Tl_sim = np.append(np.ones(l_i) * Tl_[0], np.ones(l_f) * Tl_[1])
            Tn_sim = np.append(np.ones(l_i) * Tn_[0], np.ones(l_f) * Tn_[1])
            Oc_d, Ol_d, On_d = dSdt(0, (Oc, Ol, On), eps_0, k_sim, n_sim, Tl_sim, Tn_sim)
            glitches = glitches if np.any(glitches) else np.array([[0, 0, 0]])
            sols = {'t': t_arr, 'Oc': Oc, 'Ol': Ol, 'On': On, 'dOc': Oc_d, 'dOl': Ol_d, 'dOn': On_d,
                    'g_t': glitches.T[0], 'g_dO': glitches.T[1], 'g_ddO': glitches.T[2],
                    'params': params, 'vals': vals_, 'sg': single_glitch}
            return sols

        return [get_eps_0, dO_star_quake, L_transfer_sq, L_transfer_ns,
                ddO_c, sim_, return_vals, get_vals]

    def run_sim(self, J_name, params, glitch_on=(True, True, True), max_glitch=1e6, steady=False,
                single_glitch={}, multi_glitch={}, verbose=True, rise_time=0):
        """
        run neutron star simulation
        :param J_name: pulsar name
        :param params: user-input parameters
        :param glitch_on: toggle on/off star-quake, vortex un-pinning & core glitch
        :param max_glitch: max number of glitches allowed
        :param steady: start simulation in steady state
        :param single_glitch: single glitch analysis
        :param multi_glitch: 
        :param verbose: display progress
        :param rise_time: rise time of glitch
        :return: simulation results
        """
        dO_max, omega_i, omega_dot_i = self.get_pulsar_data(J_name)
        In_0, Ic_0, Il_0, It_0 = self.moi_.values()
        t_lim, step_size = params['t_lim']
        A, B = self.get_AB()

        V_crust = 4 / 3 * np.pi * (self.profile['surface'] ** 3 - self.profile['core'] ** 3)
        shear_ = 2 * B / V_crust

        args = (A, B, Ic_0, Il_0, In_0, dO_max, omega_i, shear_, single_glitch, multi_glitch, params)
        get_eps_0, dO_star_quake, L_transfer_sq, L_transfer_ns, ddO_c, sim_, return_vals, get_vals = \
            self.sim_functions(args)

        def update_args(delta_omega_c, delta_omega_n, oc_arr, ol_arr, on_arr,  k_i, n_i, Tl_i, Tn_i,
                        delta_omega_l=0):   # update simulation parameters post glitch
            eps_0_new = get_eps_0(omega_c_arr[-1] + delta_omega_c, Ic_0)
            eps_0_arr[-1] = eps_0_new

            if single_glitch:
                k_, n_, Tl_, Tn_ = get_vals(single_glitch['t0'] + t_arr[-1] * u.s.to(u.day)+1)
            else:
                k_, n_, Tl_, Tn_ = multi_glitch.values()

            vals_.append([k_, n_, Tl_, Tn_])
            dd_omega = ddO_c(oc_arr[-1], ol_arr[-1], on_arr[-1], eps_0_arr[-2], eps_0_arr[-1], [k_i, k_], [n_i, n_],
                             [Tl_i, Tl_], [Tn_i, Tn_], dOc=delta_omega_c, dOl=delta_omega_l, dOn=delta_omega_n)

            oc_arr[-1] += delta_omega_c
            on_arr[-1] += delta_omega_n
            ol_arr[-1] += delta_omega_l

            t_arr[-1] += rise_time * u.d.to(u.s)
            print(delta_omega_c, dd_omega) if verbose else None
            glitches_arr.append([t_arr[-1], delta_omega_c, dd_omega])

        def steady_state(Oc):   # calculate steady-state separations
            if single_glitch:
                k_, n_, Tl_, Tn_ = get_vals(single_glitch['t0'] + t_arr[-1] * u.s.to(u.day))
            else:
                k_, n_, Tl_, Tn_ = multi_glitch.values()

            alpha = In_0**2/(Ic_0+Il_0+In_0) * Tn_/Ic_0 * k_ * n_ * Oc**(n_-1) + In_0/Ic_0
            beta = Il_0**2/(Ic_0+Il_0+In_0) * Tl_/Ic_0 * k_ * n_ * Oc**(n_-1) + Il_0/Ic_0
            dO_lc_steady = beta/(1+alpha+beta) * k_ * Tl_ * Oc ** n_
            dO_nc_steady = alpha/(1+alpha+beta) * k_ * Tn_ * Oc ** n_

            return dO_lc_steady, dO_nc_steady

        t_arr, omega_i = [0], single_glitch.get('O', omega_i)
        dOl_steady, dOn_steady = steady_state(omega_i) if steady else (0, 0)
        omega_c_arr, omega_l_arr, omega_n_arr = [omega_i], [omega_i+dOl_steady], [omega_i+dOn_steady]
        glitches_arr, eps_0_arr = [], [get_eps_0(omega_c_arr[-1], Ic_0)]

        sim_pars = get_vals(single_glitch.get('t0', 0)) if single_glitch else multi_glitch.values()
        vals_ = [sim_pars]

        while t_arr[-1] <= t_lim - step_size:

            omega_ci, omega_li, omega_ni = omega_c_arr[-1], omega_l_arr[-1], omega_n_arr[-1]
            eps_0i = eps_0_arr[-1]
            k_i, n_i, Tl_i, Tn_i = vals_[-1]

            t_limit = t_lim     # min(1e8 + t_arr[-1] + 1, t_lim)
            t_i = np.arange(t_arr[-1] + 1, t_limit, step_size)
            run_glitch = np.array(glitch_on) & (len(glitches_arr) < max_glitch)
            sol_i = sim_(t_i, [omega_ci, omega_li, omega_ni], eps_0i, k_i, n_i, Tl_i, Tn_i, run_glitch)
            Oc_i, Ol_i, On_i = sol_i.y

            t_arr.extend(list(sol_i.t))
            omega_c_arr.extend(list(Oc_i))
            omega_l_arr.extend(list(Ol_i))
            omega_n_arr.extend(list(On_i))
            eps_0_arr.extend((list(np.ones(len(sol_i.t)) * eps_0i)))

            if sol_i.t_events[0] and glitch_on[0]:   # star-quake
                dO_sq = dO_star_quake(omega_c_arr[-1], eps_0_arr[-1])
                dO_c, dO_ng = L_transfer_sq(omega_c_arr[-1], omega_l_arr[-1], omega_n_arr[-1], dO_sq)

                update_args(dO_sq + dO_c, dO_ng, omega_c_arr, omega_l_arr, omega_n_arr, k_i, n_i, Tl_i, Tn_i)
                print(t_arr[-1]/t_lim, 'star-quake') if verbose else None

            elif sol_i.t_events[1] and glitch_on[1]:     # vortex unpinning
                dO_c, dO_ng = L_transfer_ns(omega_c_arr[-1], omega_l_arr[-1], omega_n_arr[-1], eps_0_arr[-1])

                update_args(dO_c, dO_ng, omega_c_arr, omega_l_arr, omega_n_arr, k_i, n_i, Tl_i, Tn_i)
                print(t_arr[-1]/t_lim, 'vortex unpinning') if verbose else None

            elif sol_i.t_events[2] and glitch_on[2]:    # single glitch
                dO_c = single_glitch['dO']
                if single_glitch['source'] == 'core':
                    dO_l = -Ic_0/Il_0 * dO_c
                    update_args(dO_c, 0, omega_c_arr, omega_l_arr, omega_n_arr,  k_i, n_i, Tl_i, Tn_i,
                                delta_omega_l=dO_l)
                else:
                    dO_ng = -Ic_0/In_0 * dO_c
                    update_args(0, dO_ng, omega_c_arr, omega_l_arr, omega_n_arr, k_i, n_i, Tl_i, Tn_i)
                print(t_arr[-1] / t_lim, 'single glitch') if verbose else None
                glitch_on[2] = False

            else:
                if t_arr[-1] <= t_lim:
                    print(t_arr[-1]/t_lim, 'restart') if verbose else None
                    continue
                else:
                    break

        return return_vals((t_arr, omega_c_arr, omega_l_arr, omega_n_arr, eps_0_arr, vals_, glitches_arr))

