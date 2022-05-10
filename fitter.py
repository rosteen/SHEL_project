import juliet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sqlite3 as sql

class SHEL_Fitter():
    def __init__(self, target_name, debug=False):
        self.target = target_name
        self.debug = debug
        self.priors = {}
        self.tess_systematics = None

        # Connect to database
        self.conn = sql.connect("shel_database.sqlite")
        self.cur = self.conn.cursor()

        # Get target id
        stmt = f"select id from targets where name = '{target_name}'"
        self.target_id = self.cur.execute(stmt).fetchone()[0]

    def close_db(self):
        self.cur.close()
        self.conn.close()

    def _get_rv_inst_names(self):
        # Get all instruments used for RV observations of this target, grouped by
        # reference so we can treat the same instrument separately for different refs.
        stmt = ("select reference_id, name from radial_velocities "
                "a join instruments b on a.instrument = b.id where "
                f"a.target_id = {self.target_id} group by reference_id, name")

        rv_insts = self.cur.execute(stmt).fetchall()

        # Concat ref_id with instrument name for
        rv_inst_names = [f"{x[1]}-{x[0]}" for x in rv_insts]
        return rv_inst_names

    def _get_lc_inst_names(self):
        # Get all instruments used for LC observations of this target, grouped by
        # reference so we can treat the same instrument separately for different refs.
        stmt = ("select reference_id, name from light_curves "
                "a join instruments b on a.instrument = b.id where "
                f"a.target_id = {self.target_id} group by reference_id, name")

        lc_insts = self.cur.execute(stmt).fetchall()

        # Concat ref_id with instrument name for
        lc_inst_names = [f"{x[1]}-{x[0]}".replace("-None", "") for x in lc_insts]
        return lc_inst_names

    def get_light_curve_data(self, TESS_only=False):
        """
        Get the light curve data for the target, optionally returning
        only the TESS data.
        """
        times, fluxes, fluxes_error = {},{},{}

        if TESS_only:
            lc_inst_names = ["TESS"]
        else:
            lc_inst_names = self._get_lc_inst_names()

        for lc_inst_name in lc_inst_names:
            if len(lc_inst_name.split("-")) == 1:
                instrument = lc_inst_name
                stmt = ("select bjd, flux, flux_err from light_curves a join "
                        "instruments b on a.instrument = b.id where "
                        f"a.target_id = {self.target_id} and b.name = '{instrument}'")
            else:
                instrument, ref_id = lc_inst_name.split("-")
                stmt = ("select bjd, flux, flux_err from light_curves a join "
                        "instruments b on a.instrument = b.id where "
                        f"a.target_id = {self.target_id} and b.name = '{instrument}' "
                        f"and reference_id = {ref_id}")

            if self.debug:
                print(stmt)

            res = self.cur.execute(stmt).fetchall()
            res = np.array(res)

            t = res[:,0]
            flux = res[:,1]
            flux_error = res[:,2]

            times[lc_inst_name] = t
            fluxes[lc_inst_name] = flux
            fluxes_error[lc_inst_name] = flux_error

        return times, fluxes, fluxes_error

    def get_rv_data(self, rv_inst_names=None):
        times_rv, data_rv, errors_rv = {}, {}, {}

        if rv_inst_names is None:
            rv_inst_names = self._get_rv_inst_names()

        for rv_inst_name in rv_inst_names:
            instrument, ref_id = rv_inst_name.split("-")
            stmt = ("select bjd, rv, rv_err from radial_velocities a join "
                    "instruments b on a.instrument = b.id where "
                    f"a.target_id = {self.target_id} and b.name = '{instrument}' "
                    f"and reference_id = {ref_id}")
            if self.debug:
                print(stmt)
            res = self.cur.execute(stmt).fetchall()
            res = np.array(res)

            t = res[:,0]
            rv = res[:,1]
            rv_err = res[:,2]

            rv = rv - np.mean(rv)

            times_rv[rv_inst_name] = t
            data_rv[rv_inst_name] = rv
            errors_rv[rv_inst_name] = rv_err

        return times_rv, data_rv, errors_rv

    def fit_tess_systematics(self, period, center, duration=None):
        """
        Fit the out of transit TESS data to get the TESS systematics, so we
        can fix them in the planetary parameter fitting.

        Parameters
        ----------
        period: float
            Planet period in days
        duration: float
            Tranit duration in hours
        center:
            Center of transit, in BJD-UTC
        """
        # Set our phase boundary to be considered in/out of transit
        if duration is None:
            print("No transit duration provided, defaulting to using +/-0.05 phase oot")
            self.oot_phase_limit = 0.05
        else:
            # We'll include +/- one transit duration as in-transit
            self.oot_phase_limit = duration/(period*24)

        # Get the TESS data
        t, f, ferr = self.get_light_curve_data(TESS_only=True)
        t = t['TESS']
        f = f['TESS']
        ferr = ferr['TESS']

        # Get phases and sort data
        phases = juliet.get_phases(t, period, center)
        idx_oot = np.where(np.abs(phases)>self.oot_phase_limit)[0]

        times, fluxes, fluxes_error = {},{},{}

        sort_times = np.argsort(t[idx_oot])

        times['TESS'] = t[idx_oot][sort_times]
        fluxes['TESS'] = f[idx_oot][sort_times]
        fluxes_error['TESS'] = ferr[idx_oot][sort_times]

        # Set the priors:
        params = ['mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS', 'GP_sigma_TESS',
                   'GP_rho_TESS']
        dists = ['fixed', 'normal', 'loguniform', 'loguniform', 'loguniform']
        hyperps = [1., [0.,0.1], [1e-6, 1e6], [1e-6, 1e6], [1e-3,1e3]]

        priors = {}
        for param, dist, hyperp in zip(params, dists, hyperps):
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        # Perform the juliet fit. Load dataset first (note the GP regressor will be the times):
        detrend_dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes,
                                      yerr_lc = fluxes_error, GP_regressors_lc = times,
                                      out_folder = f'juliet_fits/{self.target}/detrend_TESS')
        # Fit:
        results = detrend_dataset.fit()
        ts = {}
        ts['mflux_TESS'] = np.median(results.posteriors['posterior_samples']['mflux_TESS'])
        ts['sigma_w_TESS'] = np.median(results.posteriors['posterior_samples']['sigma_w_TESS'])
        ts['GP_sigma_TESS'] = np.median(results.posteriors['posterior_samples']['GP_sigma_TESS'])
        ts['GP_rho_TESS'] = np.median(results.posteriors['posterior_samples']['GP_rho_TESS'])
        self.tess_systematics_results = results
        self.tess_systematics = ts

    def initialize_fit(self, period, t0, b, a=None, period_err=0.1, t0_err=0.1, a_err=1,
                       b_err=0.1, ecc="Fixed", fit_oot=False, debug=False, TESS_only=False,
                       duration=None):
        """
        Sets up prior distributions and runs the juliet fit. Currently assumes
        single-planet. If self.tess_systematics is populated, the fit will use those
        fixed values and fit only the in-transit data for TESS.

        Parameters
        ----------
        target: str
            String target name as in database
        period: float
            Period of planet in days
        t0: float
            Time of transit center in BJD-TDB
        b: float
            Impact parameter of the orbit.
        a: float
            Scaled semi-major axis of the orbit (a/R*). If none, Rho value from
            the database will be used as prior instead.
        duration: float
            Transit duration in hours, optional.
        ecc: str, float
            Eccentricity of planet orbit, defaults to "Fixed" at 0.
        fit_oot: bool
            Flag to fit TESS out of transit data separately to set systematics
        TESS_only: bool
            Flag to drop non-TESS photometry
        """

        # Name of the parameters to be fit. We always at least want TESS photometry
        params = ['P_p1',
                  't0_p1',
                  'b_p1',
                  'p_p1',
                  'q1_TESS',
                  'q2_TESS',
                  'ecc_p1',
                  'omega_p1',
                  'rho',
                  'mdilution_TESS',
                  'mflux_TESS',
                  'sigma_w_TESS',
                  'GP_rho_TESS',
                  'GP_sigma_TESS']

        # Distribution for each of the parameters:
        dists = ['normal','normal', 'normal','uniform','uniform',
                 'uniform','uniform','fixed','loguniform', 'fixed']

        hyperps = [[period, period_err],
                   [t0, t0_err],
                   [b, b_err],
                   [0, 0.3],
                   [0., 1.],
                   [0., 1.],
                   [0., 0.5],
                   90.,
                   [100., 10000.],
                   1.0,]

        # Add the appropriate distributions and values for TESS systematics
        if self.tess_systematics is None:
            dists += ['normal', 'loguniform', 'loguniform', 'loguniform']
            hyperps += [[0.,0.1], [0.1, 1000.], [1e-6, 1e6], [1e-3, 1e3]]
        else:
            dists += ['fixed', 'fixed', 'fixed', 'fixed']
            hyperps += [self.tess_systematics['mflux_TESS'],
                        self.tess_systematics['sigma_w_TESS'],
                        self.tess_systematics['GP_rho_TESS'],
                        self.tess_systematics['GP_sigma_TESS']]

        # Set either a or rho
        if a is not None:
            params += ['a_p1',]
            dists += ['normal']
            hyperps += [[a, a_err],]
        else:
            # Retrieve Rho parameter from database
            stmt = ("select rho, rho_err from stellar_parameters where "
                    f"target_id = {self.target_id}")
            rho, rho_err = self.cur.execute(stmt).fetchone()
            params += ['rho',]
            dists += ['truncatednormal',]
            hyperps += [[rho, rho_err, 0, 20000],]


        # Populate the priors dictionary:
        for param, dist, hyperp in zip(params, dists, hyperps):
            self.priors[param] = {}
            self.priors[param]['distribution'], self.priors[param]['hyperparameters'] = dist, hyperp


        # Concat ref_id with instrument name for rv data keys
        rv_inst_names = self._get_rv_inst_names()

        params = ["K_p1",]
        dists = ["uniform",]
        hyperps = [[-100,100],]

        for instrument in rv_inst_names:
            inst_params = [f'mu_{instrument}',
                           f'sigma_w_{instrument}']

            # Distributions:
            inst_dists = ['uniform', 'loguniform']

            # Hyperparameters
            inst_hyperps = [[-120,120], [1e-3, 100]]

            params += inst_params
            dists += inst_dists
            hyperps += inst_hyperps

        # Populate the priors dictionary:
        for param, dist, hyperp in zip(params, dists, hyperps):
            self.priors[param] = {}
            self.priors[param]['distribution'] = dist
            self.priors[param]['hyperparameters'] = hyperp

        # Light curve data
        times, fluxes, fluxes_error = self.get_light_curve_data(TESS_only)

        # Intialize priors for the non-TESS light curve instruments
        for inst in times.keys():
            if inst == "TESS":
                continue
            params = [f"mdilution_{inst}", f"mflux_{inst}", f"sigma_w_{inst}",
                      f"q1_{inst}", f"q2_{inst}", f"GP_sigma_{inst}",
                      f"GP_rho_{inst}"]
            hyperps = [1, [0.,0.1], [0.1, 10000.], [0, 1.0], [0, 1.0],
                       [1e-6, 1e6], [1e-3,1e3]]
            dists = ['fixed', 'normal', 'loguniform', 'uniform', 'uniform',
                     'loguniform', 'loguniform']
            for param, dist, hyperp in zip(params, dists, hyperps):
                self.priors[param] = {}
                self.priors[param]['distribution'] = dist
                self.priors[param]['hyperparameters'] = hyperp


        if not fit_oot:
            if duration is None:
                print("No transit duration provided, defaulting to using +/-0.05 phase oot")
                self.oot_phase_limit = 0.05
            else:
                # We'll include +/- one transit duration as in-transit
                self.oot_phase_limit = duration/(period*24)

            # Get phases and sort data
            for inst in times:
                phases = juliet.get_phases(times[inst], period, t0)
                idx_oot = np.where(np.abs(phases)<=self.oot_phase_limit)[0]
                sort_times = np.argsort(times[inst][idx_oot])

                times[inst] = times[inst][idx_oot][sort_times]
                fluxes[inst] = fluxes[inst][idx_oot][sort_times]
                fluxes_error[inst] = fluxes_error[inst][idx_oot][sort_times]

        out_folder = f"juliet_fits/{self.target}"
        kwargs = {"priors": self.priors, "t_lc": times, "y_lc": fluxes,
                  "yerr_lc": fluxes_error, "out_folder": out_folder}

        # Get RV data
        if not TESS_only:
            times_rv, data_rv, errors_rv = self.get_rv_data()
            kwargs["t_rv"] = times_rv
            kwargs["y_rv"] = data_rv
            kwargs["yerr_rv"] = errors_rv

        # Load the dataset
        self.dataset = juliet.load(**kwargs)

    def run_fit(self):
        # And now let's fit it! We default to Dynesty since we generally have >20 parameters
        self.results = self.dataset.fit(n_live_points = 20+len(self.priors)**2,
                                        sampler="dynamic_dynesty")

    def plot_results(self, type, instrument=None):
        """
        Plot some or all (either RV or LC) of the fitted result and input data.

        Parameters
        ----------
        type: str
            'lc' or 'rv'
        instrument_or_type: str
            The string identifier of a single instrument, including reference ID
            if relevant, e.g. 'TESS' or 'FLWO-13', if you want to plot only data
            and fit for a single instrument.
        """
        # Intialize plots, one subplot for data and fit, one for residuals
        fig = plt.figure(figsize=(14,4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2,2])

        # Get the fitted period and center of transit for phasing
        P = np.median(results.posteriors['posterior_samples']['P_p1'])
        t0 = np.median(results.posteriors['posterior_samples']['t0_p1'])

        if type == "lc":
            pass
        elif type == "rv":
            pass
