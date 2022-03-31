import juliet
import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as sql

class SHEL_Fitter():
    def __init__(self, target_name, debug=False):
        self.target = target_name
        self.debug = debug
        self.priors = {}

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
                f"a.target_id = {self.target_id} group by reference_id")

        rv_insts = self.cur.execute(stmt).fetchall()

        # Concat ref_id with instrument name for
        rv_inst_names = [f"{x[1]}-{x[0]}" for x in rv_insts]
        return rv_inst_names

    def get_light_curve_data(self, TESS_only=False):
        times, fluxes, fluxes_error = {},{},{}

        stmt = f"select * from light_curves where target_id = {self.target_id}"
        if TESS_only:
            stmt += " and instrument=1"

        res = self.cur.execute(stmt).fetchall()
        res = np.array(res)
        if self.debug:
            print(res[0,:])
        t = res[:, 3]
        f = res[:, 4]
        f_err = res[:, 5]

        times['TESS'], fluxes['TESS'], fluxes_error['TESS'] = t, f, f_err
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

            times_rv[rv_inst_name] = t 
            data_rv[rv_inst_name] = rv 
            errors_rv[rv_inst_name] = rv_err

        return times_rv, data_rv, errors_rv

    def fit(self, period, t0, a, b, period_err=0.1, t0_err=0.1, a_err=1, b_err=0.1,
            ecc="Fixed", oot=False, debug=False, TESS_only=False):
        """
        Sets up prior distributions and runs the juliet fit. Currently assumes
        single-planet.

        Parameters
        ----------
        target: str
            String target name as in database
        period: float
            Period of planet, in days
        t0: float
            Time of transit center in BJD-TDB
        a: float
            Scaled semi-major axis of the orbit (a/R*).
        b: float
            Impact parameter of the orbit.
        ecc: str, float
            Eccentricity of planet orbit, defaults to "Fixed" at 0.

        oot: bool
            Flag to fit only out-of-transit TESS data, to get systematics quickly.
        """
        # Name of the parameters to be fit. We always at least want TESS photometry
        params = ['P_p1',
                  't0_p1',
                  'a_p1',
                  'b_p1',
                  'p_p1',
                  'q1_TESS',
                  'q2_TESS',
                  'ecc_p1',
                  'omega_p1',
                  'rho',
                  'mdilution_TESS',
                  'mflux_TESS',
                  'sigma_w_TESS']

        # Distribution for each of the parameters:
        dists = ['normal','normal','normal','normal','uniform','uniform',
                 'uniform','fixed','fixed','loguniform', 'fixed', 'normal',
                 'loguniform']

        hyperps = [[period, period_err],
                   [t0, t0_err],
                   [a, a_err],
                   [b, b_err],
                   [0, 0.3],
                   [0., 1.],
                   [0., 1.],
                   0.0,
                   90.,
                   [100., 10000.],
                   1.0,
                   [0.,0.1],
                   [0.1, 1000.]]

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
            self.priors[param]['distribution'], self.priors[param]['hyperparameters'] = dist, hyperp

        # Light curve data
        times, fluxes, fluxes_error = self.get_light_curve_data(TESS_only)

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

        # And now let's fit it! We default to Dynesty since we generally have >20 parameters
        self.results = self.dataset.fit(n_live_points = 20+len(self.priors)**2,
                                        sampler="dynamic_dynesty")
