import juliet
import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as sql

class SHEL_Fitter():
    def __init__(self, target_name):
        self.target = target_name

        # Connect to database
        self.conn = sql.connect("shel_database.sqlite")
        self.cur = self.conn.cursor()

        # Get target id
        stmt = f"select id from targets where name = '{target_name}'"
        self.target_id = self.cur.execute(stmt).fetchone()[0]

    def close_db(self):
        self.cur.close()
        self.conn.close()

    def get_light_curve_data(self, TESS_only=False):
        times, fluxes, fluxes_error = {},{},{}

        stmt = f"select * from light_curves where target_id = {self.target_id}"
        if TESS_only:
            stmt += " and instrument=1"

        res = self.cur.execute(stmt).fetchall()
        res = np.array(res)
        t = res[:, 3]
        f = res[:, 4]
        f_err = res[:, 5]

        times['TESS'], fluxes['TESS'], fluxes_error['TESS'] = t, f, f_err
        return times, fluxes, fluxes_error

    def get_rv_data(self):
        times_rv, data_rv, errors_rv = {}, {}, {}

    def fit(self, period, t0, a, b, p, ecc="Fixed", oot=False, debug=False):
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
        p: float
            Planet-to-star radius ratio (Rp/Rs).
        ecc: str, float
            Eccentricity of planet orbit, defaults to "Fixed" at 0.

        oot: bool
            Flag to fit only out-of-transit TESS data, to get systematics quickly.
        """
        priors = {}
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

        hyperps = [[3.735433, 0.1],
                   [2400000.5 + 54558.68096, 0.1],
                   [8.968, 0.62],
                   [0.4, 0.04],
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
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        # Get all instruments used for RV observations of this target, grouped by
        # reference so we can treat the same instrument separately for different refs.
        stmt = ("select reference_id, name from radial_velocities "
                "a join instruments b on a.instrument = b.id where "
                f"a.target_id = {self.target_id} group by reference_id")

        rv_insts = self.cur.execute(stmt).fetchall()

        # Concat ref_id with instrument name for
        rv_inst_names = [f"{x[1]}-{x[0]}" for x in rv_insts]

        params = ["K_p1",]
        dists = ["uniform",]
        hyperps = [[-100,100],]

        for instrument in rv_inst_names:
            name = instrument[0]
            inst_params = [f'mu_{name}',
                           f'sigma_w_{name}']

            # Distributions:
            inst_dists = ['uniform', 'loguniform']

            # Hyperparameters
            inst_hyperps = [[-100,100], [1e-3, 100]]

            params += inst_params
            dists += inst_dists
            hyperps += inst_hyperps

            # Populate the priors dictionary:
            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp


