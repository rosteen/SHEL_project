from brokenaxes import brokenaxes
import juliet
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
import sqlite3 as sql

sns.set_style('ticks')
mpl.rcParams['figure.dpi']= 200


def plot_rvs(target, max_time_diff = 50):
    """
    Plot all RV data vs raw time and phase.

    Parameters
    ----------
    target: str
        Name of target, e.g. 'WASP-31'
    max_time_diff: int, optional
        Maximimum time difference between datapoints before splitting the axis
        with brokenaxes in the unphased plot
    """
    dataset = juliet.load(input_folder = f'juliet_fits/{target}/')
    results = dataset.fit(use_dynesty=True, dynamic=True)

    P = np.median(results.posteriors['posterior_samples']['P_p1'])
    t0 = np.median(results.posteriors['posterior_samples']['t0_p1'])

    rv_colors = ['orangered', 'blue', 'magenta', 'purple', 'green', 'goldenrod', 'red', 'olive', 'coral']
    rv_instruments = list(dataset.times_rv.keys())
    print(rv_instruments)

    # Get all RV times
    all_times = np.array([])
    for inst in rv_instruments:
        all_times = np.append(all_times, dataset.times_rv[inst])
    min_t = np.min(all_times)
    max_t = np.max(all_times)
    tstart = int(min_t - 5)
    tend = int(max_t + 5)

    # Create model times:
    tmodel = np.linspace(tstart, tend, 10000)

    # Evaluate model, get 68 credibility:
    model, error68_up, error68_down = results.rv.evaluate(rv_instruments[0], t = tmodel, return_err = True)
    # Same for 95 cred:
    model, error95_up, error95_down = results.rv.evaluate(rv_instruments[0], t = tmodel, return_err = True, alpha = 0.95)
    # Same, 99:
    model, error99_up, error99_down = results.rv.evaluate(rv_instruments[0], t = tmodel, return_err = True, alpha = 0.99)
    # Substract mu:
    mu_inst0 = np.median(results.posteriors['posterior_samples'][f'mu_{rv_instruments[0]}']),

    model =  model-mu_inst0
    error68_up, error68_down = error68_up-mu_inst0, error68_down-mu_inst0
    error95_up, error95_down = error95_up-mu_inst0, error95_down-mu_inst0
    error99_up, error99_down = error99_up-mu_inst0, error99_down-mu_inst0

    # Get sensible x axis splits for broken axis
    x_vals = [0,]
    sorted_times = np.sort(all_times-tstart)
    for i in range(len(sorted_times)-1):
        if sorted_times[i+1] - sorted_times[i] > max_time_diff:
            x_vals.append(int(sorted_times[i] + 5))
            x_vals.append(int(sorted_times[i+1] - 5))

    # Append final upper value
    x_vals.append(int(sorted_times[-1] + 5))

    x_lims = []
    for i in range(int(len(x_vals)/2)):
        x_lims.append([x_vals[i*2], x_vals[i*2+1]])

    fig = plt.figure(figsize=(17,5))
    bax = brokenaxes(xlims=(x_lims), hspace=.05)

    # Now plot the data points
    for i in range(len(rv_instruments)):

        instrument = rv_instruments[i]
        datapoint_color = rv_colors[i]
        print(datapoint_color)
        t, rv, rverr = (dataset.times_rv[instrument],
                        dataset.data_rv[instrument],
                        dataset.errors_rv[instrument])

        bax.errorbar(t-tstart, rv - np.median(results.posteriors['posterior_samples']['mu_'+instrument]),
                     rverr, fmt='o', ms=8, mec=datapoint_color,
                     ecolor=datapoint_color, mfc='white', zorder=3, label = instrument)

    bax.plot(tmodel-tstart,model, color = 'cornflowerblue', zorder=2)
    bax.fill_between(tmodel-tstart, error68_down, error68_up, color = 'cornflowerblue', alpha = 0.4, zorder=1)
    bax.fill_between(tmodel-tstart, error95_down, error95_up, color = 'cornflowerblue', alpha = 0.2, zorder=1)
    bax.fill_between(tmodel-tstart, error99_down, error99_up, color = 'cornflowerblue', alpha = 0.1, zorder=1)
    bax.tick_params('x', labelsize=10)
    bax.tick_params('y', labelsize=10)
    bax.set_xlabel('Time - '+str(tstart)+' (days)', fontsize = 17, labelpad = 25)
    bax.set_ylabel('Radial-velocity (m/s)', fontsize = 17)
    bax.legend(fontsize=17)
    plt.savefig(f'juliet_fits/{target}/{target}_rvs.pdf')

    # Now we plot the RV datapoints on a phased light curve
    fig = plt.figure(figsize=(16, 8))
    bax = brokenaxes(xlims=((-0.5, 0.5),), hspace=.05)

    # Now plot the data points
    for i in range(len(rv_instruments)):

        instrument = rv_instruments[i]
        datapoint_color = rv_colors[i]
        print(datapoint_color)
        t, rv, rverr = (dataset.times_rv[instrument],
                        dataset.data_rv[instrument],
                        dataset.errors_rv[instrument])
        phases = juliet.utils.get_phases(t, P, t0)

        bax.errorbar(phases, rv - np.median(results.posteriors['posterior_samples']['mu_'+instrument]),
                     rverr, fmt='o', ms=8, mec=datapoint_color,
                     ecolor=datapoint_color, mfc='white', zorder=3, label = instrument)

    model_phases = juliet.utils.get_phases(tmodel, P, t0)
    phase_sort = np.argsort(model_phases)
    model_phases = model_phases[phase_sort]

    bax.plot(model_phases, model[phase_sort], color = 'cornflowerblue', zorder=2)
    bax.fill_between(model_phases, error68_down[phase_sort], error68_up[phase_sort],
                     color = 'cornflowerblue', alpha = 0.4, zorder=1)
    bax.fill_between(model_phases, error95_down[phase_sort], error95_up[phase_sort],
                     color = 'cornflowerblue', alpha = 0.2, zorder=1)
    bax.fill_between(model_phases, error99_down[phase_sort], error99_up[phase_sort],
                     color = 'cornflowerblue', alpha = 0.1, zorder=1)
    bax.tick_params('x', labelsize=10)
    bax.tick_params('y', labelsize=10)
    bax.set_xlabel('Time - '+str(tstart)+' (days)', fontsize = 17, labelpad = 25)
    bax.set_ylabel('Radial-velocity (m/s)', fontsize = 17)
    bax.legend(fontsize=17, loc = 'upper right')
    #plt.tight_layout()
    plt.savefig(f'juliet_fits/{target}/{target}_rvs_phased.pdf')

def plot_tess(target, phased=True):

    instrument="TESS"

    # This should just load the existing fit
    dataset = juliet.load(input_folder = f'juliet_fits/{target}/')
    results = dataset.fit(use_dynesty=True, dynamic=True)

    period = np.median(results.posteriors['posterior_samples']['P_p1'])
    t0 = np.median(results.posteriors['posterior_samples']['t0_p1'])

    tess_times = dataset.times_lc['TESS']
    tdiffs = tess_times[1:]-tess_times[:-1]

    split_times = np.where(tdiffs>10)[0]
    split_inds = np.append([0,], split_times)
    split_inds = np.append(split_inds, [-1])
    split_inds

    fig = plt.figure(tight_layout=True)

    # Divide figure using gridspec:
    gs = GridSpec(4, split_inds.shape[0]-1, figure = fig)

    if phased:
        t,f,ferr = dataset.times_lc['TESS'], dataset.data_lc['TESS'], dataset.errors_lc['TESS']

        model, components = results.lc.evaluate('TESS', return_components = True)
        gp = model - components['transit']

        phases = juliet.utils.get_phases(t, period, t0) * period * 24

        plt.errorbar(phases, f - gp , ferr, fmt = '.', elinewidth=1,ms=2, zorder=1, alpha = 0.5)

        idx = np.argsort(phases)

        plt.plot(phases[idx], components['transit'][idx],zorder=2)

        plt.xlim(-3,3)
        plt.ylim(f.min()-0.005, 1.01)
        plt.ylabel('Relative flux')
        plt.xlabel('Time from mid-transit (hours)')
        plt.savefig(f'juliet_fits/{target}/tess_phased_lc_{target}.png')
    else:
        t,f,ferr = dataset.times_lc['TESS'], dataset.data_lc['TESS'], dataset.errors_lc['TESS']

        for i in range(len(split_inds)-1):
            ax = fig.add_subplot(gs[i,:])
            
            model, components = results.lc.evaluate('TESS', return_components = True)
            gp = model - components['transit']

            t_plot = t[split_inds[i]+1:split_inds[i+1]] - 2400000.5
            fgp_plot = (f - gp)[split_inds[i]+1:split_inds[i+1]]
            ferr_plot = ferr[split_inds[i]+1:split_inds[i+1]]
            
            ax.errorbar(t_plot, fgp_plot, ferr_plot, fmt = '.', elinewidth=1,ms=2, zorder=1, alpha = 0.5)
            
            ax.plot(t_plot, components['transit'][split_inds[i]+1:split_inds[i+1]],zorder=2)
            
            ax.set_xlim(t[split_inds[i]+1] - 2400000.5, t[split_inds[i+1]] - 2400000.5)
            ax.set_ylim(f.min()-0.005, 1.01)
            #plt.text(-2.5,0.985,'Sector '+sector.split('TESS')[-1], fontsize=13)
            ax.set_ylabel('Relative flux')

        ax.set_xlabel('Tess Time (days)')
        plt.savefig(f'juliet_fits/{target}/tess_lcs_{target}.png')

def plot_non_tess_lcs(target):
    pass

def plot_priors_posteriors(parameter):
    """
    Retrieve the priors and posteriors for all targets for the parameter
    we're interested in and create a scatter plot.
    """
    conn = sql.connect('shel_database.sqlite')
    cur = conn.cursor()
    data = {}
    for field in ("name", "prior", "prior_err", "posterior", 
                  "posterior_err_upper", "posterior_err_lower"):
        res = cur.execute(f"select {field} from system_parameters s join targets " 
                          f"t on s.target_id = t.id where parameter='{parameter}'").fetchall()
        data[field] = np.array(res).flatten()

    data['prior_err'][np.where(data['prior_err'] < 0)] = 0

    fig = plt.figure(figsize=(5,5))
    plt.errorbar(data['prior'], data['posterior'], fmt='o', xerr=data['prior_err'],
                 yerr=[data['posterior_err_lower'], data['posterior_err_upper']])

    # Add a line showing where perfect agreement would be
    if parameter == 't0_p1':
        plt.axline([2.454e6, 2.454e6], slope=1, alpha = 0.5, color = 'gray')
    else:
        plt.axline([0,0], slope=1, alpha = 0.5, color = 'gray')

    plt.savefig(f'plots/{parameter}_comparison_plot.png')

    cur.close()
    conn.close()
