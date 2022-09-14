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
    '''
    Plot the TESS light curves, both phase-folded and non-folded
    '''

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
    gs = GridSpec(split_inds.shape[0]-1, 4, figure = fig)

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
    '''
    Plot the non-TESS light curves. Currently coded to phase-fold each instrument
    individually - I should probably just plot all the individual transit curves. 
    '''

    dataset = juliet.load(input_folder = f'juliet_fits/{target}/') 
    results = dataset.fit(use_dynesty=True, dynamic=True)

    period, t0 = (np.median(results.posteriors['posterior_samples']['P_p1']),
                  np.median(results.posteriors['posterior_samples']['t0_p1']))

    non_tess_insts = list(dataset.times_lc.keys())
    non_tess_insts.remove("TESS")

    n_insts = len(non_tess_insts)
    if n_insts == 0:
        print(f"\nNo non-TESS light curves for {target}!\n")
        return

    n_rows = np.ceil(n_insts/3)

    t, lc, lcerr = {}, {}, {}

    for instrument in non_tess_insts:
        # Extract lightcurve data:
        t[instrument] = dataset.times_lc[instrument]
        lc[instrument] = dataset.data_lc[instrument]
        lcerr[instrument] = dataset.errors_lc[instrument]

    fig = plt.figure(figsize=(3*np.min([n_insts, 3]),3*n_rows))

    # Divide figure using gridspec:
    gs = GridSpec(int(5*n_rows), int(np.min([n_insts, 3])), figure = fig)

    for i in range(n_insts):

        instrument = non_tess_insts[i]
        counter = i%3
        min_gs = int(np.floor(i/3)*5)

        print(f"{instrument}, {counter}, {min_gs}")

        t = dataset.times_lc[instrument]
        lc = dataset.data_lc[instrument]
        lcerr = dataset.errors_lc[instrument]

        ax = fig.add_subplot(gs[min_gs:min_gs+4, counter])
        ax_residuals = fig.add_subplot(gs[min_gs+4:min_gs+5, counter])
        
        # Extract jitters; add them to the errors:
        sigma_w = np.median(results.posteriors['posterior_samples']['sigma_w_'+instrument])
        lcerr = np.sqrt(lcerr**2 + (sigma_w*1e-6)**2)
        
        # Phases:
        phases = juliet.utils.get_phases(t, period, t0) * period * 24.
        
        # Evaluate model:
        model, upper, lower, components = results.lc.evaluate(instrument, return_components = True, return_err = True)
        model, upper95, lower95, components = results.lc.evaluate(instrument, return_components = True, return_err = True, \
                                                                  alpha = 0.95)
     
        gp = model - components['transit']
        transit_upper = upper - gp
        transit_lower = lower - gp
        transit_upper95 = upper95 - gp
        transit_lower95 = lower95 - gp
        
        # Evaluate the transit model only on the entire time-range:
        transit_times = np.linspace(t0 - (5./24.), t0 + (5./24.), 1000)
        transit_model = results.lc.evaluate(instrument, t = transit_times, evaluate_transit=True)
        
        transit_model = transit_model / np.max(transit_model)
        transit_phases = juliet.utils.get_phases(transit_times, period, t0) * period * 24.
        transit_idx = np.argsort(transit_phases)
        
        nbins = 15
        # Plot detrended data:
        idx = np.argsort(phases)
        ax.errorbar(phases, lc - gp, lcerr, fmt = '.', color = 'black', alpha = 0.1, ms=2, rasterized=True, zorder = 1)
        # Plot binned data:
        xbin, ybin, ybinerr = juliet.utils.bin_data(phases[idx], lc[idx]-gp[idx], nbins)
        ax.errorbar(xbin, ybin, ybinerr, fmt = 'o', mec = 'black', ecolor = 'black', mfc = 'white', \
                    elinewidth=1, rasterized=True, zorder = 5)
        
        # Plot transit models and errors:
        ax.plot(transit_phases[transit_idx], transit_model[transit_idx], color = 'blue', zorder = 6)
        
        # Residuals:
        ax_residuals.errorbar(phases, (lc-model)*1e6, lcerr*1e6, fmt = '.', color = 'black', alpha = 0.1, ms=2, rasterized=True)
        
        ax_residuals.plot([-5,5], [0., 0.], '--', color = 'grey', zorder = 1)
        xbin, ybin, ybinerr = juliet.utils.bin_data(phases[idx], (lc[idx]-model[idx])*1e6, nbins)
        ax_residuals.errorbar(xbin, ybin, ybinerr, fmt = 'o', mec = 'black', ecolor = 'black', mfc = 'white', \
                    elinewidth=1, rasterized=True, zorder = 5)
        
        # Details for top plots:
        ax.set_xlim(-3,3)
        ax.set_ylim(0.97,1.01)
        ax.set_xticklabels([])
        
        if counter == 0:
            
            ax.set_ylabel('Relative flux', fontsize = 12)
            ax_residuals.set_ylabel('O-C (ppm)', fontsize = 12)
            

        ax.text(0, 1, f"{instrument}", horizontalalignment='center', fontsize = 12)
        # Details for bottom plots:
        ax_residuals.set_xlim(-3,3)
        ax_residuals.set_ylim(-2000, 2000)
        ax_residuals.set_xlabel('Time from mid-transit (hours)', fontsize = 12)
        counter += 1

    plt.tight_layout()
    plt.savefig(f'juliet_fits/{target}/non_tess_lcs_{target}.png')

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

    if parameter in ['P_p1', 't0_p1']:
        plt.errorbar(data['prior']-data['posterior'],data['prior']-data['posterior'],
                     fmt='o', xerr=data['prior_err'],
                     yerr=[data['posterior_err_lower'], data['posterior_err_upper']])
    else:
        plt.errorbar(data['prior'], data['posterior'], fmt='o', xerr=data['prior_err'],
                 yerr=[data['posterior_err_lower'], data['posterior_err_upper']])

    # Add a line showing where perfect agreement would be
    if parameter not in ['P_p1', 't0_p1']:
        plt.axline([0,0], slope=1, alpha = 0.5, color = 'gray')

    plt.savefig(f'plots/{parameter}_comparison_plot.png')

    cur.close()
    conn.close()
