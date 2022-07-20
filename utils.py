import csv
import numpy as np
import os
import sqlite3 as sql

from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from barycorrpy import utc_tdb
import juliet
from juliet.utils import mag_to_flux

def create_shel_db():
    conn = sql.connect("shel_database.sqlite")
    cur = conn.cursor()

    create_targets_table = """CREATE TABLE IF NOT EXISTS targets (
                            id integer PRIMARY KEY,
                            name text NOT NULL UNIQUE,
                            ra text,
                            dec text
                            );"""

    create_hubble_table = """CREATE TABLE IF NOT EXISTS hubble_programs (
                            id integer PRIMARY KEY,
                            stis_optical_obs int,
                            target_type text);"""

    create_ref_table = """CREATE TABLE IF NOT EXISTS data_refs (
                            id integer PRIMARY KEY,
                            url text NOT NULL,
                            local_filename text);"""

    create_int_table = """CREATE TABLE IF NOT EXISTS instruments (
                            id integer PRIMARY KEY,
                            name text NOT NULL,
                            sitename text);"""

    create_rv_table = """CREATE TABLE IF NOT EXISTS radial_velocities (
                            target_id int,
                            reference_id int,
                            instrument int,
                            bjd real,
                            rv real,
                            rv_err real,
                            FOREIGN KEY (target_id) REFERENCES targets (id),
                            FOREIGN KEY (instrument) REFERENCES instruments (id),
                            FOREIGN KEY (reference_id) REFERENCES data_refs (id));"""

    create_lc_table = """CREATE TABLE IF NOT EXISTS light_curves (
                            target_id int,
                            reference_id int,
                            instrument int,
                            bjd real,
                            flux real,
                            flux_err real,
                            FOREIGN KEY (target_id) REFERENCES targets (id),
                            FOREIGN KEY (reference_id) REFERENCES data_refs (id),
                            FOREIGN KEY (instrument) REFERENCES instruments (id));"""

    create_stellar_table = """CREATE TABLE IF NOT EXISTS stellar_parameters (
                              target_id integer PRIMARY KEY,
                              imass real,imass_err real, age real, age_err real,
                              mass real, mass_err real, Teff real, Teff_err real,
                              log_g real, log_g_err real, Rs real, Rs_err real,
                              L real, L_err real, rho real, rho_err real,
                              Av real, Av_err real);"""

    create_results_table = """CREATE TABLE IF NOT EXISTS system_parameters (
                            id integer PRIMARY KEY,
                            target_id int NOT NULL,
                            parameter text,
                            prior real,
                            prior_err real,
                            posterior real,
                            posterior_err_upper real,
                            posterior_err_lower real,
                            FOREIGN KEY (target_id) REFERENCES targets (id)
                            );"""

    cur.execute(create_targets_table)
    cur.execute(create_int_table)
    cur.execute(create_hubble_table)
    cur.execute(create_ref_table)
    cur.execute(create_rv_table)
    cur.execute(create_lc_table)
    cur.execute(create_stellar_table)
    cur.execute(create_results_table)

    conn.commit()
    conn.close()


def coords_to_SkyCoord(coords):
    ra = coords[0].split(":")
    ra = int(ra[0]) + int(ra[1])/60 + float(ra[2])/3600
    dec = coords[1].split(":")
    if int(dec[0]) < 1:
        dec = int(dec[0]) - int(dec[1])/60 - float(dec[2])/3600
    else:
        dec = int(dec[0]) + int(dec[1])/60 + float(dec[2])/3600

    star = SkyCoord(ra=ra*u.hour, dec=dec*u.deg)
    return star


def helio_to_bary(coords, hjd, obs_name):
    """
    From https://gist.github.com/StuartLittlefair/4ab7bb8cf21862e250be8cb25f72bb7a
    with some minor edits to the coordinate handling
    """
    star = coords_to_SkyCoord(coords)

    helio = Time(hjd, scale='utc', format='jd')
    obs = EarthLocation.of_site(obs_name)
    ltt = helio.light_travel_time(star, 'heliocentric', location=obs)
    guess = helio - ltt
    # if we assume guess is correct - how far is heliocentric time away from true value?
    delta = (guess + guess.light_travel_time(star, 'heliocentric', obs)).jd  - helio.jd
    # apply this correction
    guess -= delta * u.d

    ltt = guess.light_travel_time(star, 'barycentric', obs)

    return guess.tdb + ltt


def ingest_rv_data(filename, ref_url, t_col, rv_col, err_col, target_col=None,
                   target=None, target_prefix = None, instrument=None, inst_list=None,
                   inst_col=None, delimiter="\t", time_type="BJD-TDB", time_offset=0,
                   unit='m/s', filter_target=None, debug=False):
    """
    Ingest a CSV file with RV data into the database. User must provide column numbers
    for time, RV, and RV error. Non-data rows are assumed to be commented out with #.
    """
    # Check for required inputs
    if target_col is None and target is None:
        raise ValueError("Must specify either a target (for single-target file) or"
                         "column with target names.")

    if instrument is None and inst_list is None and inst_col is None:
        raise ValueError("Must specify either an instrument (for a single-instrument"
                         " file), a column holding instrument names, or a list of "
                         "instrument strings to look for in comment lines.")

    # Connect to database
    conn = sql.connect('shel_database.sqlite')
    cur = conn.cursor()

    # get reference ID
    ref_id = cur.execute(f"select id from data_refs where url='{ref_url}'").fetchone()
    if ref_id is None:
        stmt = f"insert into data_refs (local_filename, url) values ('{filename}', '{ref_url}')"
        cur.execute(stmt)
        ref_id = cur.execute(f"select id from data_refs where url='{ref_url}'").fetchone()[0]
    else:
        ref_id = ref_id[0]
    if debug:
        print(f"Refnum: {ref_id}")

    # Get instrument ID. Instrument info must be pre-loaded
    if instrument is not None:
        instrument_id = cur.execute("select id from instruments where "
                                    f"name='{instrument}'").fetchone()[0]

    # Get target_id as well as RA and Dec in case we need them for time conversion
    if target is not None:
        res = cur.execute(f'select id, ra, dec from targets where name = "{target}"').fetchone()
        target_id, ra, dec = res
        if debug:
            print(target_id, ra, dec)

    with open(f"data/radial_velocity/{filename}") as f:
        freader = csv.reader(f, delimiter=delimiter)
        for row in freader:
            if row[0][0] == "#":
                if inst_list is not None and row[0][1:] in inst_list:
                    instrument = row[0][1:]
                    instrument_id = cur.execute("select id from instruments where "
                                                f"name='{instrument}'").fetchone()[0]
                continue
            else:
                if delimiter == " ":
                    data = [x for x in row if x != ""]
                else:
                    data = row

            # Get instrument ID if there is an instrument name column
            if inst_col is not None:
                instrument = data[inst_col]
                try:
                    instrument_id = cur.execute("select id from instruments where "
                                                f"name='{instrument}'").fetchone()[0]
                except:
                    print(instrument)
                    raise
            # Get target ID if stored in a column
            if target_col is not None:
                if data[target_col] != target:
                    bad_target = False
                    target = data[target_col]
                    if target_prefix is not None:
                        target = target_prefix + target
                    if filter_target is not None:
                        # Skip anything but the one we're reprocessing
                        if target != filter_target:
                            print(f"Skipping {target}")
                            bad_target=True
                            continue
                    stmt = f'select id, ra, dec from targets where name = "{target}"'
                    res = cur.execute(stmt).fetchone()
                    if res is None:
                        print(f"Skipping {target}")
                        bad_target = True
                        continue
                    target_id, ra, dec = res
                    if debug:
                        print(target_id, ra, dec)
                if bad_target:
                    continue

            # Convert time to BJD-UTC if needed
            t = float(data[t_col]) + time_offset
            if time_type == "BJD-TDB":
                bjd = t
            elif time_type in ("JD", "HJD"):
                stmt = f'select sitename from instruments where name = "{instrument}"'
                obsname = cur.execute(stmt).fetchone()[0]
                if obsname is None or ra is None or dec is None:
                    raise ValueError(f"Observatory sitename and RA and Dec  for {target}"
                                     " must be populated to convert to BJD")

            if time_type == "HJD":
                bjd = helio_to_bary((ra, dec), t, obsname)
            elif time_type == "JD":
                JDUTC = Time(t, format='jd', scale='utc')
                star = coords_to_SkyCoord((ra, dec))
                bjd = utc_tdb.JDUTC_to_BJDTDB(JDUTC, ra=star.ra.deg, dec=star.dec.deg,
                                              obsname=obsname)[0][0]
            elif time_type == "BJD-UTC":
                utc = Time(t, format='jd', scale='utc')
                bjd = utc.tdb.value

            if debug:
                print(f"Original time: {t}, BJD-TDB: {bjd}")

            try:
                rv = float(data[rv_col].strip())
            except ValueError:
                # Generally this is due to an asterix flagging this RV
                continue

            rv_err = float(data[err_col].strip())

            if unit=='km/s':
                rv *= 1000
                rv_err *= 1000

            if debug:
                print(instrument_id, bjd, rv, rv_err, target)

            stmt = ("insert into radial_velocities (target_id, reference_id, instrument,"
                    f"bjd, rv, rv_err) values ({target_id}, {ref_id}, {instrument_id}, "
                    f"{bjd}, {rv}, {rv_err})")

            if debug:
                print(stmt)
            else:
                cur.execute(stmt)

    # Commit the database changes
    conn.commit()
    cur.close()
    conn.close()


def ingest_tess_data(directory, target, db_path="shel_database.sqlite",
                     debug=False):
    """
    Ingest any TESS light curves in subdirectories of given directory.
    Will look for subdirectories ending in '-s', which should hold light curves.
    """
    subdirs = [x[0] for x in os.walk(directory)]
    lc_files = []
    for subdir in subdirs:
        if subdir[-2:] == "-s":
            lc_files.append([subdir, subdir.split('/')[-1]])

    # Connect to database and get target ID
    conn = sql.connect(db_path)
    cur = conn.cursor()

    target_id = cur.execute(f"select id from targets where name='{target}'").fetchone()[0]

    for path_info in lc_files:
        fname = f"{path_info[0]}/{path_info[1]}_lc.fits"
        with fits.open(fname, mode="readonly") as hdulist:
            tess_bjds = hdulist[1].data['TIME']
            sap_fluxes = hdulist[1].data['SAP_FLUX']
            pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
            pdcsap_err = hdulist[1].data['PDCSAP_FLUX_ERR']
            quality = hdulist[1].data['QUALITY']

            # Reject bad quality flags
            bad_bits = np.array([1,2,3,4,5,6,8,10,12])
            value = 0
            for v in bad_bits:
                value = value + 2**(v-1)
            bad_data = np.bitwise_and(quality, value) >= 1
            bad_data[np.isnan(pdcsap_fluxes)] = True

            # Reject anything below this to get non-transit mean
            min_threshold = (np.nanmean(pdcsap_fluxes[~bad_data]) -
                             2*np.nanstd(pdcsap_fluxes[~bad_data]))
            non_transit = np.where(pdcsap_fluxes[~bad_data] > min_threshold)

            # Convert the times to unsubtracted BJD
            good_times = tess_bjds[~bad_data] + 2457000
            normalized_flux = (pdcsap_fluxes[~bad_data] /
                               np.nanmean(pdcsap_fluxes[~bad_data][non_transit]))
            normalized_errs = normalized_flux*(pdcsap_err[~bad_data] /
                                               pdcsap_fluxes[~bad_data])
            for i in range(len(good_times)):
                stmt = ("INSERT INTO light_curves (target_id, instrument, bjd, "
                        f"flux, flux_err) values ({target_id}, 1, {good_times[i]},"
                        f" {normalized_flux[i]}, {normalized_errs[i]})")
                if debug:
                    print(f"First insert for {fname}:")
                    print(stmt)
                    break
                else:
                    cur.execute(stmt)

        # Commit the data from each file
        conn.commit()

    cur.close()
    conn.close()


def ingest_lc_data(filename, ref_url, t_col, lc_col, err_col, target_col=None,
                   target=None, instrument=None, inst_list=None, inst_col=None,
                   delimiter="\t", time_type="BJD-TDB", time_offset=0,
                   data_type="flux", filter_target=None, constant_error=None,
                   debug=False):
    """
    Ingest non-TESS light curve data. Most of this code is shared with the RV
    file ingestion code and should be consolidated.
    """
    if target_col is None and target is None:
        raise ValueError("Must specify either a target (for single-target file) or"
                         "column with target names.")

    if instrument is None and inst_list is None and inst_col is None:
        raise ValueError("Must specify either an instrument (for a single-instrument"
                         " file), a column holding instrument names, or a list of "
                         "instrument strings to look for in comment lines.")

    # Connect to database
    conn = sql.connect('shel_database.sqlite')
    cur = conn.cursor()

    # get reference ID
    ref_id = cur.execute(f"select id from data_refs where url='{ref_url}'").fetchone()
    if ref_id is None:
        stmt = f"insert into data_refs (local_filename, url) values ('{filename}', '{ref_url}')"
        cur.execute(stmt)
        ref_id = cur.execute(f"select id from data_refs where url='{ref_url}'").fetchone()[0]
    else:
        ref_id = ref_id[0]
    if debug:
        print(f"Refnum: {ref_id}")

    # Get target_id as well as RA and Dec in case we need them for time conversion
    if target is not None:
        res = cur.execute(f'select id, ra, dec from targets where name = "{target}"').fetchone()
        target_id, ra, dec = res
        if debug:
            print(target_id, ra, dec)

    # Get instrument ID. Instrument info must be pre-loaded
    if instrument is not None:
        instrument_id = cur.execute("select id from instruments where "
                                    f"name='{instrument}'").fetchone()[0]

    with open(f"data/light_curves/{filename}", "r") as f:
        freader = csv.reader(f, delimiter=delimiter)
        for row in freader:
            if delimiter == " ":
                data = [x for x in row if x != ""]
            else:
                data = row

            if data[0][0] == "#":
                if inst_list is not None and " ".join(data[1:]) in inst_list:
                    instrument = " ".join(data[1:])
                    instrument_id = cur.execute("select id from instruments where "
                                                f"name='{instrument}'").fetchone()[0]
                continue

            # Get target ID if stored in a column
            if target_col is not None:
                temp_target = data[target_col]
                if temp_target[-1] == "b":
                    temp_target = temp_target[:-1]
                if temp_target[-1] == "A":
                    temp_target = temp_target[:-1]
                temp_target = temp_target.upper()

                if temp_target != target:
                    bad_target = False
                    target = temp_target
                    if filter_target is not None:
                        # Skip anything but the one we're reprocessing
                        if target != filter_target:
                            print(f"Skipping {target}")
                            bad_target=True
                            continue
                    stmt = f'select id, ra, dec from targets where name = "{target}"'
                    res = cur.execute(stmt).fetchone()
                    if res is None:
                        print(f"Skipping {target}")
                        print(stmt)
                        bad_target = True
                        continue
                    target_id, ra, dec = res
                    if debug:
                        print(target_id, ra, dec)
                if bad_target:
                    continue

            # Get instrument ID if there is an instrument name column
            if inst_col is not None:
                instrument = data[inst_col]
                instrument_id = cur.execute("select id from instruments where "
                                            f"name='{instrument}'").fetchone()[0]

            # Convert time to BJD-UTC if needed
            t = float(data[t_col]) + time_offset
            if time_type == "BJD-TDB":
                bjd = t
            elif time_type in ("JD", "HJD"):
                stmt = f'select sitename from instruments where name = "{instrument}"'
                obsname = cur.execute(stmt).fetchone()[0]
                if obsname is None or ra is None or dec is None:
                    raise ValueError(f"Observatory sitename and RA and Dec  for {target}"
                                     " must be populated to convert to BJD")

            if time_type == "HJD":
                bjd = helio_to_bary((ra, dec), t, obsname)
            elif time_type == "JD":
                JDUTC = Time(t, format='jd', scale='utc')
                star = coords_to_SkyCoord((ra, dec))
                bjd = utc_tdb.JDUTC_to_BJDTDB(JDUTC, ra=star.ra.deg, dec=star.dec.deg,
                                              obsname=obsname)[0][0]
            elif time_type == "BJD-UTC":
                utc = Time(t, format='jd', scale='utc')
                bjd = utc.tdb.value

            if debug:
                print(f"Original time: {t}, BJD-TDB: {bjd}")

            flux = data[lc_col]
            if err_col is None:
                if constant_error is not None:
                    flux_err = constant_error
                else:
                    flux_err = -1
            else:
                flux_err = data[err_col]

            # Convert mag to flux if needed
            if data_type.lower() == "mag":
                # Have to work around the fact that this function assumes arrays
                flux, flux_err = mag_to_flux([float(flux)], [float(flux_err)])
                flux = flux[0]
                flux_err = flux_err[0]

            stmt = ("insert into light_curves (target_id, reference_id, instrument, bjd, flux, flux_err) "
                    f"values ({target_id}, {ref_id}, {instrument_id}, {bjd}, {flux}, {flux_err})")
            if debug:
                print(stmt)
            else:
                cur.execute(stmt)

    conn.commit()
    cur.close()
    conn.close()

def load_priors(target, P, t0, a, b, ecc, p, duration):
    """
    Load prior values and error from the literature.
    """
    conn = sql.connect('shel_database.sqlite')
    cur = conn.cursor()

    parameters = ['P', 't0', 'a', 'b', 'ecc', 'p', 'duration']
    for param in parameters:
        pass

    conn.commit()

    cur.close()
    conn.close()

def load_results(target, n_planets=1):
    """
    Load the results for the parameters we're interested in.
    """
    dataset = juliet.load(input_folder = f'juliet_fits/{target}/')
    results = dataset.fit(use_dynesty=True, dynamic=True)
    parameters = ['P', 't0', 'a', 'b', 'ecc', 'p']

    conn = sql.connect('shel_database.sqlite')
    cur = conn.cursor()

    target_id = cur.execute(f"select id from targets where name='{target}'").fetchone()[0]

    for i in range(1, n_planets+1):
        for param in parameters:
            param_id = f"{param}_p{i}"
            result = results.posteriors['posterior_samples'][param_id]
            param_med, param_upper, param_lower = juliet.utils.get_quantiles(result)
            err_upper = param_upper - param_med
            err_lower = param_med - param_lower

            stmt = ("insert into system_parameters (target_id, parameter, posterior, "
                    f"posterior_err_upper, posterior_err_lower) values ({target_id}, "
                    f"'{param_id}', {param_med}, {err_upper}, {err_lower})")
            cur.execute(stmt)

    conn.commit()

    cur.close()
    conn.close()
