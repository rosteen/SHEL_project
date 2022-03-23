import sqlite3 as sql
import csv

from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.time import Time
from barycorrpy import utc_tdb

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

    cur.execute(create_targets_table)
    cur.execute(create_int_table)
    cur.execute(create_hubble_table)
    cur.execute(create_ref_table)
    cur.execute(create_rv_table)
    cur.execute(create_lc_table)

    conn.commit()
    conn.close()


def helio_to_bary(coords, hjd, obs_name):
    """
    From https://gist.github.com/StuartLittlefair/4ab7bb8cf21862e250be8cb25f72bb7a
    with some minor edits to the coordinate handling
    """

    ra = coords[0].split(":")
    ra = int(ra[0]) + int(ra[1])/60 + float(ra[2])/3600
    dec = coords[1].split(":")
    if int(dec[0]) < 1:
        dec = int(dec[0]) - int(dec[1])/60 - float(dec[2])/3600
    else:
        dec = int(dec[0]) + int(dec[1])/60 + float(dec[2])/3600
    star = SkyCoord(ra=ra*u.hour, dec=dec*u.deg)

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
                   target=None, instrument=None, inst_list=None, inst_col=None,
                   delimiter="\t", time_type="BJD-TDB", time_offset=0, debug=False):
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
                instrument_id = cur.execute("select id from instruments where "
                                            f"name='{instrument}'").fetchone()[0]
            # Get target ID if stored in a column
            if target_col is not None:
                if data[target_col] != target:
                    bad_target = False
                    target = data[target_col]
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

            t = float(data[t_col]) + time_offset
            #TODO: If we need to convert to BJD, get observatory name
            if time_type == "BJD-TDB":
                bjd = t
            else:
                stmt = f'select sitename from instruments where name = "{instrument}"'
                obsname = cur.execute(stmt).fetchone()[0]
                if obsname is None or ra is None or dec is None:
                    raise ValueError("Observatory sitename and target RA and Dec "
                                     "must be populated to convert to BJD")

            if time_type == "HJD":
                bjd = helio_to_bary((ra, dec), t, obsname)
            elif time_type == "JD":
                JDUTC = Time(t, format='jd', scale='utc')
                bjd = utc_tdb.JDUTC_to_BJDTDB(JDUTC, ra=ra, dec=dec, obsname=obsname)
            if debug:
                print(f"Original time: {t}, BJD-TDB: {bjd}")

            try:
                rv = float(data[rv_col])
            except ValueError:
                # Generally this is due to an asterix flagging this RV
                continue

            rv_err = float(data[err_col])
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
