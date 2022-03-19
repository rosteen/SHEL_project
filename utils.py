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


def ingest_rv_data(filename, ref_url, t_col, rv_col, err_col, instrument=None,
                   target_col=None, target=None, inst_list=None, delimiter="\t",
                   time_type="BJD-UTC", time_offset=0, debug=False):
    """
    Ingest a CSV file with RV data into the database. User must provide column numbers
    for time, RV, and RV error. Non-data rows are assumed to be commented out with #.
    """
    # Check for required inputs
    if target_col is None and target is None:
        raise ValueError("Must specify either a target (for single-target file) or"
                         "column with target names.")

    if instrument is None and inst_list is None:
        raise ValueError("Must specify either an instrument (for a single-instrument"
                         " file) or a list of instrument strings to look for in comment lines.")

    # Connect to database
    conn = sql.connect('shel_database.sqlite')
    cur = conn.cursor()

    # get reference ID
    refnum = cur.execute(f"select id from data_refs where url='{url}'").fetchone()
    if refnum is None:
        stmt = f"insert into data_refs (local_filename, url) values ('{fname}', '{url}')"
        cur.execute(stmt)
        refnum = cur.execute(f"select id from data_refs where url='{url}'").fetchone()
    if debug:
        print(f"Refnum: {refnum}")

    # Get target_id as well as RA and Dec in case we need them for time conversion
    if target is not None:
        res = cur.execute(f'select id, ra, dec from targets where name = "{target}"').fetchone()
        target_id, ra, dec = res
        if debug:
            print(target_id, ra, dec)

    with open(filename) as f:
        freader = csv.reader(f, delimiter=delimiter)
        for row in freader:
            if row[0][0] == "#":
                if inst_list is not None and row[0][1:] in inst_list:
                    instrument = row[0][1:]
                continue

            if target_col is not None:
                if row[target_col] != target:
                    target = row[target_col]
                    stmt = f'select id, ra, dec from targets where name = "{target}"'
                    res = cur.execute(stmt).fetchone()
                    target_id, ra, dec = res
                    if debug:
                        print(target_id, ra, dec)

            t = row[t_col] + time_offset
            #TODO: If we need to convert to BJD, get observatory name
            if time_type != "BJD":
                stmt = f'select sitename from instruments where name = "{instrument}"'
                obsname = cur.execute(stmt).fetchone()[0]
                if obsname is None or ra is None or dec is None:
                    raise ValueError("Observatory sitename and target RA and Dec "
                                     "must be populated to convert to BJD")

            if time_type == "HJD":
                t = helio_to_bary((ra, dec), t, obsname)
            elif time_type == "JD":
                JDUTC = Time(2458000, format='jd', scale='utc')
                t = utc_tdb.JDUTC_to_BJDTDB(JDUTC, ra=ra, dec=rec, obsname=obsname)


            rv = row[rv_col]
            rv_err = row[err_col]
            if debug:
                print(instrument, bjd, rv, rv_err, target)

            stmt = ("insert into radial_velocity (target_id, reference_id, instrument,"
                    f"bjd, rv, rv_err) values ({target}, {ref_id}, {instrument}, "
                    f"{bjd}, {rv}, {rv_err})")

            if debug:
                print(stmt)
            else:
                # Leaving this commented out until after testing, just in case
                #cur.execute(stmt)
                pass
