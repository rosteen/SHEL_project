import sqlite3 as sql
import csv


def create_shel_db():
    conn = sql.connect("shel_database.sqlite")
    cur = conn.cursor()

    create_targets_table = """CREATE TABLE IF NOT EXISTS targets (
                            id integer PRIMARY KEY,
                            name text NOT NULL,
                            ra text NOT NULL,
                            dec text NOT NULL
                            );"""

    create_hubble_table = """CREATE TABLE IF NOT EXISTS hubble_programs (
                            id integer PRIMARY KEY,
                            stis_optical_obs int,
                            target_type text);"""

    create_ref_table = """CREATE TABLE IF NOT EXISTS data_refs (
                            id integer PRIMARY KEY,
                            url text NOT NULL,
                            local_filename text);"""

    create_rv_table = """CREATE TABLE IF NOT EXISTS radial_velocities (
                            target_id int,
                            reference_id int,
                            instrument text,
                            bjd real,
                            rv real,
                            rv_err real,
                            FOREIGN KEY (target_id) REFERENCES targets (id),
                            FOREIGN KEY (reference_id) REFERENCES data_refs (id));"""

    create_lc_table = """CREATE TABLE IF NOT EXISTS radial_velocities (
                            target_id int,
                            reference_id int,
                            instrument text,
                            bjd real,
                            flux real,
                            flux_err real,
                            FOREIGN KEY (target_id) REFERENCES targets (id),
                            FOREIGN KEY (reference_id) REFERENCES data_refs (id));"""

    cur.execute(create_targets_table)
    cur.execute(create_hubble_table)
    cur.execute(create_ref_table)
    cur.execute(create_rv_table)
    cur.execute(create_lc_table)

    conn.commit()
    conn.close()


def ingest_rv_data(filename, t_col, rv_col, err_col, instrument=None,
                   target_col=None, target=None, inst_list=None, delimiter="/t"):
    """
    Ingest a CSV file with RV data into the database. User must provide column numbers
    for time, RV, and RV error. Non-data rows are assumed to be commented out with #.
    """
    if target_col is None and target is None:
        raise ValueError("Must specify either a target (for single-target file) or"
                         "column with target names.")

    if instrument is None and inst_list is None:
        raise ValueError("Must specify either an instrument (for a single-instrument"
                         " file) or a list of instrument strings to look for in comment lines.")

    # Insert into target table if needed
    if target is not None:

    with open(filename) as f:
        freader = csv.reader(f, delimiter=delimiter)
        for row in freader:
            if row[0][0] == "#"
                if inst_list is not None and row[0][1:] in inst_list:
                    instrument = row[0][1:]
                continue
            bjd = row[t_col]
            rv = row[rv_col]
            rv_err = row[err_col]
            if target_col is not None:
                if row[target_col] != target:
                    target = row[target_col]
                    # Insert into target table if needed, get target ID





