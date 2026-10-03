"""Dataset acquisition, name parsing and loading."""

import glob
import json
import os
import shutil
import zipfile

import numpy as np

# Column layout of one simulation array (see Phase 0, section 4)
COLS = ['x', 'y', 'uinf_x', 'uinf_y', 'sdf', 'nx', 'ny', 'Ux', 'Uy', 'p', 'nut', 'surface']
IN_COLS = [0, 1, 2, 3, 4, 5, 6]      # x, y, uinf_x, uinf_y, sdf, nx, ny
OUT_COLS = [7, 8, 9, 10]             # Ux, Uy, p, nut
OUT_NAMES = ['Ux', 'Uy', 'p', 'nut']
SURF_COL = 11


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def default_paths(use_drive):
    """(DRIVE_DIR, ROOT_DL, WORK_DIR): Drive cache, local extraction folder, folder for outputs.

    On Colab these are the paths the notebooks have always used. Elsewhere, data and outputs go
    to ./airfrans_data and ./work next to the notebook.
    """
    drive_dir = '/content/drive/MyDrive/airfrans'
    if in_colab():
        root_dl = '/content/airfrans_data'
        work_dir = drive_dir if use_drive else '/content'
    else:
        if use_drive:
            raise RuntimeError('USE_DRIVE = True needs Google Colab; set USE_DRIVE = False to run locally.')
        root_dl = os.path.abspath('airfrans_data')
        work_dir = os.path.abspath('work')
    return drive_dir, root_dl, work_dir


def _have_data(root_dl):
    return bool(glob.glob(f'{root_dl}/**/manifest.json', recursive=True))


def _gb(path):
    return os.path.getsize(path) / 1e9


def prepare_dataset(use_drive, drive_dir, root_dl):
    """Make the extracted dataset available under root_dl.

    Restores it from the Drive cache when there is one, otherwise downloads it, and caches the zip
    to Drive only after a successful extraction so a partial download is never saved.
    """
    import airfrans as af

    drive_zip = f'{drive_dir}/Dataset.zip'
    local_zip = f'{root_dl}/Dataset.zip'
    if use_drive:
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs(drive_dir, exist_ok=True)
    os.makedirs(root_dl, exist_ok=True)

    if _have_data(root_dl):
        print('Dataset already extracted on this machine.')
    elif use_drive and os.path.exists(drive_zip):
        print(f'Copying {_gb(drive_zip):.1f} GB zip from Drive (unzipping from local disk is much faster)...')
        shutil.copy(drive_zip, local_zip)
        print('Unzipping...')
        with zipfile.ZipFile(local_zip) as z:
            z.extractall(root_dl)
    else:
        print('No cached copy found. Downloading from the AirfRANS server (slow, one time only)...')
        af.dataset.download(root=root_dl, file_name='Dataset', unzip=True, OpenFOAM=False)

    if use_drive and os.path.exists(local_zip) and _have_data(root_dl):
        if not os.path.exists(drive_zip) or _gb(drive_zip) != _gb(local_zip):
            print(f'Saving {_gb(local_zip):.1f} GB zip to Drive (10-20 min, one time only)...')
            shutil.copy(local_zip, drive_zip)
        print(f'Drive copy OK: {_gb(drive_zip):.2f} GB  (local {_gb(local_zip):.2f} GB)')

    print('Data ready:', _have_data(root_dl))
    return _have_data(root_dl)


def locate_root(root_dl, remove_zip=True):
    """Folder containing manifest.json; the local zip is deleted once extracted to free disk."""
    root = os.path.dirname(glob.glob(f'{root_dl}/**/manifest.json', recursive=True)[0])
    zip_path = f'{root_dl}/Dataset.zip'
    if remove_zip and os.path.exists(zip_path):
        os.remove(zip_path)
    return root


def load_manifest(root):
    with open(f'{root}/manifest.json') as f:
        return json.load(f)


def parse_name(name):
    """AirfRANS names look like airFoil2D_SST_<Uinf>_<AoA deg>_<NACA params...>"""
    parts = name.split('_')
    u_inf, aoa = float(parts[2]), float(parts[3])
    naca = [float(v) for v in parts[4:]]
    return u_inf, aoa, naca


def family(name):
    return '4-digit' if len(parse_name(name)[2]) == 3 else '5-digit'


def sim_to_array(sim):
    """One simulation as an (N, 12) float32 array in the COLS layout."""
    u_in = (np.array([np.cos(sim.angle_of_attack), np.sin(sim.angle_of_attack)])
            * sim.inlet_velocity).reshape(1, 2) * np.ones_like(sim.sdf)
    return np.concatenate([sim.position, u_in, sim.sdf, sim.normals, sim.velocity,
                           sim.pressure, sim.nu_t, sim.surface.reshape(-1, 1)], axis=-1).astype(np.float32)


def load_split(root, manifest, key):
    """Read the simulations of one manifest split, converting each to float32 immediately."""
    import airfrans as af
    arrays, names = [], []
    for k, s in enumerate(manifest[key]):
        arrays.append(sim_to_array(af.Simulation(root=root, name=s))); names.append(s)
        if (k + 1) % 50 == 0:
            print(f'  {key}: {k + 1}/{len(manifest[key])}')
    return arrays, names
