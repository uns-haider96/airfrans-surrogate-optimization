"""Shared helpers for the AirfRANS surrogate-optimization notebooks.

Every notebook imports from here instead of carrying its own copy, so a fix
made once applies to all phases.
"""

__version__ = '0.2.0'


def print_versions():
    """Print the versions of the packages the notebooks depend on (recorded with every run)."""
    import platform
    from importlib.metadata import PackageNotFoundError, version
    print(f'{"python":14s}{platform.python_version()}')
    # Installed distribution versions: airfrans 0.1.5.1, for one, still reports __version__ = '0.1.2'
    for dist in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'airfrans', 'pyvista', 'vtk', 'matplotlib'):
        try:
            print(f'{dist:14s}{version(dist)}')
        except PackageNotFoundError:
            print(f'{dist:14s}not installed')
    print(f'{"helpers":14s}{__version__}')
