# data_cleaning.py
"""
Data Cleaning Module
====================

Module for cleaning temporary/obsolete files and folders.

NOTE: This module should be run ONLY via main_c.py

Available commands:
    python main_c.py --clean-results    # Clean empty/obsolete Results folders
    python main_c.py --clean-temp       # Clean temporary files (.tmp, .pyc, __pycache__)
    python main_c.py --clean-all        # Full cleanup
"""

from pathlib import Path
import shutil
import logging

logger = logging.getLogger("DataCleaning")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "root_dir": Path("."),
    "results_dir": Path("Results"),

    # Temporary file patterns to delete
    "temp_patterns": [
        "*.tmp",
        "*.pyc",
        "*.pyo",
        "*~",
        ".DS_Store",
        "Thumbs.db",
    ],

    # Cache directories to delete
    "cache_dirs": [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "*.egg-info",
    ],

    # Root files to delete (old output)
    "root_cleanup_patterns": [
        "*.png",  # Old plots saved in root
        "thesis_pipeline.log",  # Log file
    ],

    # Valid Results directories (do NOT delete)
    "valid_results_dirs": [
        "fat_tails_kurtosis",
        "implied_volatility_smile",
        "volatility_term_structure",
        "inverse_options",
    ],
}


# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

def clean_temp_files():
    """
    Clean temporary files and cache.

    Removes:
    - .tmp, .pyc, .pyo files
    - __pycache__, .pytest_cache directories
    - System files (.DS_Store, Thumbs.db)
    """
    logger.info("Cleaning temporary files...")

    removed_files = 0
    removed_dirs = 0

    root = CONFIG["root_dir"]

    # Remove temporary files
    for pattern in CONFIG["temp_patterns"]:
        for f in root.rglob(pattern):
            try:
                f.unlink()
                removed_files += 1
                logger.debug(f"Removed: {f}")
            except Exception as e:
                logger.warning(f"Cannot remove {f}: {e}")

    # Remove cache directories
    for pattern in CONFIG["cache_dirs"]:
        for d in root.rglob(pattern):
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                    removed_dirs += 1
                    logger.debug(f"Removed directory: {d}")
                except Exception as e:
                    logger.warning(f"Cannot remove {d}: {e}")

    logger.info(f"Temp cleanup completed: {removed_files} files, {removed_dirs} directories removed")
    return removed_files, removed_dirs


def clean_root_files():
    """
    Clean obsolete files from the root directory.

    Removes:
    - Old .png files saved in root (now go in Results/)
    - Log files
    """
    logger.info("Cleaning obsolete root files...")

    removed = 0
    root = CONFIG["root_dir"]

    for pattern in CONFIG["root_cleanup_patterns"]:
        for f in root.glob(pattern):
            # Don't delete if it's in a subdirectory
            if f.parent == root:
                try:
                    f.unlink()
                    removed += 1
                    logger.info(f"Removed from root: {f.name}")
                except Exception as e:
                    logger.warning(f"Cannot remove {f}: {e}")

    logger.info(f"Root cleanup completed: {removed} files removed")
    return removed


def clean_results_folders():
    """
    Clean obsolete or empty Results folders.

    Removes:
    - Timestamp-named folders (e.g. 20240101_120000)
    - Empty folders
    - Folders not in the valid folders list
    """
    logger.info("Cleaning obsolete Results folders...")

    results_dir = CONFIG["results_dir"]
    if not results_dir.exists():
        logger.info("Results folder does not exist, nothing to clean")
        return 0

    removed = 0
    valid_dirs = CONFIG["valid_results_dirs"]

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        dir_name = subdir.name

        # Check if it's a valid folder
        if dir_name in valid_dirs:
            continue

        # Check if it's a timestamp folder (format: YYYYMMDD_HHMMSS)
        is_timestamp = (len(dir_name) == 15 and
                       dir_name[8] == '_' and
                       dir_name[:8].isdigit() and
                       dir_name[9:].isdigit())

        # Check if empty
        is_empty = not any(subdir.iterdir())

        if is_timestamp or is_empty:
            try:
                shutil.rmtree(subdir)
                removed += 1
                reason = "timestamp" if is_timestamp else "empty"
                logger.info(f"Removed {reason} folder: {dir_name}")
            except Exception as e:
                logger.warning(f"Cannot remove {subdir}: {e}")

    logger.info(f"Results cleanup completed: {removed} folders removed")
    return removed


def show_current_status():
    """Show current status of folders and files."""

    print("\n" + "="*60)
    print("CURRENT STATUS")
    print("="*60)

    # Files in root
    root = CONFIG["root_dir"]
    root_files = list(root.glob("*.py")) + list(root.glob("*.png")) + list(root.glob("*.log"))

    print("\nPython files in root:")
    for f in sorted(root.glob("*.py")):
        print(f"  - {f.name}")

    png_files = list(root.glob("*.png"))
    if png_files:
        print("\nPNG files in root (to clean):")
        for f in png_files:
            print(f"  - {f.name}")

    log_files = list(root.glob("*.log"))
    if log_files:
        print("\nLog files in root:")
        for f in log_files:
            print(f"  - {f.name}")

    # Results folders
    results_dir = CONFIG["results_dir"]
    if results_dir.exists():
        print(f"\nFolders in Results/:")
        for subdir in sorted(results_dir.iterdir()):
            if subdir.is_dir():
                n_files = len(list(subdir.rglob("*")))
                status = "OK" if subdir.name in CONFIG["valid_results_dirs"] else "TO CLEAN?"
                print(f"  - {subdir.name}/ ({n_files} files) [{status}]")

    # Cache
    pycache = list(root.rglob("__pycache__"))
    if pycache:
        print(f"\n__pycache__ directories found: {len(pycache)}")

    print("="*60)


# =============================================================================
# PUBLIC FUNCTIONS (called from main_c.py)
# =============================================================================

def run_clean_temp():
    """Clean temporary files."""
    logger.info("="*60)
    logger.info("TEMPORARY FILE CLEANUP")
    logger.info("="*60)

    files, dirs = clean_temp_files()

    print(f"\nCleanup completed:")
    print(f"  - Temporary files removed: {files}")
    print(f"  - Cache directories removed: {dirs}")


def run_clean_results():
    """Clean obsolete Results folders."""
    logger.info("="*60)
    logger.info("RESULTS FOLDER CLEANUP")
    logger.info("="*60)

    show_current_status()

    removed_dirs = clean_results_folders()
    removed_root = clean_root_files()

    print(f"\nCleanup completed:")
    print(f"  - Results folders removed: {removed_dirs}")
    print(f"  - Root files removed: {removed_root}")


def run_clean_all():
    """Run full cleanup."""
    logger.info("="*60)
    logger.info("FULL CLEANUP")
    logger.info("="*60)

    show_current_status()

    print("\nRunning cleanup...")

    files, cache_dirs = clean_temp_files()
    result_dirs = clean_results_folders()
    root_files = clean_root_files()

    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    print(f"  - Temporary files removed: {files}")
    print(f"  - Cache directories removed: {cache_dirs}")
    print(f"  - Results folders removed: {result_dirs}")
    print(f"  - Root files removed: {root_files}")
    print("="*60)


def run_show_status():
    """Show status only, without cleaning."""
    show_current_status()


# NOTE: This module should be run only via main_c.py
# Example: python main_c.py --clean-all
