"""
Wu CLI Launcher for PyInstaller.

This launcher works both in development and when frozen by PyInstaller.
"""
import sys
import os

# PyInstaller creates a temporary folder and stores path in _MEIPASS
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    # PyInstaller already handles the path, but we need to ensure wu is importable
    # The wu package should be in the same directory as the executable
    pass
else:
    # Running as script - add src to path
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

try:
    from wu.cli import main
except ImportError as e:
    print(f"Error importing wu.cli: {e}", file=sys.stderr)
    print(f"Python path: {sys.path}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
