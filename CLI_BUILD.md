# Building the Wu CLI Standalone Executable

This guide explains how to build a standalone executable of the Wu CLI that can be distributed without requiring Python installation.

## Prerequisites

1. **Python 3.10+** installed
2. **PyInstaller** installed:
   ```bash
   pip install pyinstaller
   ```
3. **All Wu dependencies** installed (for building):
   ```bash
   pip install -e .
   ```

## Building the Executable

### Option 1: Using the Build Script (Recommended)

```bash
python build_cli.py
```

This will:
- Check prerequisites
- Build the executable using `wu.spec`
- Report the location and size of the built executable

### Option 2: Using PyInstaller Directly

```bash
pyinstaller --clean --noconfirm wu.spec
```

The executable will be created at: `dist/wu.exe`

## Testing the Executable

After building, test the executable:

```bash
dist\wu.exe --help
dist\wu.exe --version
dist\wu.exe formats
```

## Distribution

The `dist/wu.exe` file is a standalone executable that includes:
- All Python dependencies
- The complete Wu package
- Native SIMD libraries
- Report templates

**No Python installation required** - users can simply download and run `wu.exe`.

## File Size

The executable is typically 50-100 MB depending on included dependencies. This is normal for PyInstaller executables that bundle Python and all libraries.

## Troubleshooting

### Import Errors

If you see import errors when running the executable:
1. Rebuild with `--clean` flag
2. Check that all required modules are in `hiddenimports` in `wu.spec`
3. Verify the package structure is correct

### Missing DLL Errors

If native libraries are missing:
1. Ensure `wu_simd.dll` exists in `src/wu/native/`
2. Check the `binaries` section in `wu.spec` includes the DLL

### Large File Size

To reduce size:
1. Remove optional dependencies you don't need
2. Use `--exclude-module` for unused packages
3. Consider using `--onefile` vs `--onedir` (current is onefile)

## Usage Examples

Once built, users can use the CLI like any other command-line tool:

```bash
# Analyze a file
wu.exe analyze photo.jpg

# Generate JSON output
wu.exe analyze photo.jpg --json

# Batch process
wu.exe batch *.jpg --output results/

# Generate PDF report
wu.exe report photo.jpg -o report.pdf
```

