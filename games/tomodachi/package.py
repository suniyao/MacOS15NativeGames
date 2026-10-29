#!/usr/bin/env python3
"""Package a local native Tomodachi app; game data stays in the external host folder."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from games.tomodachi.tools.common import outside_repository, verified_file


def package(work, output, replace=False, language=None):
    work, output = outside_repository(work,ROOT), outside_repository(output,ROOT)
    if output.suffix != ".app":
        raise ValueError("Output must end in .app")
    info = json.loads((work / "host-info.json").read_text())
    if info["title_id"] != "010051F0207B2000":
        raise ValueError("Wrong game host")
    verified_file(work / "build/tomodachi-native")
    verified_file(work / "build/spirv-cross/spirv-cross",info["spirv_cross_sha256"])
    if output.exists():
        previous_language=output / "Contents/Resources/language.txt"
        if language is None and previous_language.exists(): language=previous_language.read_text().strip()
        if not replace or output.is_symlink() or not output.is_dir():
            raise ValueError("App already exists; use --replace to archive the previous build")
        backup = Path(tempfile.mkdtemp(prefix="app-",dir=work / "tests")) / (output.name + ".backup")
        shutil.move(output,backup)
    macos, resources = output / "Contents/MacOS", output / "Contents/Resources"
    macos.mkdir(parents=True)
    resources.mkdir()
    shutil.copy2(work / "build/tomodachi-native",macos / "Tomodachi")
    (resources / "host-path.txt").write_text(os.path.relpath(work,resources) + "\n")
    (resources / "romfs-path.txt").write_text(os.path.relpath(Path(info["prepared"]) / "romfs",resources) + "\n")
    if language:
        (resources / "language.txt").write_text(language + "\n")
    metadata = dict(CFBundleExecutable="Tomodachi",CFBundleIdentifier="org.macnativegames.tomodachi",
                    CFBundleName="Tomodachi Life",CFBundleDisplayName="Tomodachi Life — Experimental",
                    CFBundlePackageType="APPL",CFBundleShortVersionString="1.0.0",CFBundleVersion="1",
                    LSMinimumSystemVersion="15.0",NSHighResolutionCapable=True,LSSupportsGameMode=True,
                    NSBluetoothAlwaysUsageDescription="Connect a wireless game controller.",
                    NSBluetoothPeripheralUsageDescription="Connect a wireless game controller.")
    with (output / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump(metadata,stream)
    subprocess.run(["codesign","--force","--sign","-",str(output)],check=True)
    subprocess.run(["codesign","--verify","--strict",str(output)],check=True)
    print(output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--language",choices=("ja","en-US","en-GB","fr","fr-CA","de","es","es-419","it","nl","pt","pt-BR","ru","ko","zh-Hans","zh-Hant"))
    parser.add_argument("--replace",action="store_true",help="Archive the previous app before packaging")
    args=parser.parse_args()
    try:package(args.work,args.output,args.replace,args.language)
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as error:
        parser.exit(2,f"Tomodachi packaging failed: {error}\n")
