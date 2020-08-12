# -*- mode: python -*-

block_cipher = None


a = Analysis(['/Users/quinnvinlove/Documents/sugarsBio/src/main/python/main.py'],
             pathex=['/Users/quinnvinlove/Documents/sugarsBio/target/PyInstaller'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=['/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/fbs/freeze/hooks'],
             runtime_hooks=['/Users/quinnvinlove/Documents/sugarsBio/target/PyInstaller/fbs_pyinstaller_hook.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='ISOviewer',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=False , icon='/Users/quinnvinlove/Documents/sugarsBio/target/Icon.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='ISOviewer')
app = BUNDLE(coll,
             name='ISOviewer.app',
             icon='/Users/quinnvinlove/Documents/sugarsBio/target/Icon.icns',
             bundle_identifier=None)
