# -*- mode: python -*-

block_cipher = None


a = Analysis(['C:\\users\\quinn vinlove\\documents\\sugarsbio\\src\\main\\python\\main.py'],
             pathex=['C:\\users\\quinn vinlove\\documents\\sugarsbio\\target\\PyInstaller'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=['C:\\Users\\Quinn Vinlove\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\fbs\\freeze\\hooks'],
             runtime_hooks=['C:\\users\\quinn vinlove\\documents\\sugarsbio\\target\\PyInstaller\\fbs_pyinstaller_hook.py'],
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
          console=False , icon='C:\\users\\quinn vinlove\\documents\\sugarsbio\\src\\main\\icons\\Icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='ISOviewer')
