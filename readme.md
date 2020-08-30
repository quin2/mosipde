# ISOviewer desktop application
A cross platform tool to visualize ISOdat output files and perform corrections & de-derivitization.

## Features
* Works on MacOS and Windows
* Able to draw scatter and box plots
* Identifies run order trends
* Identifies outliers
* Can perform run order correction and IS correction
* Can perform de-deriv with excel file standard input
* Outputs a neat .xlsx workbook

## Limitations
* Runs slow
* Can't do Nitrogen corrections
* Only works on 64 bit Windows 10 
* Will only run as Administrator on Windows - to do this, right click the icon in the start menu

## How to use
To get started, go to "Releases" on the left. You'll want the .exe if you're on Windows and the .dmg if you're on Mac. After running the installer, you'll be prompted to select the output folder for storing all exported files, along with the de-deriv standards file and the p^-1 file. (examples are provided here if you don't have one). This should only happen after you open it for the first time, although you can change these files in the menu, and will have to if the location of the files change. 

## Troubleshooting
If the program crashes, the number one problem is malformed data. Make sure the column names and data types match the `example_input.xls` file in this repo. Also make sure you are running as administrator. If problems persist, or you get a "Can't execute script 'main'" error, send me an email.

## Contributing
See `contributing.md`
