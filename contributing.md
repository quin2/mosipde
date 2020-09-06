# Contributing
## Pull requests always welcome!

## Code Layout
The code lives in `stc/main/python` and is divided into two files: `main.py` and `plots.py`. Everything related to generating plots, performing corrections and de-deriv, and loading data is all inside the `plots.py`. `main.py` includes all the user interface logic. Main "talks" to plots by passing around and calling the `ISOviewer` object that is defined inside of Plots. Plots also contains a corrector class, which is called by Main as well, but I'll likely just integrate this class into `ISOViewer` because it only contains a few methods

## Data representation
When an input file is loaded into ISOviewer, it is stored inside the `df` dataframe inside of the `ISOViewer`object. Because it takes some time, in order to load the data, the GUI will also call on `chart_all` which takes the single dataframe and turns it into several other dataframes which are needed for calculations. Try to do this as little as possible, because it does take some time. Then, `generate_all` must be called to generate all the plots from these dataframes. `df` should be the only dataframe that is directly changed or written to, and if that happens, you MUST do a `chart_all` call afterwards if you need to keep extracting recent insights from the data.

## Dependencies 
* PyQT5 for the GUI framework
* Pandas for dataframes
* Scikit-learn for some regression
* Matplotlib for making plots
* fbs for building

## Running Locally
* fbs requires python 3.6.4. You need this first, NOT the latest version of Python.
* to make builds, you will also need to install NSIS, google around for this (it's a seperate application)
* do `pip install -r requirements.txt`
* do `fbs run` in the root directory of this project

## Building to installer form
This project uses the fman build system to generate premade applications for Mac and Windows that don't need python or anything else to run. if you installed the requirements correctly, you should be able to do `fbs freeze` followed by `fbs installer` to make the latest version into in an installer. This process does take a long time, so avoid doing it unless you made changes you want to share. Also, because the result files are so large, you'll have to use Git LFS if you want to commit them to your pull request. Otherwise, if you just want to run locally, untrack all `.exe` and `.dmg` files. Fbs will generate windows installers on windows, and mac installers on mac. For windows, you'll have to do `py -m pip install pypiwin32` to get the build to run right, do everything in an administer-privilaged command prompt, and download NSIS to greate installers. 

## Notes
I don't use the QT5agg backend in Matplotlib because it's SLOW: It takes forever to zoom in on plots, load plots, and resize the window. It doesn't play well with the (very large) plots that we generate. To step around this, I used a SVG backend coupled with a custom SVG viewer object for QT that rasterizes SVGs quickly. The main disvantage to doing this is that changing the plot requires a reload of the object inside, which isn't implicit. But, I figured we're not changing the data enough to warrent adding a realtime backend to this application.

## **Special notes for building on Windows**
Scikit-learn, the library that calculates the regression in some of the plots, has SciPy as a dependency. Because SciPy has some system specific and processor specific extensions that allow for hardware acceleration of some math, it's kind of a pain to build and bundle it. SciPy alone is responsible for causing a lot of pain for me around PyInstaller. You will have to change pyinstaller according to [this link](https://github.com/pyinstaller/pyinstaller/pull/3911), scroll down to ths comment by GlennS. If you get an error that looks like `ModuleNotFoundError: No module named 'scipy.special.cython_special'`,  [this link](https://stackoverflow.com/questions/62581504/why-do-i-have-modulenotfounderror-no-module-named-scipy-special-cython-specia) might also be helpful. Honestly, if you keep running into build issues, it might be a better idea to avoid the headache: Make the changes you want to make to the program, run the program with `fbs run`, commit the changes if you want to share them, and have someone else build the .exe installer on a machine with a non-broken build system. 