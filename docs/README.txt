In order to work with Sphinx and MyST first install the following packages
    pip install sphinx
    pip install sphinx-rtd-theme
    pip install myst-parser
    pip install "rst-to-myst[sphinx]"

If you only change docstrings inside the .py files simply skip to the final .\make.bat html. Otherwise, you need to set up the documentation of new modules from the Powershell with the following commands.

Inside the root directory (where pop.py is) use 

    sphinx-apidoc -o docs . 

to generate .rst files

Then use 

    rst2myst convert -R docs/**/*.rst 

to convert all of them into markdown files (and remove .rst via -R).

Use 

    .\make.bat html

to make the html files from inside the docs folder

The index.md file should not change by these steps. 
