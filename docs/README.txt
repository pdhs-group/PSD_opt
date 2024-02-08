In order to work with Sphinx and MyST first install the following packages
    pip install sphinx
    pip install sphinx-rtd-theme
    pip install myst-parser
    pip install "rst-to-myst[sphinx]"

To update the documentation from the Powershell the following commands are required.

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
