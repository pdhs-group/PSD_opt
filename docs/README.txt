In order to work with Sphinx and MyST first install the following packages
    pip install sphinx
    pip install sphinx-rtd-theme
    pip install myst-parser
    pip install "rst-to-myst[sphinx]"

For the first time using, use

    .\make.bat html
    
to generate the documentation in .html data form, these documents are saved in the _build/html directory

If you only change docstrings inside the .py files, also simply use 

    .\make.bat html
    
Otherwise, you need to set up the documentation of new modules from the Powershell with the following commands.

Inside the root directory use 

    sphinx-apidoc -o docs . 

to generate .rst files

If you create a new package (folder) containing multiple modules, 
you need to add a __init__.py this folder, then sphinx will process it

Then use 

    rst2myst convert -R docs/**/*.rst 

to convert all of them into markdown files (and remove .rst via -R).

Use 

    .\make.bat html

to make the html files from inside the docs folder

The index.md file should not change by these steps. 
