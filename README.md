# flamelet2table
python scripts to generate table from flamelet solutions

FM2csv.py:  convert FlameMaster solutions to csv files. The file "speciestranslated" is required in the folder of FlameMaster solutions, which is generated when converting CK format to FM format. 

flameletTableSingleParameter.py:  integrate the flamelet solutions to obtain a table, output to a hdf5 file

Beta distribution is assumed for the independent variable in flamelet solution, e.g., mixture fraction and progress variable for nonpremixed and premixed flames respectively. For the flamelet parameter, such as the stoichiometric scalar dissipation rate and reaction progress parameter, delta and beta distributions are available.

To get the options and help information:
```
python FM2csv -h
```
```
python flameletTableSingleParameter.py -h
```

reference:
@INPROCEEDINGS{Lu2017,
       author = {{Lu}, Zhen and {Elbaz}, Ayman M. and {Hernandez Perez}, Francisco E. and {Roberts}, William L. and {Im}, Hong G.},
        title = "{Large Eddy Simulations of the Vortex-Flame Interaction in a Turbulent Swirl Burner}",
    booktitle = {APS Division of Fluid Dynamics Meeting Abstracts},
         year = 2017,
       series = {APS Meeting Abstracts},
        month = nov,
          eid = {Q2.005}
}
