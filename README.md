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
```
@inproceedings{Lu2017,
       author = {{Lu}, Zhen and {Elbaz}, Ayman M. and {Hernandez Perez}, Francisco E. and {Roberts}, William L. and {Im}, Hong G.},
        title = "{LES/flamelet study of vortex-flame interaction in a turbulent nonpremixed swirl burner}",
    booktitle = {14th International Conference on Combustion and Energy Utilization},
         year = "2018",
        month = "November",
      address = "Sendai, Japan"
}
```
