## Neutron Standards Database file

The database file `data2017.gma` was used in the
[STD2017 evaluation][std2017-paper]. The file
`data2017.json` contains the same information but
stored in JSON format. The resulting evaluations
are stored in the `std2017/` directory.

The database file `data.json` contains compared
to `data2017.json` the following updates:

-  Updated uncertainty quantification of Pu-239(n,f) data using templates by [Neudecker2020].
-  Included U-238(n,f) and Pu-239(n,f) measurements relative to U-235(n,f) from the NIFFTE collaboration, see [Neudecker2021],
-  Revised SACS data and included ratio of SACS measurements [Capote2023]


[std2017-paper]: https://www.sciencedirect.com/science/article/pii/S0090375218300218
[Neudecker2020]: https://www.sciencedirect.com/science/article/abs/pii/S0090375219300729
[Neudecker2021]: https://www.osti.gov/biblio/1788383
[Capote2023]: https://www.epj-conferences.org/articles/epjconf/abs/2023/07/epjconf_cw2023_00027/epjconf_cw2023_00027.html
