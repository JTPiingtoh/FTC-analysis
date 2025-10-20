Used to semi_automate the analysis of femoral trochlear cartilage ultrasound images. Typically measurements are taken using ImageJ/Fiji. To streamline this process, a directory containing .tiff files can be supplied, with a region of interest being highlighted using ImageJ's polygon tool.

Common measurements can then be taken from these images, including thickness, cross sectional area, and echo intensity, which are then automatically saved to CSV file.

```bash
$ python main.py 
```

Here, the way in which the script splits the region of interest is shown, with the medial, intercondyl, and lateral region shown in blue, green and red respectively. 
![Example analysis output](assets/503_ANALYSED.png)
