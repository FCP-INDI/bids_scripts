BIDS scripts
============

These are scripts that we use to convert our behavioral and imaging data to BIDS format.  

The function to arrange imaging data into BIDS format relies mostly upon a database with the following information:
 * The name of the series that is being converted to BIDS (which can correspond to one of several types of scans- anatomical, functional, DTI, field map).
 * The DICOM path to that series (for extracting meta information stored in the sidecar JSONs).
 * The NifTI path for that series.

Parts of it will need to be adapted for non-Rockland datasets.  The functions to convert behavioral data will differ substantially from other studies.  
Some elements of it we would certainly like to improve.

It chiefly depends upon:
 * nibabel
 * pydicom
 * numpy
 * sqlite3 (for the database portion; the scripts might also be adapted to use other inputs, such as csvs containing the necessary info)
