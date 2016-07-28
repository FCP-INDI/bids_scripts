#!/usr/bin/env python
'''
John Pellman, Chris Gorgolewski 2015/2016
--------------------------------------------------------------------------------------------
This script will take NifTI data stored locally and tracked in the database and re-organize
it in BIDS format.
Usage: Type -h to see usage.
'''
import os, sys, shutil
import re
from glob import glob
import argparse
from datetime import datetime
import csv
import yaml
import sqlite3
from nibabel.nicom import csareader
import nibabel as nib
import numpy as np
import dicom
import json
from tsv import *
from utils import mkdir_p, setup_logger
import logging

def get_phase_encode(dcm_file):
    ## note x and y are flipped relative to dicom 
    ## to get direction in nifti
    ret_dirs = [("i","i-"),("j","j-"),("k","k-")] 

    dcm_dat = dicom.read_file(dcm_file)
    ## get the direction cosine
    dcm_image_dir = dcm_dat.ImageOrientationPatient
    dcm_image_dir_row = (dcm_image_dir[0],dcm_image_dir[1],dcm_image_dir[2])
    dcm_image_dir_col = (dcm_image_dir[3],dcm_image_dir[4],dcm_image_dir[5])
    ## get the phase encode
    dcm_phase_encode = dcm_dat.InPlanePhaseEncodingDirection
    if dcm_phase_encode == "ROW":
        dcm_vec = dcm_image_dir_row
    else:
        dcm_vec = dcm_image_dir_col
    ## we now have the direction of the phase encode vector
    max_index = np.argmax(np.abs(dcm_vec))
    return ret_dirs[max_index][dcm_vec[max_index] > 0]

def get_slice_direction(dcm_file):
    ret_dirs = [("i","i-"),("j","j-"),("k-","k")] 
    dcm_dat = dicom.read_file(dcm_file)
    csa = csareader.get_csa_header(dcm_dat)
    norm = csareader.get_slice_normal(csa)
    max_index = np.argmax(np.abs(norm))
    return ret_dirs[max_index][norm[max_index] > 0]

def get_effective_echo_spacing(dcm_file):
    dcmobj = dicom.read_file(dcm_file)
    return 1/(dcmobj[("0019", "1028")].value * dcmobj.AcquisitionMatrix[0])

def get_total_readout_time(dcm_file):
    dcmobj = dicom.read_file(dcm_file)
    return 1/(dcmobj[("0019", "1028")].value)

def nifti_sanity_check(nifti_path,bids_path,bids_log):
    # A dict of volume counts to ensure that the NifTI being copied over is complete.
    anat_vol_counts={ 
        'ses-NFB3_T1w':192,
        'T1w':176 }
    func_vol_counts={
        'task-CHECKERBOARD_acq-1400_bold':98,
        'task-CHECKERBOARD_acq-645_bold':240,
        'task-rest_acq-CAP_bold':120,
        'task-rest_acq-1400_bold':404,
        'dwi':137,
        'task-BREATHHOLD_acq-1400_bold':186,
        'task-rest_acq-645_bold':900,
        'task-rest_pcasl':80,
        'task-DMNTRACKINGTEST_bold':412,
        'task-DMNTRACKINGTRAIN_bold':182,
        'mask_bold':3,
        'task-MSIT_bold':198,
        'task-PEER1_bold':54,
        'task-PEER2_bold':54,
        'task-MORALDILEMMA_bold':144}
    fmap_properties={
        'magnitude1':33,
        'phasediff':66}
    img=nib.load(nifti_path)
    # Boolean that denotes whether or not the sanity check succeeded or failed.
    sane=True
    # Check for potential issues and log.
    # Anatomical
    for anat_name in anat_vol_counts.keys():
        if anat_name in bids_path and 'NFB3' not in bids_path and not img.header["dim"][3]==anat_vol_counts[anat_name]:
            sane=False
    # Functional/DWI
    for func_name in func_vol_counts.keys():
        if func_name in bids_path and not img.header["dim"][4]==func_vol_counts[func_name]:
            sane=False
    # Fieldmap
    for fmap_name in fmap_properties.keys():
        if fmap_name in bids_path and not img.header["dim"][3]*img.header["dim"][4]==fmap_properties[fmap_name]:
            sane=False
    return sane

# mcverter adds a tiny amount of rounding error that don't want.
def nifti_copy(nifti_path,dicom_path,bids_path,bids_log):
    sane=nifti_sanity_check(nifti_path,bids_path,bids_log)
    img=nib.load(nifti_path)
    if not sane:
        bids_log.info('Sanity check failed for %s.  Check that this is the NifTI you want.' % bids_path)
    dicom_files=glob(os.path.join(dicom_path,'*.dcm'))
    if dicom_files:
        dicom_file=dicom_files[0]
        dcmobj = dicom.read_file(dicom_file)
        if dcmobj.RepetitionTime:
            img.header["pixdim"][4]=dcmobj.RepetitionTime/1000.0
    nib.save(img,bids_path)

def compare_jsons(json_a,json_b):
    '''
    Name: compare_jsons
    Description: This function will compare two JSONs and return True if they are the same.
    This is necessary because sometimes the lists in the JSON files aren't *exactly* the same
    but rather are off due to rounding errors. It uses numpy's allclose function to perform these comparisons.
    Arguments:
    ================================================================================================
    json_a : dict
        A dictionary representation of one JSON to be compared.
    json_b : dict
        A dictionary representation of other JSON to be compared.
    Returns:
    ================================================================================================
    True if the two JSONs are the same, False otherwise.
    '''
    comp_results=[]
    for key in json_a.keys():
        if key not in json_b.keys():
            return False
        if not isinstance(json_a[key],list) and not isinstance(json_b[key],list):
            comp_results.append(json_a[key]==json_b[key])
        elif isinstance(json_a[key],list) and isinstance(json_b[key],list):
            comp_results.append(np.allclose(json_a[key],json_b[key],atol=0.01))
        else:
            # On the off-chance that one JSON has a list and the other has something entirely different.
            return False
    return (not False in comp_results)

def extract_dicom_task_metadata(dicom_file, output_json, toplevel_json='', extra_fields={}):
    '''
    Name: extract_dicom_task_metadata
    Description: This function will extract metadata from a DICOM header and use
    that to create a BIDS sidecar JSON. If there is a JSON at the root level that would
    be inherited by this task for this particular participant that is identical
    to the data in the DICOM header, this function will not write out a new participant-level
    JSON.
    Arguments:
    ================================================================================================
    dicom_file : string
        A full path to a DICOM file whose header is to be read.
    output_json : string 
        A full path to the JSON that should be written out.
    toplevel_json : string (optional)
        A full path to the JSON that will be inherited from. 
    extra_fields : dictionary (optional)
        Miscellaneous fields to add to the sidecar JSON that may not be contained in the DICOM header.
    '''
    # Load in the JSON that is inherited by default.
    toplevel_json_dict={}
    if os.path.isfile(toplevel_json):
        toplevel_json_dict=json.load(open(toplevel_json,'rU'))
    dcmobj = dicom.read_file(dicom_file)
    json_dict = {}
    if not dcmobj.RepetitionTime and not dcmobj.ProtocolName:
        raise IOError('DICOM does not have Repetition Time or Task Name in header')
    json_dict.update({"RepetitionTime": dcmobj.RepetitionTime/1000.0,
                      "TaskName": dcmobj.ProtocolName,
                      "Manufacturer": dcmobj.Manufacturer,
                      "ManufacturerModelName": dcmobj.ManufacturerModelName,
                      "MagneticFieldStrength": float(dcmobj.MagneticFieldStrength),
                      # Something odd about the data type for these two; saved out with single quotes by json module.
                      "FlipAngle": float(dcmobj[('0018','1314')].value),
                      "PhaseEncodingDirection": get_phase_encode(dicom_file),
                      "EchoTime": dcmobj.EchoTime/1000.0,
                     })
    if dcmobj.ProtocolName != 'Localizer':
        slice_timing = dcmobj[("0019", "1029")].value
        slice_timing = np.around(np.array(slice_timing)/1000.0, 3).tolist()
        json_dict.update({"SliceTiming":slice_timing,
                          "SliceEncodingDirection": get_slice_direction(dicom_file),
                          "EffectiveEchoSpacing": get_effective_echo_spacing(dicom_file),
                         })
        zero_slices_count = (np.array(slice_timing) == 0).sum()
        if zero_slices_count > 1:
            json_dict["MultibandAccelerationFactor"] = zero_slices_count
    json_dict.update(extra_fields)
    # Write out at the individual participant level if the participant has different data or no super JSON was defined.
    if toplevel_json:
        if not compare_jsons(json_dict, toplevel_json_dict):
            json.dump(json_dict, open(output_json, "w"),
                      sort_keys=True, indent=4, separators=(',', ': '))
    else:
        json.dump(json_dict, open(output_json, "w"),
                    sort_keys=True, indent=4, separators=(',', ': '))


def extract_dwi_metadata(dicom_file, output_json, toplevel_json='', extra_fields={}):
    '''
    Name: extract_dwi_metadata
    Description: This function will extract metadata from a DICOM header and use
    that to create a BIDS sidecar JSON. If there is a JSON at the root level that would
    be inherited by this task for this particular participant that is identical
    to the data in the DICOM header, this function will not write out a new participant-level
    JSON.
    Arguments:
    ================================================================================================
    dicom_file : string
        A full path to a DICOM file whose header is to be read.
    output_json : string 
        A full path to the JSON that should be written out.
    toplevel_json : string (optional)
        A full path to the JSON that will be inherited from.
    extra_fields : dictionary (optional)
        Miscellaneous fields to add to the sidecar JSON that may not be contained in the DICOM header.
    '''
    # Load in the JSON that is inherited by default.
    toplevel_json_dict={}
    if os.path.isfile(toplevel_json):
        toplevel_json_dict=json.load(open(toplevel_json,'rU'))
    dcmobj = dicom.read_file(dicom_file)
    json_dict = {}
    json_dict.update({"PhaseEncodingDirection": get_phase_encode(dicom_file),
                      "EffectiveEchoSpacing": get_effective_echo_spacing(dicom_file),
                      "EchoTime": dcmobj.EchoTime/1000.0,
                      "TotalReadoutTime": get_total_readout_time(dicom_file)
                     })
    json_dict.update(extra_fields)
    # Write out at the individual participant level if the participant has different data or no super JSON was defined.
    if toplevel_json:
        if not compare_jsons(json_dict, toplevel_json_dict):
            json.dump(json_dict, open(output_json, "w"),
                      sort_keys=True, indent=4, separators=(',', ': '))
    else:
        json.dump(json_dict, open(output_json, "w"),
                    sort_keys=True, indent=4, separators=(',', ': '))

def extract_fmap_phasediff_metadata(dicom_file_e1, dicom_file_e2, output_json, toplevel_json='', extra_fields={}):
    '''
    Name: extract_fmap_phasediff_metadata
    Description: This function will extract metadata from a DICOM header and use
    that to create a BIDS sidecar JSON. If there is a JSON at the root level that would
    be inherited by this task for this particular participant that is identical
    to the data in the DICOM header, this function will not write out a new participant-level
    JSON.
    Arguments:
    ================================================================================================
    dicom_file_e1 : string
        A full path to the first echo DICOM file whose header is to be read.
    dicom_file_e2 : string
        A full path to the second echo DICOM file whose header is to be read.
    output_json : string 
        A full path to the JSON that should be written out.
    toplevel_json : string (optional)
        A full path to the JSON that will be inherited from. 
    extra_fields : dictionary (optional)
        Miscellaneous fields to add to the sidecar JSON that may not be contained in the DICOM header.
    '''

    # Load in the JSON that is inherited by default.
    toplevel_json_dict={}
    if os.path.isfile(toplevel_json):
        toplevel_json_dict=json.load(open(toplevel_json,'rU'))
    echo1=dicom.read_file(dicom_file_e1)
    echo2=dicom.read_file(dicom_file_e2)
    json_dict = {}
    json_dict.update({"EchoTime1": echo1.EchoTime/1000.0,
                      "EchoTime2": echo2.EchoTime/1000.0,
                     })
    json_dict.update(extra_fields)
    if toplevel_json:
        if not compare_jsons(json_dict, toplevel_json_dict):
            json.dump(json_dict, open(output_json, "w"),
                      sort_keys=True, indent=4, separators=(',', ': '))
    else:
        json.dump(json_dict, open(output_json, "w"),
                    sort_keys=True, indent=4, separators=(',', ': '))

def img_to_bids(db_file,output_dir,bids_log,bids_nifti_sanity_log,series,ursi,study_name,anonmap={}):
    '''
    Name: img_to_bids
    Description: This function will BIDSify data that is in the dicom_stats database.
    Arguments:
    ================================================================================================
    db_file : string
        A full path to the DICOM database.
    output_dir : string
        A full path to the directory where the BIDSified data should be stored.
    bids_log : logger
        A logger for the image to BIDS conversion.
    bids_nifti_sanity_log : logger
        A logger for the image sanity checks.
    series : string
        A full path to the series label configuration YAML (also used by QAP in the scheduler).  Used to determine what type of series designation (anat, func, dwi) a given series name in the database should be given.
    ursi : string
        The URSI of the participant to be BIDSified.
    study_name : string
        The study name (e.g.,'NFB3','DS2').
    anonmap : dict (optional)
        A dictionary mapping URSIs to anonymous IDs.  Used if anonymization is to occur. URSIs are keys, anonymous IDs are values.
    '''
    # Init connection to the database.
    conn = sqlite3.connect(db_file, timeout=60)
    cursor = conn.cursor()
    series_labels=yaml.load(open(series))

    # Query the database to fetch all series for this particular URSI/study.  Iterate through each of the fetched series and BIDSify.
    cursor.execute('select series_name,nifti_path,dicom_path,series_id,series_complete,series_file_count from dicom_stats where ursi=? and study_name=? and nifti_state=?;', (ursi,study_name,1))
    all_series=cursor.fetchall()
    if not all_series:
        bids_log.info('URSI / study not in database or not converted to NifTI: %s, %s' % (ursi,study_name))
        bids_log.info('This row of the URSI csv will not be BIDsified.  Fix and re-run.')
        return

    # Re-map URSI
    if anonmap:
        if ursi not in anonmap.keys():
            bids_log.info('Could not find the A number for URSI %s' % ursi)
            bids_log.info('This URSI will not be BIDsified.  Fix the anon ID to URSI mapping file and re-run.')
            return
        ursi=anonmap[ursi]

    for series in all_series:
        series_name, nifti_path,dicom_path,series_id, series_complete,series_file_count = series
        if not nifti_path:
            bids_log.info('No NifTI file for %s, series %, but NifTI state set to 1.  Check this!' % (ursi,series_name))
            continue
        sub='sub-'+ursi
        ses='ses-'+study_name
        series_type = ''
        suffix = ''
        #TODO Should check somewhere if these keys are in series_labels
        # Probably before we even loop through the database entries.
        if series_name in series_labels['anat_series']:
            series_type='anat'
            suffix='T1w' 
            defaced_path=nifti_path.replace('.nii.gz','_defaced.nii.gz')
            # Don't add anat if it hasn't been defaced.
            if os.path.isfile(defaced_path):
                nifti_path=defaced_path
            else:
                bids_log.info('Defaced anatomical image does not exist for %s.  Skipping...' % ursi)
                continue
        elif series_name in series_labels['bold_series']:
            series_type='func'
            suffix='bold'
        elif series_name in series_labels['asl_series']:
            series_type='func'
            suffix='pcasl'
        elif series_name in series_labels['dwi_series']:
            series_type='dwi'
            suffix='dwi'
        elif series_name in series_labels['fieldmap_series']:
            # Maybe have the series file count numbers to differentiate phasediff and magnitude configurable in the YAML rather than hardcoded.
            series_type='fmap'
            if series_file_count==66:
                suffix='phasediff'
            elif series_file_count==33:
                suffix='magnitude1'
            else:
                bids_log.info('Could not determine series type (phasediff or magnitude) for field map for %s.  Series file count = %d' % (ursi,series_file_count))
        if series_type and suffix:
            if series_type == 'anat':
                bids_output = os.path.join(output_dir,sub,ses,series_type,'_'.join([sub,ses,suffix])+'.nii.gz')
            elif series_type == 'func':
                # Have this configurable as well if possible.
                if 'REST' in series_name and 'asl' not in suffix:
                    task='rest'
                    acq=series_name.split('_')[1]
                    bids_output = os.path.join(output_dir,sub,ses,series_type,'_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq,suffix])+'.nii.gz')
                elif 'REST' in series_name and 'asl' in suffix:
                    task='rest'
                    acq=''
                    bids_output = os.path.join(output_dir,sub,ses,series_type,'_'.join([sub,ses,'task-%s' % task,suffix])+'.nii.gz')
                elif 'BREATH_HOLD' in series_name:
                    task='BREATHHOLD'
                    acq=series_name.split('_')[2]
                    bids_output = os.path.join(output_dir,sub,ses,series_type,'_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq,suffix])+'.nii.gz')
                elif 'CHECKERBOARD' in series_name:
                    task='CHECKERBOARD'
                    acq=series_name.split('_')[1]
                    bids_output = os.path.join(output_dir,sub,ses,series_type,'_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq,suffix])+'.nii.gz')
                elif 'MASK' in series_name:
                    bids_output = os.path.join(output_dir,'derivatives',sub,ses,'mask','_'.join([sub,ses,'mask',suffix])+'.nii.gz')
                    acq=''
                else:
                    task=series_name.replace('_','')
                    acq=''
                    bids_output = os.path.join(output_dir,sub,ses,series_type,'_'.join([sub,ses,'task-%s' % series_name.replace('_',''),suffix])+'.nii.gz')
                if 'asl' in suffix:
                    toplevel_json=os.path.join(output_dir,'_'.join(['task-%s' % series_name.replace('_',''),suffix])+'.json')
                elif 'MASK' in series_name:
                    toplevel_json=os.path.join(output_dir,'_'.join(['mask',suffix])+'.json')
                elif acq:
                    toplevel_json = os.path.join(output_dir,'_'.join(['task-%s' % task,'acq-%s' % acq,suffix])+'.json')
                else:
                    toplevel_json = os.path.join(output_dir,'_'.join(['task-%s' % task,suffix])+'.json')
            elif series_type == 'fmap':
                bids_output = os.path.join(output_dir,sub,ses,series_type,'_'.join([sub,ses,suffix])+'.nii.gz')
                toplevel_json=os.path.join(output_dir,'phasediff.json')
            elif series_type == 'dwi':
                bids_output = os.path.join(output_dir,sub,ses,series_type,'_'.join([sub,ses,suffix])+'.nii.gz')
                toplevel_json=os.path.join(output_dir,'dwi.json')
                bval_source = nifti_path.replace('.nii.gz','_bvals')
                bval_target = bids_output.replace('.nii.gz','.bval')
                bvec_source = nifti_path.replace('.nii.gz','_bvecs')
                bvec_target = bids_output.replace('.nii.gz','.bvec')
                if not os.path.isfile(bval_target):
                    try:
                        mkdir_p(os.path.dirname(bval_target))
                        shutil.copy(bval_source,bval_target)
                        bids_log.info('Successfully copied to %s:' % bval_target)
                    except Exception as e:
                        if not anonmap:
                            bids_log.info('Could not copy %s:  %s' % (series_id,e))
                            bids_log.info('Source Path : %s'  % bval_source)
                        bids_log.info('Could not copy to: %s' % bval_target)
                    else:
                        bids_log.info('Could not copy to : %s (already exists)'  % bvec_target)
                if not os.path.isfile(bvec_target):
                    try:
                        mkdir_p(os.path.dirname(bvec_target))
                        shutil.copy(bvec_source,bvec_target)
                        bids_log.info('Successfully copied to %s:' % bvec_target)
                    except Exception as e:
                        if not anonmap:
                            bids_log.info('Could not copy %s:  %s' % (series_id,e))
                            bids_log.info('Source Path : %s'  % bvec_source)
                        bids_log.info('Could not copy to : %s'  % bvec_target)
                    else:
                        bids_log.info('Could not copy to : %s (already exists)'  % bvec_target)

            json_output = bids_output.replace('.nii.gz','.json')
            mkdir_p(os.path.dirname(bids_output))
            if not os.path.isfile(bids_output):
                try:
                    nifti_copy(nifti_path,dicom_path,bids_output,bids_nifti_sanity_log)
                    bids_log.info('Successfully copied to: %s' % bids_output)
                except Exception as e:
                    if not anonmap:
                        bids_log.info('Could not copy %s: %s' % (series_id,e))
                        bids_log.info('Source Path : %s' % nifti_path)
                    bids_log.info('Could not copy to : %s'  % bids_output)
                    bids_log.info('Could not copy %s: %s' % (bids_output,e))
            else:
                sane_nifti_path=nifti_sanity_check(nifti_path,bids_output,bids_log)
                sane_bids_path=nifti_sanity_check(bids_output,bids_output,bids_log)
                if sane_nifti_path and not sane_bids_path:
                    bids_nifti_sanity_log.info('Replacing incomplete series %s with complete version of series' % bids_output)
                    os.remove(bids_output)
                    try:
                        nifti_copy(nifti_path,dicom_path,bids_output,bids_nifti_sanity_log)
                        if series_type == 'dwi':
                            bval_source = nifti_path.replace('.nii.gz','_bvals')
                            bval_target = bids_output.replace('.nii.gz','.bval')
                            bvec_source = nifti_path.replace('.nii.gz','_bvecs')
                            bvec_target = bids_output.replace('.nii.gz','.bvec')
                            os.remove(bval_target)
                            shutil.copy(bval_source,bval_target)
                            os.remove(bvec_target)
                            shutil.copy(bvec_source,bvec_target)
                        bids_log.info('Successfully copied to: %s' % bids_output)
                    except Exception as e:
                        if not anonmap:
                            bids_log.info('Could not copy %s: %s' % (series_id,e))
                            bids_log.info('Source Path : %s' % nifti_path)
                            bids_nifti_sanity_log.info('Replacement of %s failed: %s' % (bids_output,e))
                        else:
                            bids_nifti_sanity_log.info('Replacement of %s failed' % bids_output)
                        bids_log.info('Could not copy to : %s'  % bids_output)

            # Check JSONs that would be generated at individual-level against top-level JSONs.
            if series_type == 'func' and 'asl' not in suffix:
                dicom_files=glob(os.path.join(dicom_path,'*.dcm'))
                if dicom_files:
                    extract_dicom_task_metadata(dicom_files[0],json_output, toplevel_json)
                else:
                    bids_log.info('Could not find DICOMs to generate functional JSON for %s series %s.' % (ursi,series_name))
            elif series_type == 'dwi':
                dicom_files=glob(os.path.join(dicom_path,'*.dcm'))
                if dicom_files:
                    extract_dwi_metadata(dicom_files[0],json_output, toplevel_json)
                else:
                    bids_log.info('Could not find DICOMs to generate DTI JSON for %s series %s.' % (ursi,series_name))
            elif series_type == 'fmap' and series_file_count==66:
                dicom_files_echo1=glob(os.path.join(dicom_path,'*-1.dcm'))
                dicom_files_echo2=glob(os.path.join(dicom_path,'*-2.dcm'))
                if dicom_files_echo1 and dicom_files_echo2:
                    extract_fmap_phasediff_metadata(dicom_files_echo1[0],dicom_files_echo2[0],json_output, toplevel_json)
                else:
                    bids_log.info('Could not find DICOMs to generate Fieldmap JSON for %s series %s.' % (ursi,series_name))
        else:
            bids_log.info('BIDSification failed for the following series for %s : \'%s\' is not in YAML' % (ursi,series_name))
        if series_complete==0 :
            bids_log.info('Warning for %s series %s : series may be incomplete' % (ursi,series_name))
        elif series_complete==-1 :
            bids_log.info('Warning for %s series %s : series may be missing.' % (ursi,series_name))
            #TODO Add a BIDS state to database and update it here.
    conn.close()

def beh_to_bids(beh_warehouse, output_dir, bids_log, ursi, study_name, anonmap={}):
    '''
    Name: beh_to_bids
    Description: A function to convert all behavioral data to BIDS.
    Arguments:
    -----------------------------------------------------------------------------
    beh_warehouse : string
        The path to the directory containing the anonymous IDs.
    output_dir : string
        The root directory for the BIDs output.
    bids_log : logger
        A logger for the behavioral to BIDS conversion.
    ursi : string
        The URSI of the participant to be BIDSified.
    study_name : string
        The study name (e.g.,'NFB3','DS2').
    anonmap : dict (optional)
        A dictionary mapping URSIs to anonymous IDs.  Used if anonymization is to occur. URSIs are keys, anonymous IDs are values.
    '''

    # Get the path in the beh_warehouse directory.
    ursi_src_path=glob(os.path.join(beh_warehouse,'*','_'.join([ursi,study_name])))
    if ursi_src_path:
        ursi_src_path=ursi_src_path[0]
    else:
        bids_log.info('Could not find behavioral data directory for URSI %s, study/visit %s' % (ursi,study_name))
    if anonmap:
        if ursi not in anonmap.keys():
            bids_log.info('Could not find the A number for URSI %s' % ursi)
            bids_log.info('This URSI will not be BIDsified.  Fix the anon ID to URSI mapping file and re-run.')
            return
        ursi=anonmap[ursi]
    # Determine the form of the output directory
    # Prepend 'sub-'
    sub='sub-'+ursi
    ses='ses-'+study_name
    func_dir=os.path.join(output_dir,sub,ses,'func')
    deriv_dir=os.path.join(output_dir,'derivatives',sub,ses)
    mkdir_p(func_dir)
    mkdir_p(deriv_dir)

    # Produce events TSVs
    if study_name=='NFB3':
        # PEER
        peerone_nifti=glob(os.path.join(func_dir,'*task-PEER1_bold.nii.gz'))
        if peerone_nifti:
            try:
                bids_log.info('Creating PEER1 TSV for %s %s' % (ursi,study_name))
                generate_peer(func_dir,sub,ses,1)
            except Exception as e:
                bids_log.info('Could not create PEER1 TSV:')
                bids_log.info(e)
                bids_log.info('='*50)
        peertwo_nifti=glob(os.path.join(func_dir,'*task-PEER2_bold.nii.gz'))
        if peertwo_nifti:
            try:
                bids_log.info('Creating PEER2 TSV for %s %s' % (ursi,study_name))
                generate_peer(func_dir,sub,ses,2)
            except Exception as e:
                bids_log.info('Could not create PEER2 TSV:')
                bids_log.info(e)
                bids_log.info('='*50)

    if study_name=='NFB3' and ursi_src_path:
        # Keys are series names per the behavioral data warehouse, values are the series names as used by the BIDS warehouse.
        series_dict={'FEEDBACK':'DMNTRACKINGTEST','TRAIN':'DMNTRACKINGTRAIN','MORAL':'MORALDILEMMA','MSIT':'MSIT','PEER1':'PEER1','PEER2':'PEER2'}

        # DMN_TRACKING_TEST
        dmntrackingtask_source=glob(os.path.join(ursi_src_path,ursi+'_FEEDBACK','NFB*_FEEDBACK*_log.txt'))
        #dmntrackingtask_source=glob(os.path.join(ursi_src_path,'FEEDBACK',ursi+'_NFB*_FEEDBACK*_log.txt'))
        if dmntrackingtask_source:
            dmntrackingtask_source=dmntrackingtask_source[0]
            dmntrackingtask_target=os.path.join(func_dir,sub+'_'+ses+'_task-DMNTRACKINGTEST_events.tsv')
            if not os.path.isfile(dmntrackingtask_target):
                try:
                    bids_log.info('Converting NFB tracking log to TSV for %s %s' % (ursi,study_name))
                    nfb_log_to_tsv(dmntrackingtask_source,dmntrackingtask_target)
                except Exception as e:
                    bids_log.info('Could not convert NFB tracking log to TSV:')
                    bids_log.info(ursi_src_path)
                    bids_log.info(e)
                    bids_log.info('='*50)
        # MORAL_DILEMMA
        moraldilemmatask_source=glob(os.path.join(ursi_src_path,ursi+'_MORAL','*.csv'))
        if moraldilemmatask_source:
            moraldilemmatask_source=moraldilemmatask_source[0]
            moraldilemmatask_target=os.path.join(func_dir,sub+'_'+ses+'_task-MORALDILEMMA_events.tsv')
            if not os.path.isfile(moraldilemmatask_target):
                try:
                    bids_log.info('Converting moral dilemma psychopy log to TSV for %s %s' % (ursi,study_name))
                    md_psychopy_to_tsv(moraldilemmatask_source,moraldilemmatask_target)
                except Exception as e:
                    bids_log.info('Could not convert moral dilemma task data to TSV:')
                    bids_log.info(ursi_src_path)
                    bids_log.info(e)
                    bids_log.info('='*50)
        # MSIT
        msittask_source=glob(os.path.join(ursi_src_path,ursi+'_MSIT','*.csv'))
        if msittask_source:
            msittask_source=msittask_source[0]
            msittask_target=os.path.join(func_dir,sub+'_'+ses+'_task-MSIT_events.tsv')
            if not os.path.isfile(msittask_target):
                try:
                    bids_log.info('Converting MSIT psychopy log to TSV for %s %s' % (ursi,study_name))
                    msit_psychopy_to_tsv(msittask_source,msittask_target)
                except Exception as e:
                    bids_log.info('Could not convert MSIT task data to TSV:')
                    bids_log.info(ursi_src_path)
                    bids_log.info(e)
                    bids_log.info('='*50)
    if study_name!='NFB3':
        series_dict={'BREATH_HOLD_1400':'BREATHHOLD','CHECKER_645':'CHECKERBOARD','CHECKER_1400':'CHECKERBOARD','DIFF_137':'dwi','PCASL_REST':'rest','REST_645':'rest','REST_1400':'rest','REST_CAP':'rest'}
        # Produce events TSVs
        for series in series_dict.keys():
            series_nifti=glob(os.path.join(func_dir,'*task-%s_bold.nii.gz' % series_dict[series]))
            if 'BREATH_HOLD' in series:
                acq=series.split('_')[2]
                try:
                    bids_log.info('Creating Breathhold TSV for %s %s' % (ursi,study_name))
                    generate_breathhold(func_dir,sub,ses,acq)
                except Exception as e:
                    bids_log.info('Could not create breathhold TSV:')
                    bids_log.info(ursi_src_path)
                    bids_log.info(e)
                    bids_log.info('='*50)
            elif 'CHECKER' in series:
                acq=series.split('_')[1]
                try:
                    bids_log.info('Creating Checkerboard TSV for %s %s' % (ursi,study_name))
                    generate_checkerboard(func_dir,sub,ses,acq)
                except Exception as e:
                    bids_log.info('Could not create checkerboard TSV:')
                    bids_log.info(ursi_src_path)
                    bids_log.info(e)
                    bids_log.info('='*50)

    # If there is no behavioral data directory, no more can be done.
    if not ursi_src_path:
        return

    # Convert physio data to tsvs.
    for series in series_dict.keys():
        task=series_dict[series]
        source=glob(os.path.join(ursi_src_path,ursi+'_'+series,'%s_%s.txt') % (study_name,series))
        source.extend(glob(os.path.join(ursi_src_path,ursi+'_'+series,'%s_%s_physio.txt' % (study_name,series))))
        if source:
            source=source[0]
            if 'REST' in series and 'PCASL' not in series:
                acq=series.split('_')[1]
                target = os.path.join(func_dir,'_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq])+'_physio.tsv')
            elif 'BREATH_HOLD' in series:
                acq=series.split('_')[2]
                target = os.path.join(func_dir,'_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq])+'_physio.tsv')
            elif 'CHECKER' in series:
                acq=series.split('_')[1]
                target = os.path.join(func_dir,'_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq])+'_physio.tsv')
            elif 'DIFF_137' in series:
                mkdir_p(os.path.join(output_dir,sub,ses,'dwi'))
                target = os.path.join(output_dir,sub,ses,'dwi','_'.join([sub,ses,'dwi'])+'_physio.tsv')
            else:
                target=os.path.join(func_dir,sub+'_'+ses+'_task-%s_physio.tsv' % task)
            if not os.path.isfile(target+'.gz'):
                try:
                    bids_log.info('Converting physio data for %s %s %s' % (ursi,study_name, series))
                    physio_to_tsv(source,target)
                except Exception as e:
                    bids_log.info('Could not convert raw physio to TSV for %:' % source)
                    bids_log.info(ursi_src_path)
                    bids_log.info(e)
                    bids_log.info('='*50)

        despiked_source=glob(os.path.join(ursi_src_path,ursi+'_'+series,'despike_*.txt'))
        if despiked_source:
            despiked_source=despiked_source[0]
            if 'REST' in series and 'PCASL' not in series:
                acq=series.split('_')[1]
                despiked_target = os.path.join(deriv_dir,'func','_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq])+'_recording-despiked_physio.tsv')
            elif 'BREATH_HOLD' in series:
                acq=series.split('_')[2]
                despiked_target = os.path.join(deriv_dir,'func','_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq])+'_recording-despiked_physio.tsv')
            elif 'CHECKER' in series:
                acq=series.split('_')[1]
                despiked_target = os.path.join(deriv_dir,'func','_'.join([sub,ses,'task-%s' % task,'acq-%s' % acq])+'_recording-despiked_physio.tsv')
            elif 'DIFF_137' in series:
                mkdir_p(os.path.join(deriv_dir,'dwi'))
                despiked_target = os.path.join(deriv_dir,'dwi','_'.join([sub,ses,'dwi'])+'_recording-despiked_physio.tsv')
            else:
                despiked_target=os.path.join(deriv_dir,'func',sub+'_'+ses+'_task-%s_recording-despiked_physio.tsv' % task)
            if not os.path.isfile(despiked_target+'.gz'):
                try:
                    bids_log.info('Converting despiked physio data for %s %s %s' % (ursi,study_name, series))
                    physio_to_tsv(despiked_source,despiked_target)
                except Exception as e:
                    bids_log.info('Could not convert despiked physio to TSV:')
                    bids_log.info(ursi_src_path)
                    bids_log.info(e)
                    bids_log.info('='*50)

def anonymize_existing(bids_warehouse,anonmap,bids_log):
    '''
    Name: anonymize_existing
    Description: This function will anonymized BIDSified data in the non-anonymized directory.
    Arguments:
    ================================================================================================
    bids_warehouse : string
        A full path to the BIDS warehouse.
    anonmap : dict (optional)
        A dictionary mapping URSIs to anonymous IDs.  Used if anonymization is to occur. URSIs are keys, anonymous IDs are values.
    bids_log : logger
        A logger for the image to BIDS conversion.
    '''
    nonanon_dir=os.path.join(bids_warehouse,'Non-anonymized')
    anon_dir=os.path.join(bids_warehouse,'Anonymized')
    for nonanon_root, dirnames, filenames in os.walk(nonanon_dir):
        for filename in filenames:
            participants_tsv=False
            nonanon_file = os.path.join(nonanon_root,filename)
            ursi=re.findall('M[0-9]{8}',nonanon_file)
            if ursi:
                ursi=ursi[0]
            elif 'participants.tsv' in nonanon_file:
                participants_tsv=True
                anon_file=os.path.join(anon_dir,'participants.tsv')
            else:
                bids_log.info('Could not find URSI in file %s.  (Probably an inherited JSON)' % nonanon_file)
                continue
            if not participants_tsv:
                if ursi not in anonmap.keys():
                    bids_log.info('URSI %s not in anonymization map.   Skipping...' % ursi)
                    continue
                anon_root = nonanon_root.replace(ursi,anonmap[ursi])
                anon_root = anon_root.replace(nonanon_dir,anon_dir)
                anon_file = nonanon_file.replace(ursi,anonmap[ursi])
                anon_file = anon_file.replace(nonanon_dir,anon_dir)
                mkdir_p(anon_root)
            if not os.path.isfile(anon_file):
                if '.nii.gz' in nonanon_file:
                    try:
                        shutil.copy(nonanon_file,anon_file)
                    except:
                        bids_log.info('Could not copy %s' % nonanon_file)
                else:
                    try:
                        with open(nonanon_file,'rU') as nonanon_f:
                            with open(anon_file,'w') as anon_f:
                                for line in nonanon_f:
                                    ursi=re.findall('M[0-9]{8}',line)
                                    if ursi:
                                        ursi=ursi[0]
                                        if ursi in anonmap.keys():
                                            anon_f.write(line.replace(ursi,anonmap[ursi]))
                                    else:
                                        anon_f.write(line)
                    except:
                        bids_log.info('Could not copy %s' % nonanon_file)
            else:
                bids_log.info('%s is already anonymized' % nonanon_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # Parameters for all functions
    parser.add_argument('-b', '--bids_warehouse', required=True,
                                    help='Path to the BIDS warehouse, where all the BIDSified data will be stored by release/anonymization status.')
    parser.add_argument('-l', '--log_dir', required=True,
                                    help='Path to the logs directory.')
    # Parameters for both img_to_bids and beh_to_bids 
    parser.add_argument('-u', '--ursis', required=False,
                                    help='A full path to a CSV file with the URSI in one column and the study name/visit (e.g., CLG4) in the second column.')
    parser.add_argument('-s', '--series', required=False,
                                    help='A YAML file containing the names of series and what their task names should be changed to (as well as their type in BIDS notation (e.g., anat, func).')
    parser.add_argument('-a', '--anon_ursi', required=False,
                                    help='A two-column CSV file with anonymous IDs in the first column and URSIs in the second column.  Used to anonymize participant labels.  If this is included the script will only generate the anonymized warehouse.  Run again without this flag for the non-anonymized warehouse.')
    # For img_to_bids only
    parser.add_argument('-d', '--db_file', required=False,
                                    help='Path to the database.')
    # For beh_to_bids only
    parser.add_argument('-t', '--beh_warehouse', required=False,
                                    help='Path to the behavioral data warehouse, where all the behavioral data that needs to be BIDSified resides.')
    # For participants.tsv and session tsvs.
    parser.add_argument('-p', '--phenotype_file', required=False,
                                    help='A csv containing phenotypic info for the release. Used to create participants.tsv') 
    # For copying inherited files.
    parser.add_argument('-i', '--inherited', required=False,
                                    help='A path to a directory containing task metadata in JSON format, as well as other files to be included at the top level of the BIDS directory (such as the dataset description).  Some of this data will be inherited by all scans.')
    
    args = parser.parse_args()
    bids_warehouse = os.path.abspath(args.bids_warehouse)
    log_dir = os.path.abspath(args.log_dir)

    # Log errors
    now = datetime.now()
    bids_logs=os.path.join(log_dir,'bids_logs')
    log_path=os.path.join(bids_logs,str(now.year),str(now.month),str(now.day))
    if not os.path.exists(log_path):
        mkdir_p(log_path)
    bids_nifti_sanity_log=setup_logger('bids_nifti_sanity',os.path.join(log_path,'bids_nifti_sanity.log'), logging.INFO)
    
    # Read anonymization key in as dictionary and define output directory.
    anonmap={}
    if args.anon_ursi:
        with open(os.path.abspath(args.anon_ursi),'rU') as map_file:
            csvreader = csv.reader(map_file,delimiter=',')
            for row in csvreader:
                anonmap[row[1]]=row[0]
        output_dir=os.path.join(bids_warehouse,'Anonymized')
    else:
        output_dir=os.path.join(bids_warehouse,'Non-anonymized')
    mkdir_p(output_dir)

    if args.inherited:
        bids_inherit_log=setup_logger('bids_inherit',os.path.join(log_path,'bids_inherit_jsons.log'), logging.INFO)
        inherited =  os.path.abspath(args.inherited)
        inherit_jsons(output_dir,bids_inherit_log,inherited)
        for toplevel in glob(os.path.join(inherited,'*')):
                try:
                    shutil.copy(toplevel, os.path.join(output_dir,os.path.basename(toplevel)))
                except:
                    bids_log.info('Unable to copy %s to %s.' % (toplevel, output_dir))

    # Open the URSI csv, iterate through each row and BIDSify behavioral data, imaging data, or both.
    if args.ursis and args.series and (args.db_file or args.beh_warehouse):
        ursis = os.path.abspath(args.ursis)
        series = os.path.abspath(args.series)
        if args.db_file:
            db_file = os.path.abspath(args.db_file)
        if args.beh_warehouse:
            beh_warehouse = os.path.abspath(args.beh_warehouse)
        bids_csv_log=setup_logger('bids_csv',os.path.join(log_path,'bids_csv.log'), logging.INFO)
        with open(ursis,'rU') as ursi_csv:
            ursi_study = csv.reader(ursi_csv,delimiter=',')
            for row in ursi_study:
                ursi,study_name=row
                if not ursi or not study_name:
                    bids_csv_log.info('URSI or study field blank in URSI csv: %s,%s' % (ursi,study_name))
                    bids_csv_log.info('This row of the URSI csv will not be BIDsified.  Fix and re-run.')
                    continue
                if args.db_file:
                    bids_nifti_log=setup_logger('bids_nifti',os.path.join(log_path,'bids_nifti.log'), logging.INFO)
                    if anonmap:
                        img_to_bids(db_file,output_dir,bids_nifti_log,bids_nifti_sanity_log,series,ursi,study_name,anonmap)
                    else:
                        img_to_bids(db_file,output_dir,bids_nifti_log,bids_nifti_sanity_log,series,ursi,study_name)
                if args.beh_warehouse:
                    bids_beh_log=setup_logger('bids_beh',os.path.join(log_path,'bids_behavioral.log'), logging.INFO)
                    if anonmap:
                        beh_to_bids(beh_warehouse, output_dir, bids_beh_log, ursi, study_name,anonmap)
                    else:
                        beh_to_bids(beh_warehouse, output_dir, bids_beh_log, ursi, study_name)

    # TODO Fix odd bug where anonymization on the fly produces a participants.tsv with no headers.
    if args.phenotype_file and args.ursis:
        bids_phenotype_log=setup_logger('bids_phenotype_tsvs',os.path.join(log_path,'bids_phenotype_tsv.log'), logging.INFO)
        ursis = os.path.abspath(args.ursis)
        phenotype_file = os.path.abspath(args.phenotype_file)
        bids_phenotype_log.info('Generating participants.tsv and session tsvs using %s' % phenotype_file)
        phenotype_dict=gen_phenotype_dict(phenotype_file)
        with open(ursis,'rU') as ursi_csv:
            ursi_study = csv.reader(ursi_csv,delimiter=',')
            for row in ursi_study:
                ursi,study_name=row
                if anonmap:
                    sessions_tsv(output_dir,ursi,study_name,phenotype_dict,anonmap)
                    participants_tsv(output_dir,ursis,phenotype_dict,anonmap)
                else:
                    sessions_tsv(output_dir,ursi,study_name,phenotype_dict)
                    participants_tsv(output_dir,ursi,phenotype_dict)

    if anonmap and not (args.phenotype_file and args.ursis) and not (args.ursis and args.series and (args.db_file or args.beh_warehouse)):
        bids_anon_log=setup_logger('bids_anon',os.path.join(log_path,'bids_anon.log'), logging.INFO)
        anonymize_existing(bids_warehouse,anonmap,bids_anon_log)


