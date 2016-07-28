import os
from datetime import datetime
import json
import pandas
import subprocess
import numpy as np

def md_psychopy_to_tsv(source,target):
    '''
    Name: md_psychopy_to_tsv
    Description: A function to convert Moral Dilemma task PsychoPy output csvs to 
    BIDS style TSVs.
    Arguments:
    -----------------------------------------------------------------------------
    source : string
        The path to the CSV to be converted to a TSV.
    target : string
        The path to where the TSV should be stored.
    '''
    # Constants related to experimental paradigm and a variable to track the onset
    isi=5
    # Correctness has been double-checked for correctness using:
    # https://github.com/OpenCogLabRepository/moral-dillema/blob/master/PNAS_2008_Harrison_et_al.pdf
    correctness={'c1':'Yes',
            'c2':'No',
            'c3':'Yes',
            'c4':'No',
            'c5':'No',
            'c6':'Yes',
            'c7':'No',
            'c8':'No',
            'c9':'Yes',
            'c10':'No',
            'c11':'Yes',
            'c12':'Yes',
            'c13':'No',
            'c14':'No',
            'c15':'Yes',
            'c16':'Yes',
            'c17':'No',
            'c18':'Yes',
            'c19':'Yes',
            'c20':'No',
            'c21':'Yes',
            'c22':'No',
            'c23':'Yes',
            'c24':'No'}
    onset=0

    # Lists to make dataframe to store the TSV
    tsv_header=['onset','duration','block','response','response_correct','response_time','stimulus_image','stimulus_sound']
    tsv_list=[]

    # Load in Psychopy CSV and check to make sure that all fields are present.
    psychopy_data=pandas.read_csv(source)
    fields=['dilemma_image','dilemma_sound','control_image','control_sound','key_resp_3.keys','key_resp_3.rt','key_resp_4.keys','key_resp_4.rt']
    for field in fields:
        if field not in psychopy_data.columns:
            raise KeyError("Field %s is not present in %s. TSV cannot be generated." % (field, source))

    for row in psychopy_data.iterrows():
        if row[1]['key_resp_3.keys'] == 'None':
            row[1]['key_resp_3.keys'] = '-1'
        if row[1]['key_resp_4.keys'] == 'None':
            row[1]['key_resp_4.keys'] = '-1'
        if not np.isnan(float(row[1]['key_resp_3.keys'])):
            block='Control'
            response=row[1]['key_resp_3.keys']
            response_time=row[1]['key_resp_3.rt']
            stimulus_image=os.path.basename(row[1]['control_image'])
            stimulus_sound=os.path.basename(row[1]['control_sound'])
        elif not np.isnan(float(row[1]['key_resp_4.keys'])):
            block='Moral Dilemma'
            response=row[1]['key_resp_4.keys']
            response_time=row[1]['key_resp_4.rt']
            stimulus_image=os.path.basename(row[1]['dilemma_image'])
            stimulus_sound=os.path.basename(row[1]['dilemma_sound'])
            response_correct='n/a'
        else:
            onset+=isi
            continue
#        print response
#        print type(response)
        if response == '1' or str(response) == '1.0':
            response = 'Yes'
        elif response == '2' or str(response) == '2.0':
            response = 'No'
        elif response == '-1':
            response = 'None'
        if block == 'Control':
            response_correct = response==correctness[stimulus_image.lower().replace('.jpg','')]
        tsv_list.append([onset,isi,block,response,response_correct,response_time,stimulus_image,stimulus_sound])
        onset+=isi 
    tsv_df=pandas.DataFrame(tsv_list,columns=tsv_header)
    tsv_df.to_csv(target,sep='\t',na_rep='n/a',header=True,index=False)
    return tsv_df

def msit_psychopy_to_tsv(source,target):
    '''
    Name: msit_psychopy_to_tsv
    Description: A function to convert MSIT task PsychoPy output csvs to 
    BIDS style TSVs.
    Arguments:
    -----------------------------------------------------------------------------
    source : string
        The path to the CSV to be converted to a TSV.
    target : string
        The path to where the TSV should be stored.
    '''
    # Constants related to experimental paradigm and a variable to track the onset
    isi=1.75
    fixation_buffer=30.0
    all_control_stim=['100','020','003','20','3']
    all_int_stim=['221','212','331','313','112','211','332','233','131','311', '232','322']
    onset=0

    # Lists to make dataframe to store the TSV
    tsv_header=['onset','duration','block','response','response_correct','response_time','stimulus']
    tsv_list=[]

    # Load in Psychopy CSV and check to make sure that all fields are present.
    psychopy_data=pandas.read_csv(source,dtype=str)
    fields=['first_text_str','first_resp.keys','first_resp.corr','first_resp.rt','second_stim_str','second_response.keys','second_response.corr', 'second_response.rt']
    alt_fields=['ctrl_stim','ctrl_correct','control_resp.corr','control_resp.rt','target_stim','stim_response.keys','stim_response.corr', 'stim_response.rt']
    trans=dict(zip(alt_fields,fields))
    if alt_fields[0] in psychopy_data.columns:
            psychopy_data=psychopy_data.rename(columns=trans)
    for field in fields:
        if field not in psychopy_data.columns:
            raise KeyError("Field %s is not present in %s. TSV cannot be generated." % (field, source))

    # Check the first loop text str- if it is a control stimulus, the first loop is control, else first loop is interference.
    zeropad=False
    if psychopy_data['first_text_str'][0] in all_control_stim:
        first_loop='Control'
        second_loop='Interference'
    elif psychopy_data['first_text_str'][0] in all_int_stim:
        first_loop='Interference'
        second_loop='Control'
    for row in psychopy_data.iterrows():
        if np.isnan(float(row[1]['first_text_str'])) and np.isnan(float(row[1]['second_stim_str'])):
            onset+=fixation_buffer
            continue
        elif not np.isnan(float(row[1]['first_text_str'])):
            block=first_loop
            response=row[1]['first_resp.keys']
            if response == 'None':
                response='n/a'
            response_correct=row[1]['first_resp.corr']
            response_time=row[1]['first_resp.rt']
            stimulus=row[1]['first_text_str']
            if stimulus == '20':
                stimulus = '020'
            if stimulus == '3':
                stimulus = '003'
            tsv_list.append([onset,isi,block,response,response_correct,response_time,stimulus])
            onset+=isi
        elif not np.isnan(float(row[1]['second_stim_str'])):
            block=second_loop
            response=row[1]['second_response.keys']
            if response == 'None':
                response='n/a'
            response_correct=row[1]['second_response.corr']
            response_time=row[1]['second_response.rt']
            stimulus=row[1]['second_stim_str']
            tsv_list.append([onset,isi,block,response,response_correct,response_time,stimulus])
            onset+=isi

    tsv_df=pandas.DataFrame(tsv_list,columns=tsv_header)
    tsv_df.to_csv(target,sep='\t',na_rep='n/a',header=True,index=False)
    return tsv_df

def nfb_log_to_tsv(source,target):
    '''
    Name: nfb_log_to_tsv
    Description: A function to convert neurofeedback output logs to 
    BIDS style TSVs.
    Arguments:
    -----------------------------------------------------------------------------
    source : string
        The path to the CSV to be converted to a TSV.
    target : string
        The path to where the TSV should be stored.
    '''
    # Constants related to experimental paradigm and a variable to track the onset
    onset=0

    # Lists to make dataframe to store the TSV
    tsv_header=['onset','duration','left_text','right_text','instruction','feedback','classifier','needle_position']
    tsv_list=[]

    # Load in Psychopy CSV and check to make sure that all fields are present.
    log_data=pandas.read_csv(source, delimiter=';',skiprows=8)
    fields=['Time Stamp', ' STIM', ' Left Text', ' Right Text', ' Stim Text',' Show', ' Sign', ' Classifier Output', ' Detrended Output',' Cumulative Score']
    for field in fields:
        if field not in log_data.columns:
            raise KeyError("Field %s is not present in %s. TSV cannot be generated." % (field, source))
    stimstart=False
    for row in log_data.iterrows():
        if row[1]['Time Stamp'] == 'Time Stamp':
            continue
        if row[1][' STIM'] == ' TR' and stimstart:
            stimstart=False
            tsv_list[-1][1]=float(row[1]['Time Stamp'])-tsv_list[-1][0]
        elif row[1][' STIM'] == ' STIM':
            stimstart=True
            onset=float(row[1]['Time Stamp'])
            duration=-1
            left_text=row[1][' Left Text']
            right_text=row[1][' Right Text']
            instruction=row[1][' Stim Text']
            if abs(row[1][' Show']) == 1:
                feedback='On'
            elif abs(row[1][' Show']) == 9999:
                feedback='Off'
            else:
                feedback='n/a'
            classifier=row[1][' Classifier Output']
            needle_position=row[1][' Cumulative Score']
            tsv_list.append([onset,duration,left_text,right_text,instruction,feedback,classifier,needle_position])
    tsv_df=pandas.DataFrame(tsv_list,columns=tsv_header)
    tsv_df.to_csv(target,sep='\t',na_rep='n/a',header=True,index=False)
    return tsv_df

def gen_phenotype_dict(phenotype_file):
    phenotype_dict={}
    phenotype_df=pandas.read_csv(phenotype_file)
    visits=['NFB2','NFB2R','NFB3','NFBA','NFBAR','DS2','DSA','CLG2','CLG2R','CLG4','CLG4R','CLG5','CLGA']
    for row in phenotype_df.iterrows():
        row=row[1]
        ursi=row['URSI']
        if isinstance(row['Consent Date'],str):
            consent_date=datetime.strptime(row['Consent Date'],'%m/%d/%y')
        else:
            consent_date=''
        consent_age=row['Consent Age']
        self_reported_age=row['Self-reported Age']
        sex=row['Sex']
        handedness=row['Handedness']
        if ursi not in phenotype_dict.keys():
            phenotype_dict[ursi]={}
        new_consent_date=False
        old_consent_date=''
        # Prefer the earliest consent date.
        if 'consent_date' not in phenotype_dict[ursi].keys():
            phenotype_dict[ursi]['consent_date']=consent_date
        elif consent_date:
            if consent_date < phenotype_dict[ursi]['consent_date']:
                phenotype_dict[ursi]['consent_date']=consent_date
                new_consent_date=True
        if 'age' not in phenotype_dict[ursi].keys() or new_consent_date:
            # Prefer self-reported age.
            if not np.isnan(self_reported_age):
                phenotype_dict[ursi]['age']=self_reported_age
            elif not np.isnan(consent_age):
                phenotype_dict[ursi]['age']=consent_age
            else:
                phenotype_dict[ursi]['age']='n/a'
        if 'sex' not in phenotype_dict[ursi].keys() and isinstance(sex,str):
            phenotype_dict[ursi]['sex']=sex.upper()
        if 'handedness' not in phenotype_dict[ursi].keys() and isinstance(handedness,str):
            phenotype_dict[ursi]['handedness']=handedness.upper()
        # Store visit dates
        for visit in visits:
            if visit not in phenotype_dict[ursi].keys() and isinstance(row[visit+' Visit Date'], str):
                phenotype_dict[ursi][visit]=datetime.strptime(row[visit+' Visit Date'],'%m/%d/%y')
    return phenotype_dict

def sessions_tsv(output_dir,ursi,study_name,phenotype_dict,anonmap={}):
    # session_id, age at session, time since T0 (in minutes)
    # Create dict that contains row data.
    row_data={}
    row_data['session_id']=['ses-'+study_name]
    row_data['age']=[np.nan]
    row_data['time_since_enrollment']=[np.nan]
    if ursi in phenotype_dict.keys():
        if study_name in phenotype_dict[ursi].keys() and 'consent_date' in phenotype_dict[ursi].keys():
            row_data['time_since_enrollment']=phenotype_dict[ursi][study_name]-phenotype_dict[ursi]['consent_date']
            row_data['time_since_enrollment']=[row_data['time_since_enrollment'].total_seconds()/60]
        if not np.isnan(row_data['time_since_enrollment']) and 'age' in phenotype_dict[ursi].keys():
            row_data['age']=[phenotype_dict[ursi]['age']+row_data['time_since_enrollment'][0]/(60*24*365.25)]
    if anonmap:
        if ursi not in anonmap.keys():
            return False
        ursi=anonmap[ursi]
    target=os.path.join(output_dir,'sub-'+ursi,'sub-'+ursi+'_sessions.tsv')
    if not os.path.exists(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    if os.path.isfile(target):
        # Open file and append
        target_df=pandas.read_csv(target,sep='\t',na_values=['n/a'])
        row_df=pandas.DataFrame(row_data)
        target_df=pandas.concat([target_df,row_df])
        # Make sure that there are no duplicate entries.
        target_df.drop_duplicates(inplace=True)
    else:
        target_df=pandas.DataFrame(row_data)
    # Create file
    cols = target_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("session_id")))
    cols.insert(1, cols.pop(cols.index("age")))
    cols.insert(2, cols.pop(cols.index("time_since_enrollment")))
    target_df = target_df[cols]
    target_df.to_csv(target, sep="\t", na_rep="n/a", index=False)
    return target_df

def participants_tsv(output_dir,ursi,phenotype_dict,anonmap={}):
    '''
    Name: participants_tsv
    Description: A function to convert phenotype files to a
    BIDS style participants.tsv file / add URSIs to a participants.tsv file.
    Arguments:
    -----------------------------------------------------------------------------
    output_dir : string
        A full path to the directory where the BIDSified data should be stored.
    ursis : string
        The URSI to add to the participants.tsv file.
    phenotype_dict : dict
        All of the phenotype data loaded in as a dictionary that can be queried. See structure/layering of dict under gen_phenotype_dict.
    anonmap : dict (optional)
        A dictionary mapping URSIs to anonymous IDs.  Used if anonymization is to occur. URSIs are keys, anonymous IDs are values.
    '''
    # participant ID, age at start, handedness, sex
    target=os.path.join(output_dir,'participants.tsv')
    if not os.path.exists(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    # Create dict that contains row data.
    row_data={}
    if anonmap:
        if ursi not in anonmap.keys():
            return False
        row_data['participant_id']=[anonmap[ursi]]
    else:
        row_data['participant_id']=[ursi]
    row_data['age']=[np.nan]
    row_data['handedness']=['']
    row_data['sex']=['']
    if ursi in phenotype_dict.keys():
        if 'age' in phenotype_dict[ursi].keys():
            row_data['age']=[phenotype_dict[ursi]['age']]
        if 'handedness' in phenotype_dict[ursi].keys():
            row_data['handedness']=[phenotype_dict[ursi]['handedness']]
        if 'sex' in phenotype_dict[ursi].keys():
            row_data['sex']=[phenotype_dict[ursi]['sex']]
    if os.path.isfile(target):
        # Open file and append
        target_df=pandas.read_csv(target,sep='\t',na_values=['n/a'])
        row_df=pandas.DataFrame(row_data)
        target_df=pandas.concat([target_df,row_df])
        # Make sure that there are no duplicate entries.
        target_df.drop_duplicates(inplace=True)
    else:
        target_df=pandas.DataFrame(row_data)
    cols = target_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("participant_id")))
    cols.insert(1, cols.pop(cols.index("age")))
    cols.insert(2, cols.pop(cols.index("sex")))
    cols.insert(3, cols.pop(cols.index("handedness")))
    target_df = target_df[cols]
    target_df.to_csv(target,sep='\t',na_rep='n/a',header=True,index=False)
    return target_df

#TODO Rewrite to act on .acqs directly.
def physio_to_tsv(source,target):
    '''
    Name: physio_to_tsv
    Description: A function to convert physiological data to
    BIDS style gzipped TSVs and JSONs.
    Arguments:
    -----------------------------------------------------------------------------
    source : string
        The path to the CSV to be converted to a TSV.
    target : string
        The path to where the TSV should be stored.  Filename should be specified
        assuming that the TSV is not gzipped (i.e., 'task.tsv' rather than
        'task.tsv.gz'.
    '''

    # Create a dictionary to store metadata
    # Determine start time, sampling frequency, number of lines to skip when loading in with pandas.
    metadata_dict =  {}
    start_time=0
    frequency=0
    numchannels=0
    columns=[]
    units=[]
    unitline=False
    endmetadata=False
    skiprows=0
    with open(source,'rb') as src_file:
        for line in src_file:
            if 'CH' in line:
                endmetadata=True
            for column in columns:
                if column in line:
                    endmetadata=True
            if endmetadata:
                skiprows+=2
                break
            elif 'msec/sample' in line:
                frequency=line.strip().strip('msec/sample').strip()
                frequency=1000.0/float(frequency)
            elif 'channels' in line:
                numchannels=int(line.strip().strip('channels').strip())
            elif unitline:
                units.append(line.strip())
                unitline=False
            else:
                columns.append(line.strip())
                unitline=True
            skiprows+=1
    # Re-label columns.
    for idx, column in enumerate(columns):
        if 'RSP' in column:
            columns[idx]='respiratory'
        elif 'PPG' in column:
            columns[idx]='cardiac'
        elif 'Digital' in column:
            columns[idx]='trigger'
        elif 'EDA' in column:
            columns[idx]='gsr'
    metadata_dict["SamplingFrequency"]=frequency
    metadata_dict["StartTime"]=start_time
    metadata_dict["Columns"]=columns
    json.dump(metadata_dict, open(target.replace('.tsv','.json'),"w"))
    # Load in lines afterwards as a pandas table.
    physio_tab = pandas.read_table(source,skiprows=skiprows, header=None)
    physio_tab = physio_tab[range(0,numchannels)]
    # Save the table to a headerless tsv and gzip.
    physio_tab.to_csv(target,sep='\t',na_rep='n/a',header=False,index=False)
    p = subprocess.Popen(['gzip',target], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = p.communicate()
    if err:
        raise Exception(err)
    return physio_tab

def generate_events_file(durations, labels, out_file):
    onsets = []
    cur_time = 0
    for i in range(len(durations)):
        onsets.append(cur_time)
        cur_time+=durations[i]
    df = pandas.DataFrame({"onset": onsets, 
                        "duration": durations, 
                        "trial_type": labels})
    df.sort(columns=["onset"], inplace=True)
    # put onset, duration and trial_type in front
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("onset")))
    cols.insert(1, cols.pop(cols.index("duration")))
    cols.insert(2, cols.pop(cols.index("trial_type")))
    df = df[cols]
    # Only save out the JSON if there is a corresponding NifTI
    if os.path.isfile(out_file.replace('.json','.nii.gz')):
        df.to_csv(out_file, sep="\t", na_rep="n/a", index=False)

def generate_breathhold(output_dir,sub,ses,acq):
    TASK_SEQUENCE = ['R', 'G', 'In', 'Out', 'Deep', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
                     'R', 'G', 'In', 'Out', 'Deep', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
                     'R', 'G', 'In', 'Out', 'Deep', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
                     'R', 'G', 'In', 'Out', 'Deep', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
                     'R', 'G', 'In', 'Out', 'Deep', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
                     'R', 'G', 'In', 'Out', 'Deep', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
                     'R', 'G', 'In', 'Out', 'Deep', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6']
    translate = dict(R="Rest",
                    G="Get Ready",
                    In="Breathe In",
                    Out="Breathe Out",
                    Deep="Deep Breath and Hold",
                    H1="Hold1",
                    H2="Hold2",
                    H3="Hold3",
                    H4="Hold4",
                    H5="Hold5",
                    H6="Hold6",
                    )
    TASK_SEQUENCE = [translate[i] for i in TASK_SEQUENCE]

    durations =     [10, 2,  2,   2,    2,  3,  3,  3,  3,  3,  3,
                     10, 2,  2,   2,    2,  3,  3,  3,  3,  3,  3,
                     10, 2,  2,   2,    2,  3,  3,  3,  3,  3,  3,
                     10, 2,  2,   2,    2,  3,  3,  3,  3,  3,  3,
                     10, 2,  2,   2,    2,  3,  3,  3,  3,  3,  3,
                     10, 2,  2,   2,    2,  3,  3,  3,  3,  3,  3,
                     10, 2,  2,   2,    2,  3,  3,  3,  3,  3,  3 ]

    out_file = '_'.join([sub,ses,'task-BREATHHOLD','acq-%s' % acq])+'_events.tsv'
    generate_events_file(durations, TASK_SEQUENCE, os.path.join(output_dir, out_file))

def generate_checkerboard(output_dir,sub,ses,acq):
    block_types     = [ "FIXATION", "CHECKER", "FIXATION", "CHECKER", "FIXATION", "CHECKER", "FIXATION" ] 
    durations = [ 20.0,     20.0,    20.0,     20.0,    20.0,     20.0, 20.0 ]
    out_file = '_'.join([sub,ses,'task-CHECKERBOARD','acq-%s' % acq])+'_events.tsv'
    generate_events_file(durations, block_types, os.path.join(output_dir, out_file))

def generate_peer(output_dir,sub,ses,num):
    nsamples=27
    sampleDuration=4
    onsets = range(0,(sampleDuration*nsamples-1)+1,sampleDuration)
    durations = [sampleDuration]*nsamples
    # Hardcoded for now
    vertical_coordinates = [384.0,
                            691.20000000000005,
                            76.799999999999997,
                            629.75999999999999,
                            629.75999999999999,
                            691.20000000000005,
                            476.16000000000003,
                            599.03999999999996,
                            138.24000000000001,
                            691.20000000000005,
                            76.799999999999997,
                            76.799999999999997,
                            168.96000000000001,
                            76.799999999999997,
                            691.20000000000005,
                            599.03999999999996,
                            291.83999999999997,
                            76.799999999999997,
                            384.0,
                            291.83999999999997,
                            691.20000000000005,
                            138.24000000000001,
                            168.96000000000001,
                            384.0,
                            384.0,
                            384.0,
                            476.16000000000003]
    horizontal_coordinates = [512.0,
                            880.63999999999999,
                            512.0,
                            51.200000000000003,
                            972.79999999999995,
                            51.200000000000003,
                            189.44,
                            373.75999999999999,
                            51.200000000000003,
                            143.36000000000001,
                            143.36000000000001,
                            972.79999999999995,
                            373.75999999999999,
                            51.200000000000003,
                            972.79999999999995,
                            650.24000000000001,
                            834.55999999999995,
                            880.63999999999999,
                            512.0,
                            189.44,
                            512.0,
                            972.79999999999995,
                            650.24000000000001,
                            51.200000000000003,
                            972.79999999999995,
                            512.0,
                            834.55999999999995]
    df = pandas.DataFrame({"onset": onsets, 
                        "duration": durations, 
                        "horizontal_coordinate": horizontal_coordinates,
                        "vertical_coordinate": vertical_coordinates}) 
    df.sort(columns=["onset"], inplace=True)
    # put onset, duration and trial_type in front
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("onset")))
    cols.insert(1, cols.pop(cols.index("duration")))
    cols.insert(2, cols.pop(cols.index("vertical_coordinate")))
    cols.insert(3, cols.pop(cols.index("horizontal_coordinate")))
    df = df[cols]
    out_file = '_'.join([sub,ses,'task-PEER%d' % num])+'_events.tsv'
    # Only save out the JSON if there is a corresponding NifTI
    if os.path.isfile(os.path.join(output_dir,out_file.replace('.json','.nii.gz'))):
        df.to_csv(os.path.join(output_dir,out_file), sep="\t", na_rep="n/a", index=False)
