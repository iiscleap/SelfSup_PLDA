import subprocess as sp
import argparse
import os
#from path import Path
import sys
import numpy as np

def gen_sdm_wav(ami_corpus_dir,wav_file_path,files):
  total = len(files)
  with open(wav_file_path, "w") as f:
    for i, f_name in enumerate(files):
      file_name = f'{ami_corpus_dir}/{f_name}/audio/{f_name}.Array1-01.wav'
      if not os.path.exists(file_name):
        print(f"WARNING:{file_name} file not present in AMI corpus.Using Array2-02 instead.")
        file_name = f'{ami_corpus_dir}/{f_name}/audio/{f_name}.Array2-02.wav'
      line = f'{f_name}  {file_name}'
      if i != total - 1:
        print(line, file=f)
      else:
        print(line, end='', file=f)


def gen_mdm_wav(ami_corpus_dir,wav_file_path,files):
  total = len(files)
  with open(wav_file_path, "w") as f:
    for i, f_name in enumerate(files):
      file_name = f'{ami_corpus_dir}/beamformed/{f_name}/{f_name}_MDM8.wav'
      if not os.path.exists(file_name):
        print(f"WARNING:{file_name} file not present in AMI corpus.Please check corpus.")
        continue
      line = f'{f_name}  {file_name}'
      if i != total - 1:
        print(line, file=f)
      else:
        print(line, end='', file=f)

def gen_ihm_wav(ami_corpus_dir,wav_file_path,files):
  total = len(files)
  with open(wav_file_path, "w") as f:
    for i, f_name in enumerate(files):
      file_name = f'{ami_corpus_dir}/{f_name}/audio/{f_name}.Mix-Headset.wav'
      if not os.path.exists(file_name):
        print(f"WARNING:{file_name} file not present in AMI corpus.Please check corpus.")
        continue
      line = f'{f_name}  {file_name}'
      if i != total - 1:
        print(line, file=f)
      else:
        print(line, end='', file=f)

def generate_wav_file(ami_corpus_dir, wav_op_dir, dataset):
  
  if not os.path.exists(ami_corpus_dir):
    raise RuntimeError(f'Cannot locate AMI corpus directory at path: {ami_corpus_dir}')
  
  if not os.path.exists(wav_op_dir):
    print(f'WARNING:Cannot locate data directory at path: {wav_op_dir}.Creating data directory')
    cmd=f'mkdir -p {wav_op_dir}'
    os.system(cmd)
  
  if dataset != "sdm" and dataset != "mdm" and dataset != "ihm":
    print("ERROR: Incorrect value for dataset. Must be sdm, mdm or ihm.\nsdm: Single Distant Microphone\nmdm: Multiple Distant Microphone(beamformed)\nihm: Individual Headset Microphone")
    sys.exit(1)
  
  files_path = os.path.join(wav_op_dir,'files')
  if not os.path.exists(files_path):
    raise RuntimeError('Meetings List file not present at {}'.format(files_path))

  wav_file_path = os.path.join(wav_op_dir,'wav.scp')
  if os.path.exists(wav_file_path):
    print(f'WARNING: {wav_file_path} already exists. Overwriting file.')
  
  with open(files_path, 'r') as f:
    files = np.sort(f.read().splitlines())
  
  if dataset == "sdm":
    gen_sdm_wav(ami_corpus_dir,wav_file_path,files)
  if dataset == "mdm":
    gen_mdm_wav(ami_corpus_dir,wav_file_path,files)
  if dataset == "ihm":
    gen_ihm_wav(ami_corpus_dir,wav_file_path,files)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Generate wav.scp file', add_help=True)
  parser.add_argument(
        'ami_corpus_dir', metavar='ami-corpus-dir', type=None,
        help='dir containing amicorpus.All audio files under meeting IDs.Refer README for details.')
  parser.add_argument(
        'wav_op_dir', metavar='wav-op-dir', type=None,
        help='path to data directory where we must create wav.scp file')
  parser.add_argument(
        'dataset', metavar='dataset', type=None,
        help='Type of dataset(sdm/mdm/ihm) for which wav.scp is being created.\
              sdm: Single Distant Microphone\
              mdm: Multiple Distant Microphone(beamformed)\
              ihm: Individual Headset Microphone')
  
  if(len(sys.argv) != 4):
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()

  generate_wav_file(args.ami_corpus_dir, args.wav_op_dir, args.dataset)

