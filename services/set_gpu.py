import subprocess

def cuda_gpu_available():
    try:
        gpu_available_info = subprocess.Popen('/state/partition1/softwares/Kaldi_Sept_2020/kaldi/src/nnet3bin/cuda-gpu-available', stderr=subprocess.PIPE)
        gpu_available_info = gpu_available_info.stderr.read().decode('utf-8')
        Node = gpu_available_info[86:97]
        gpu = int(gpu_available_info[gpu_available_info.index('active GPU is') + 15])
    except:
        raise Exception("Error running /state/partition1/softwares/Kaldi_Sept_2020/kaldi/src/nnet3bin/cuda-gpu-available ... Did you submit the job through a GPU queue?")
    print("Successfully selected the free GPU {} in {}.".format(gpu, Node))
    return gpu