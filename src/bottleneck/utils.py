import torch
import os


def show_img(tensor):
    import matplotlib.pyplot as plt

    if type(tensor) == torch.Tensor:
        tensor = tensor.cpu().squeeze()
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
    plt.imshow(tensor)
    plt.show()


def get_project_dir():
    pwd = os.popen('pwd').read()
    # no / at the end
    return pwd[:pwd.index('/clip_representation') + len('/clip_representation')]


def get_cache_dir():
    path = os.path.join(get_project_dir(), 'cache')
    os.makedirs(path, exist_ok=True)
    return path


def get_device():
    def get_index_of_free_gpus():
        def get_free_gpu():
            lines = os.popen('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free').readlines()

            memory_available = [int(x.split()[2]) for x in lines]
            gpus = {index: mb for index, mb in enumerate(memory_available)}
            return gpus

        gpus = get_free_gpu()

        gpus = {index: mega for index, mega in gpus.items()}
        gpus = {k: v for k, v in sorted(gpus.items(), key=lambda x: x[1], reverse=True)}

        return gpus

    def compute_gpu_indent(gpus):
        try:
            # return 'cuda'
            best_gpu = max(gpus, key=lambda gpu_num: gpus[int(gpu_num)])
            return 'cuda:' + str(best_gpu)
        except:
            return 'cuda'

    gpus = get_index_of_free_gpus()
    # print(gpus)
    device = compute_gpu_indent(gpus) if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    return torch.device(device)
