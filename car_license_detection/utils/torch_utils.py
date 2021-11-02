import os
import time
import torch
import torch.backends.cudnn as cudnn


def init_seeds(seed=0):
    torch.manual_seed(seed)

    # Reduce randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device="", apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), (
            "CUDA unavailable, invalid device %s requested" % device
        )  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, "batch-size %g not multiple of GPU count %g" % (
                batch_size,
                ng,
            )
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA " + (
            "Apex " if apex else ""
        )  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = " " * len(s)
            print(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                % (s, i, x[i].name, x[i].total_memory / c)
            )
    else:
        print("Using CPU")

    print("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )

    try:  # FLOPS
        from thop import profile

        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 480, 640),), verbose=False)
        fs = ", %.1f GFLOPS" % (macs / 1e9 * 2)
    except:
        fs = ""

    print(
        "Model Summary: %g layers, %g parameters, %g gradients%s"
        % (len(list(model.parameters())), n_p, n_g, fs)
    )
