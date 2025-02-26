import math
import torch
from torch.nn import functional as F
from project import project_params


def initialize(model_ori, dataset, scale=1.0, args=None):
    if args.init_method != 'default':
        for p in model_ori.named_parameters():
            if p[0].endswith('.weight'):
                std_before = p[1].std().item()
                if args.init_method == 'ibp':
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(p[1])
                    std = math.sqrt(2 * math.pi / (fan_in**2)) * scale
                    torch.nn.init.normal_(p[1], mean=0, std=std)
                    print(f'[IBP init] Reinitialize {p[0]}, std before {std_before:.5f}, std now {p[1].std():.5f}')
                elif args.init_method == 'kaiming':
                    torch.nn.init.kaiming_uniform_(p[1])
                    print(f'[Kaiming init] Reinitialize {p[0]}, std before {std_before:.5f}, std now {p[1].std():.5f}')
                else:
                    raise NotImplementedError

    lower_limit = dataset.lower_limit.to(args.device)
    upper_limit = dataset.upper_limit.to(args.device)
    x = torch.rand(args.batch_size, dataset.x_dim, device=args.device)
    x = x * (upper_limit - lower_limit) + lower_limit

    if not args.load_controller:
        # Check controller
        if model_ori.output_feedback:
            pass
        else:
            proj_controller = model_ori.controller.net.project_layer
            while True:
                u = model_ori.controller(x)
                x_next, _ = model_ori.dynamics.forward(x / model_ori.scale_input, u)
                x_next = x_next * model_ori.scale_input
                out_of_bound = (F.relu(x_next - upper_limit) + F.relu(lower_limit - x_next)).amax(dim=-1)
                out_of_bound = out_of_bound.mean()
                print('out_of_bound', out_of_bound)
                if out_of_bound > 1:
                    proj_controller.weight.data = proj_controller.weight / 2
                    proj_controller.bias.data = proj_controller.bias / 2
                else:
                    break

    if 'QuadraticLyapunov' in str(type(model_ori.lyapunov)):
        for _ in range(100):
            V_psd = model_ori.lyapunov(x)
            print('Checking V_PSD', V_psd.mean())
            if V_psd.mean() > 1.0:
                model_ori.lyapunov.R.data = model_ori.lyapunov.R / 2
            else:
                break

    elif model_ori.lyapunov.V_psd_form == 'new':
        for param in model_ori.lyapunov.named_parameters():
            if '.weight' in param[0]:
                torch.nn.init.kaiming_uniform_(param[1])

        project_params(model_ori, use_abs=True)

        # # # Pre-fit lyapunov function to the default quadratic lyapunov function
        # opt = torch.optim.Adam(model_ori.lyapunov.parameters(), lr=1e-3)
        # Q = (
        #     model_ori.lyapunov.eps * torch.eye(dataset.x_dim, device=args.device)
        #     + model_ori.lyapunov.R.transpose(0, 1) @ model_ori.lyapunov.R
        # ).detach()
        # for t in range(50000):
        #     x = torch.rand(10000, dataset.x_dim, device=args.device)
        #     x = x * (upper_limit - lower_limit) + lower_limit
        #     output_V = model_ori.lyapunov(x)
        #     output_V_quad = torch.sum(x * (x @ Q), axis=1, keepdim=True)
        #     loss = ((output_V - output_V_quad)**2).mean()
        #     if t % 100 == 0:
        #         print('Pre-training NN Lyapunov', t, loss)
        #     opt.zero_grad()
        #     loss.backward()
        #     opt.step()
        #     project_params(model_ori)
        #     if loss < 1e-2:
        #         break

    elif model_ori.lyapunov.V_psd_form == 'L1':
        assert model_ori.lyapunov.scale == 1

        for _ in range(100):
            V_psd = model_ori.lyapunov._psd(x)
            print('Checking V_PSD', V_psd.max())
            if V_psd.max() > args.lyapunov_psd_init:
                model_ori.lyapunov.R.data = model_ori.lyapunov.R / 1.25
            else:
                break

        proj_net = model_ori.lyapunov.net.net[args.lyapunov_depth * 2 - 2]
        while True:
            V_net1 = model_ori.lyapunov._net1(x)
            V = V_psd + V_net1
            print('Checking V', V.mean(), V.max())
            if args.init_shrink and V.max() > 1:
                proj_net.weight.data = proj_net.weight / 1.25
                proj_net.bias.data = proj_net.bias / 1.25
            elif V.max() < 0.8:
                proj_net.weight.data = proj_net.weight * 1.25
                proj_net.bias.data = proj_net.bias * 1.25
            else:
                break

    V = model_ori.lyapunov(x)
    print('Final V', V.mean(), V.max())
