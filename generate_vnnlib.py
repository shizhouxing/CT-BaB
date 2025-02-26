# Generate VNNLIB specification used for verifying Lyapunov condition under level set constraint.

import os
import sys
import time
import socket
import argparse
import torch


def add_hole(box_low, box_high, inner_low, inner_high):
    boxes_low = []
    boxes_high = []
    for i in range(box_low.size(0)):
        # Split on dimension i.
        box1_low = box_low.clone()
        box1_low[i] = inner_high[i]
        box1_high = box_high.clone()
        box2_low = box_low.clone()
        box2_high = box_high.clone()
        box2_high[i] = inner_low[i]
        boxes_low.extend([box1_low, box2_low])
        boxes_high.extend([box1_high, box2_high])
        box_low[i] = inner_low[i]
        box_high[i] = inner_high[i]
    boxes_low = torch.stack(boxes_low, dim=0)
    boxes_high = torch.stack(boxes_high, dim=0)
    return boxes_low, boxes_high


def box_data(lower_limit=-1.0, upper_limit=1.0, ndim=2, scale=1.0, hole_size=0):
    """
    Generate a box between (-1, -1) and (1, 1) as our region to verify stability.
    We may place a small hole around the origin.
    """
    if isinstance(lower_limit, list):
        data_min = scale * torch.tensor(
            lower_limit, dtype=torch.get_default_dtype()
        ).unsqueeze(0)
    else:
        data_min = scale * torch.ones((1, ndim)) * lower_limit
    if isinstance(upper_limit, list):
        data_max = scale * torch.tensor(
            upper_limit, dtype=torch.get_default_dtype()
        ).unsqueeze(0)
    else:
        data_max = scale * torch.ones((1, ndim)) * upper_limit
    if hole_size != 0:
        hole_lower = data_min.squeeze(0) * hole_size
        hole_upper = data_max.squeeze(0) * hole_size
        data_min, data_max = add_hole(
            data_min.squeeze(0), data_max.squeeze(0), hole_lower, hole_upper
        )
    return data_max, data_min


def generate_preamble(out, num_x, num_y):
    out.write(
        f"; Generated at {time.ctime()} on {socket.gethostname()} by {os.getlogin()}\n"
    )
    out.write(f'; Generation command: \n; {" ".join(sys.argv)}\n\n')
    for i in range(num_x):
        out.write(f"(declare-const X_{i} Real)\n")
    out.write('\n')
    for i in range(num_y):
        out.write(f"(declare-const Y_{i} Real)\n")
    out.write("\n")


def generate_limits(out, lower_limit, upper_limit, value_levelset):
    lower_limit = lower_limit.tolist()
    upper_limit = upper_limit.tolist()
    assert len(lower_limit) == len(upper_limit)
    for i, (l, u) in enumerate(zip(lower_limit, upper_limit)):
        out.write(f"(assert (<= X_{i} {u}))\n")
        out.write(f"(assert (>= X_{i} {l}))\n\n")
    out.write('\n')


def generate_specs(out, full_x_L, full_x_U, value_levelset, tolerance,
                   output_spec_index=None):
    out.write(f"(assert (or\n")
    if output_spec_index is None or output_spec_index == 0:
        out.write(f"  (and (>= Y_0 {tolerance}))\n")
    for i, (l, u) in enumerate(zip(full_x_L, full_x_U)):
        if output_spec_index is None or output_spec_index == i * 2 + 1:
            out.write(f"  (and (<= Y_{i+2} {l - tolerance}))\n")
        if output_spec_index is None or output_spec_index == i * 2 + 2:
            out.write(f"  (and (>= Y_{i+2} {u + tolerance}))\n")
    out.write("))\n")
    out.write(f"(assert (<= Y_1 {value_levelset}))\n")


def generate_csv(output_path, num_regions):
    fname = output_path + ".csv"
    with open(fname, "w") as out:
        for i in range(num_regions):
            out.write(f"{output_path}_{i}.vnnlib\n")


def generate_instances(x_L, x_U, full_x_L, full_x_U, value_levelset,
                       tolerance=1e-6, output_path='specs', single_and=False):
    assert x_L.ndim == x_U.ndim == 2
    if single_and:
        index = 0
        for i in range(x_L.shape[0]):
            for j in range(x_L.shape[0] * 2 + 1):
                generate_instance(
                    f"{output_path}_{index}.vnnlib",
                    x_L[i], x_U[i], full_x_L, full_x_U, value_levelset, tolerance,
                    output_spec_index=j,
                )
                index += 1
        generate_csv(output_path, index)
    else:
        for i in range(x_L.shape[0]):
            generate_instance(
                f"{output_path}_{i}.vnnlib",
                x_L[i], x_U[i], full_x_L, full_x_U, value_levelset, tolerance,
            )
        generate_csv(output_path, x_L.shape[0])


def generate_instance(fname, x_L, x_U, full_x_L, full_x_U, value_levelset,
                      tolerance, output_spec_index=None):
    with open(fname, "w") as out:
        generate_preamble(out, len(x_L), len(x_L) + 2)
        generate_limits(out, x_L, x_U, value_levelset)
        generate_specs(out, full_x_L, full_x_U, value_levelset, tolerance,
                       output_spec_index=output_spec_index)


def main():
    parser = argparse.ArgumentParser(
        prog="VNNLIB Generator",
        description="Generate VNNLIB property file for verification of Lyapunov condition under level set constraint",
    )
    parser.add_argument("output_filename", type=str,
                        help="Output filename prefix. A single csv file and multiple VNNLIB files will be generated.")
    parser.add_argument("--input_dim", type=int, default=None,
                        help="Broadcast scalar lower_limit and upper_limit if input_dim is set.")
    parser.add_argument("-l", "--lower_limit", type=float, nargs="+",
                        help="Lower limit of state dimension. A list of state_dim numbers.")
    parser.add_argument("-u", "--upper_limit", type=float, nargs="+",
                        help="Upper limit of state dimension. A list of state_dim numbers.")
    parser.add_argument("--hole_lower", type=float, nargs="+")
    parser.add_argument("--hole_upper", type=float, nargs="+")
    parser.add_argument("-s", "--scale", type=float, default=1.0,
                        help="Scaling of lower limit and upper limit. Used for quickly try different sizes.")
    parser.add_argument("-o", "--hole_size", type=float, default=0,
                        help="Relative size of the hole in the middle to skip verification (0.0 - 1.0).")
    parser.add_argument("-t", "--tolerance", type=float, default=1e-6,
                        help="Numerical tolerance for verification. For single precision it is around 1e-6.")
    parser.add_argument("-v", "--value_levelset", type=float, default=0.0,
                        help="Level set value. We verify Lyapunov condition only when Lyapunov function is smaller than this value. Ignored when set to 0.")
    parser.add_argument("--single_and", action="store_true",
                        help="Only a single AND condition for verification in each VNNLIB.")

    args = parser.parse_args()
    assert args.hole_size >= 0 and args.hole_size <= 0.2
    if not os.path.exists(os.path.dirname(args.output_filename)):
        os.makedirs(os.path.dirname(args.output_filename))
    if args.input_dim:
        args.lower_limit = [args.lower_limit[0]] * args.input_dim
        args.upper_limit = [args.upper_limit[0]] * args.input_dim
    if not args.lower_limit:
        args.lower_limit = [-item for item in args.upper_limit]
    assert len(args.lower_limit) == len(args.upper_limit)
    state_dim = len(args.lower_limit)

    # Obtain the number of regions (subproblems) to verify.
    data_max, data_min = box_data(
        lower_limit=args.lower_limit,
        upper_limit=args.upper_limit,
        ndim=len(args.lower_limit),
        hole_size=args.hole_size,
        scale=args.scale,
    )

    generate_instances(
        data_min, data_max, args.lower_limit, args.upper_limit,
        args.value_levelset, args.tolerance, args.output_filename,
        single_and=args.single_and,
    )


if __name__ == "__main__":
    main()