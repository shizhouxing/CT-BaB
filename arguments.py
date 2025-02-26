# pylint: disable=line-too-long

"""Arguments."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dir', type=str, default='experiment')
    parser.add_argument('--hf', type=str)
    parser.add_argument('--load', type=str)
    parser.add_argument('--load_last', action='store_true')
    parser.add_argument('--load_domains', type=str)
    parser.add_argument('--load_controller', type=str)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--no_save_domains', action='store_false', dest='save_domains')

    # Data
    parser.add_argument('--hole_size', type=float, default=1e-3)
    parser.add_argument('--hole_mode', type=str, default='non-overlapping')
    parser.add_argument('--max_in_hole_dim', type=int, default=6)
    parser.add_argument('--border_size', type=float, default=None)
    parser.add_argument('--box_size', type=float, default=None)
    parser.add_argument('--box_dim', type=int, default=None)
    parser.add_argument('--lower_limit', type=float, nargs='+')
    parser.add_argument('--upper_limit', type=float, nargs='+')
    parser.add_argument('--max_init_size', type=float, default=[1.0], nargs='+')
    parser.add_argument('--sample_splits', type=int, default=0)
    parser.add_argument('--num_data_workers', type=int, default=8)
    parser.add_argument('--scale_input', type=float, default=1.0)
    parser.add_argument('--sample_all_interval', type=int, default=1)

    # Modeling
    parser.add_argument('--proj_params', action='store_true')
    parser.add_argument('--load_checkpoint_scaled', type=int, default=1)
    parser.add_argument('--lyapunov_func', type=str, default='quadratic')
    parser.add_argument('--lyapunov_width', type=int, default=[64], nargs='+')
    parser.add_argument('--lyapunov_depth', type=int, default=2)
    parser.add_argument('--lyapunov_eps', type=float, default=1e-8)
    parser.add_argument('--lyapunov_act', type=str, default='relu')
    parser.add_argument('--lyapunov_R_rows', type=int, default=6)
    parser.add_argument('--lyapunov_R_scale', type=float, default=1.0)
    parser.add_argument('--lyapunov_nn_scale', type=float, default=1.0)
    parser.add_argument('--lyapunov_scale', type=float, default=1.0)
    parser.add_argument('--lyapunov_psd_form', type=str, default="quadratic")
    parser.add_argument('--lyapunov_psd_init', type=float, default=1e-4)
    parser.add_argument('--controller_width', type=int, default=8)
    parser.add_argument('--controller_depth', type=int, default=2)
    parser.add_argument('--controller_act', type=str, default='relu')
    parser.add_argument('--controller_arch', type=str, default='ff')
    parser.add_argument('--observer_width', type=int, default=8)
    parser.add_argument('--observer_depth', type=int, default=2)
    parser.add_argument('--dynamics', type=str, default="pendulum")
    parser.add_argument('--dynamics_version', type=str, default="default")
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--kappa', type=float, default=0)
    parser.add_argument('--kappa_adv', type=float, default=0, help='Kappa for adv only. Do not change `kappa`.')
    parser.add_argument('--init_method', type=str, default='default')
    parser.add_argument('--no_init', action='store_false', dest='init')
    parser.add_argument('--init_scale', type=float, default=1.0)
    parser.add_argument('--init_shrink', action='store_true')
    parser.add_argument('--debug_init', action='store_true')

    # Optimizer
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_interval', type=int, default=1000)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--lr_scheduler', type=str, default='multistep')
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--grad_norm', type=float, default=10.0)
    parser.add_argument('--scale_grad_norm', action='store_true')
    parser.add_argument('--adam_beta_1', type=float, default=0.9)
    parser.add_argument('--adam_beta_2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--wd', type=float, default=0)

    # Training
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_steps', type=int, default=100000)
    parser.add_argument('--loss_x_next_weight', type=float, default=1.0)
    parser.add_argument('--loss_observer_weight', type=float, default=0.0)
    parser.add_argument('--loss_rho_size', type=float, default=1.0)
    parser.add_argument('--no_loss_rho_sorted', action='store_false', dest='loss_rho_sorted')
    parser.add_argument('--no_loss_rho_clip', dest='loss_rho_clip', action='store_false', help='Clip loss rho with `rho_ratio')
    parser.add_argument('--tune_rho', action='store_true')
    parser.add_argument('--tune_rho_enlarge', action='store_true')
    parser.add_argument('--tune_rho_weight', type=float, default=1.0)
    parser.add_argument('--rho_ratio', type=float, default=0.5, help='No need to apply rho loss when the inside ratio is above a threshold.')
    parser.add_argument('--rho_penalty', type=float, default=1.0)
    parser.add_argument('--no_loss_for_included_only', action='store_false',
                        dest='loss_for_included_only')
    parser.add_argument('--loss_scale', type=float, default=1.0)
    parser.add_argument('--loss_empirical_weight', type=float, default=1e4)
    parser.add_argument('--loss_empirical_sum', action='store_true')
    parser.add_argument('--max_empirical_sum_points', type=int, default=1000000)
    parser.add_argument('--loss_verified_weight', type=float, default=1.0)
    parser.add_argument('--loss_adv_guided_weight', type=float, default=10.0)
    parser.add_argument('--loss_verified_sum', action='store_true')
    parser.add_argument('--loss_weighted', action='store_true')
    parser.add_argument('--loss_for_all', action='store_true', help='Apply loss even for points out of the levelset.')
    parser.add_argument('--no_loss_normalize', dest='loss_normalize', action='store_false')
    parser.add_argument('--epochs', type=int, default=int(1e8))
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--margin', type=float, default=1e-2)
    parser.add_argument('--margin_adv', type=float, default=1e-2)
    parser.add_argument('--margin_rho', type=float, default=0.)
    parser.add_argument('--adv_only', action='store_true')
    parser.add_argument('--adv_guided_version', type=str, default='v4')
    parser.add_argument('--adv_guided_thresh', type=float, default=1.5, help='Apply the guided loss when V found by attack is greater than thresh*rho')
    parser.add_argument('--max_adv_unsafe', type=int, default=1000)
    parser.add_argument('--no_loss_max', action='store_false', dest='loss_max')

    parser.add_argument('--focus_emp', type=int, default=100000, help='Focus on empirical training if the number of unsafe domains exceeds this threshold.')
    parser.add_argument('--focus_emp_weight', type=float, default=1, help='When focus on empirical training, scale other loss terms by this factor.')

    # Spliting
    parser.add_argument('--max_split_domains', type=int, default=1000000,
                        help='Maximum number of domains to split in each iteration.')
    parser.add_argument('--split_ub_thresh', type=float, default=100.0,
                        help='Always split if ub is larger than this threshold regardless of max_split_domains.')
    parser.add_argument('--max_split_domains_start_steps', type=int, default=1,
                        help='`max_split_domains` begins to take effect from this training step.')
    parser.add_argument('--split_interval', type=int, default=1)
    parser.add_argument('--split_threshold', type=float, default=1.0,
                        help='Split when the verified accuracy is already above the threshold.')
    parser.add_argument('--split_start_steps', type=int, default=100)
    parser.add_argument('--split_end_steps', type=int, default=5000)
    parser.add_argument('--max_num_domains', type=int, default=100000000)
    parser.add_argument('--no_use_even_splits', action='store_false',
                        dest='use_even_splits')
    parser.add_argument('--split_heuristic', default='bf')
    parser.add_argument('--split_batch_size', type=int, default=30000)
    parser.add_argument('--split_max_bf_domains', type=int, default=200000,
                        help='Use naive split if there are too many domains.')
    parser.add_argument('--split_bound_method', type=str)
    parser.add_argument('--split_ratio', type=float, default=0.5)

    # Bounding
    parser.add_argument('--bound_method', type=str, default='CROWN',
                        choices=['IBP', 'CROWN-IBP', 'CROWN', 'full-CROWN',
                                 'pruned-CROWN', 'pruned-full-CROWN'])
    parser.add_argument('--more_crown_for_output_feedback', action='store_true')
    parser.add_argument('--mul_middle', action='store_true')
    parser.add_argument('--relu_relaxation', type=str, default='adaptive')
    parser.add_argument('--drelu_relaxation', type=str, default='two_relus',
                        choices=['focus_on_x', 'focus_on_dx', 'two_relus'])
    parser.add_argument('--ibp_for_rx', action='store_true')
    parser.add_argument('--no_compare_crown_with_ibp', dest='compare_crown_with_ibp',
                        action='store_false')
    parser.add_argument('--no_crown_for_x_next', action='store_false',
                        dest='crown_for_x_next')

    # Simulation
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--sim_steps', type=int, default=100)
    parser.add_argument('--loss_sim_weight', type=float, default=1.0)

    # Attack
    parser.add_argument('--pgd_alpha', type=float, default=0.25)
    parser.add_argument('--pgd_steps', type=int, default=10)
    parser.add_argument('--pgd_restarts', type=int, default=1)
    parser.add_argument('--adv_size', type=int, default=20000)

    # Evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_roa_ticks', type=int, default=20)

    args = parser.parse_args()

    return args
