from model import SampleRNN, Predictor, Generator
from optim import gradient_clipping
from nn import sequence_nll_loss_bits
from dataset import FolderDataset, DataLoader

from natsort import natsorted
from librosa.output import write_wav
from tensorboardX import SummaryWriter
from glob import glob

import torch
import os
import shutil
import sys
import re
import argparse
import numpy as np
import datetime
import time



# dropout probability



default_params = {
    # model parameters
    'n_rnn': 3,
    'dim': 1024,
    'learn_h0': True,
    'q_levels': 256,
    'seq_len': 1024,
    'weight_norm': True,
    'batch_size': 128,
    'val_frac': 0.1,
    'test_frac': 0.1,
    'dropout': 0.0,
    'lr': 0.001,

    # training parameters
    'keep_old_checkpoints': False,
    'datasets_path': 'datasets',
    'results_path': 'results',
    'epoch_limit': 1000,
    'resume': False,
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 80000,
    'loss_smoothing': 0.99,
    'cuda': True,
    'gpu': '0'
}

tag_params = [
    'exp', 'frame_sizes', 'n_rnn', 'dim', 'learn_h0', 'q_levels', 'seq_len',
    'batch_size', 'dataset', 'val_frac', 'test_frac', 'dropout', 'lr', 'sample_rate'
]

def param_to_string(value):
    if isinstance(value, bool):
        return 'T' if value else 'F'
    elif isinstance(value, list):
        return ','.join(map(param_to_string, value))
    else:
        return str(value)

def make_tag(params):
    return '-'.join(
        key + ':' + param_to_string(params[key])
        for key in tag_params
        if key not in default_params or params[key] != default_params[key]
    )

def setup_results_dir(params):
    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    tag = make_tag(params)
    results_path = os.path.abspath(params['results_path'])
    ensure_dir_exists(results_path)
    results_path = os.path.join(results_path, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    elif not params['resume']:
        shutil.rmtree(results_path)
        os.makedirs(results_path)

    for subdir in ['checkpoints', 'samples']:
        ensure_dir_exists(os.path.join(results_path, subdir))

    return results_path

def load_last_checkpoint(checkpoints_path, params):
    if 'load_model' in params:
        """
        checkpoint_path = params['load_model']
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(
            'best-ep{}-it{}'.format(r'(\d+)', r'(\d+)'),
            checkpoint_name
        )
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return (torch.load(checkpoint_path), epoch, iteration)
        """
        checkpoint_path = params['load_model']
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(
            'best-ep{}-it{}'.format(r'(\d+)', r'(\d+)'),
            checkpoint_name
        )
        epoch = int(match.group(1))
        iteration = int(match.group(2))

        print("\n", checkpoint_path, epoch, iteration, "\n")
        return (torch.load(checkpoint_path), epoch, iteration)

    checkpoints_pattern = os.path.join(
        checkpoints_path, 'ep{}-it{}'.format('*', '*')
    )
    checkpoint_paths = natsorted(glob(checkpoints_pattern))
    if len(checkpoint_paths) > 0:
        checkpoint_path = checkpoint_paths[-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(
            'ep{}-it{}'.format(r'(\d+)', r'(\d+)'),
            checkpoint_name
        )
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return (torch.load(checkpoint_path), epoch, iteration)
    else:
        return None

def tee_stdout(log_path):
    log_file = open(log_path, 'a', 1)
    stdout = sys.stdout

    class Tee:

        def write(self, string):
            log_file.write(string)
            stdout.write(string)

        def flush(self):
            log_file.flush()
            stdout.flush()

    sys.stdout = Tee()

def make_data_loader(overlap_len, params):
    path = os.path.join(params['datasets_path'], params['dataset'])
    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(
            path, overlap_len, params['q_levels'], split_from, split_to
        )
        return DataLoader(
            dataset,
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            overlap_len=overlap_len,
            shuffle=(not eval),
            drop_last=(not eval)
        )
    return data_loader




def generate_sample(generator, params, writer, global_step, results_path, e):
            # generator: epoch
            pattern = 'ep{}-s{}.wav'
            samples = generator(params['n_samples'], params['sample_length']) \
                        .cpu().float().numpy()

            norm_samples = ((samples[:] - samples[:].min()) / (0.00001 + (samples[:].max() - samples[:].min()))) * 1.9 - 0.95

            if writer is not None:
                writer.add_scalar('validation/sample average', np.mean(samples), global_step)
                writer.add_scalar('validation/sample min', samples.min(), global_step)
                writer.add_scalar('validation/sample max', samples.max(), global_step)

            start = time.time()
            for i in range(params['n_samples']):
                if writer is not None:
                    writer.add_audio('validation/sound{}'.format(i), norm_samples[i], global_step, sample_rate=params['sample_rate'])
                write_wav(
                    os.path.join(
                    os.path.join(results_path, 'samples'), pattern.format(e, i + 1)
                    ),
                    samples[i, :], sr=params['sample_rate'], norm=True
                )
            avg_time = time.time() - start

            print("== Generated {} Samples ==".format(params['n_samples']))



def main(exp, frame_sizes, dataset, **params):
    params = dict(
        default_params,
        exp=exp, frame_sizes=frame_sizes, dataset=dataset,
        **params
    )

    os.environ['CUDA_VISIBLE_DEVICES']=params['gpu']

    results_path = setup_results_dir(params)
    tee_stdout(os.path.join(results_path, 'log'))

    model = SampleRNN(
        frame_sizes=params['frame_sizes'],
        n_rnn=params['n_rnn'],
        dim=params['dim'],
        learn_h0=params['learn_h0'],
        q_levels=params['q_levels'],
        weight_norm=params['weight_norm'],
        dropout=params['dropout']
    )
    predictor = Predictor(model)
    if params['cuda']:
        model = model.cuda()
        predictor = predictor.cuda()

    optimizer = gradient_clipping(torch.optim.Adam(predictor.parameters(), lr=params['lr']))

    data_loader = make_data_loader(model.lookback, params)
    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']

    criterion = sequence_nll_loss_bits

    checkpoints_path = os.path.join(results_path, 'checkpoints')
    checkpoint_data = load_last_checkpoint(checkpoints_path, params)
    if checkpoint_data is not None:
        (state_dict, epoch, iteration) = checkpoint_data
        start_epoch = int(epoch)
        global_step = iteration
        start_epoch = iteration
        predictor.load_state_dict(state_dict)
    else:
        start_epoch = 0
        global_step = 0


    #writer = SummaryWriter("runs/{}-{}".format(params['dataset'], str(datetime.datetime.now()).split('.')[0].replace(' ', '-')))
    writer = SummaryWriter(os.path.join(results_path, "{}-{}".format(params['dataset'], str(datetime.datetime.now()).split('.')[0].replace(' ', '-'))))
    dataset_train = data_loader(0, val_split, eval=False)
    dataset_val = data_loader(val_split, test_split, eval=True)
    dataset_test = data_loader(test_split, 1, eval=True)

    generator = Generator(predictor.model, params['cuda'])
    best_val_loss = 10000000000000

    for e in range(start_epoch, int(params['epoch_limit'])):
        for i, data in enumerate(dataset_train):

            batch_inputs = data[:-1]
            batch_target = data[-1]

            def wrap(input):
                if torch.is_tensor(input):
                    input = torch.autograd.Variable(input)
                    if params['cuda']:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = torch.autograd.Variable(batch_target)
            if params['cuda']:
                batch_target = batch_target.cuda()

            plugin_data = [None, None]

            def closure():
                batch_output = predictor(*batch_inputs)

                loss = criterion(batch_output, batch_target)
                loss.backward()

                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data

                return loss

            optimizer.zero_grad()
            optimizer.step(closure)
            train_loss = plugin_data[1]

            # stats: iteration
            writer.add_scalar('train/train loss', train_loss, global_step)
            print("E:{:03d}-S{:05d}: Loss={}".format(e, i, train_loss))
            global_step += 1





        # validation: per epoch
        predictor.eval()
        with torch.no_grad():
            loss_sum = 0
            n_examples = 0
            for data in dataset_val:
                batch_inputs = data[: -1]
                batch_target = data[-1]
                batch_size = batch_target.size()[0]

                def wrap(input):
                    if torch.is_tensor(input):
                        input = torch.autograd.Variable(input)
                        if params['cuda']:
                            input = input.cuda()
                    return input
                batch_inputs = list(map(wrap, batch_inputs))

                batch_target = torch.autograd.Variable(batch_target)
                if params['cuda']:
                    batch_target = batch_target.cuda()

                batch_output = predictor(*batch_inputs)

                loss_sum += criterion(batch_output, batch_target).item() * batch_size

                n_examples += batch_size

            val_loss = loss_sum / n_examples
            writer.add_scalar('validation/validation loss', val_loss, global_step)
            print("== Validation Step E:{:03d}: Loss={} ==".format(e, val_loss))

        predictor.train()


        # saver: epoch
        last_pattern = 'ep{}-it{}'
        best_pattern = 'best-ep{}-it{}'
        if not params['keep_old_checkpoints']:
            pattern = os.path.join(checkpoints_path, last_pattern.format('*', '*'))
            for file_name in glob(pattern):
                os.remove(file_name)
        torch.save(predictor.state_dict(), os.path.join(checkpoints_path, last_pattern.format(e, global_step)))

        cur_val_loss = val_loss
        if cur_val_loss < best_val_loss:
            pattern = os.path.join(checkpoints_path, last_pattern.format('*', '*'))
            for file_name in glob(pattern):
                os.remove(file_name)
            torch.save(predictor.state_dict(), os.path.join(checkpoints_path, best_pattern.format(e, global_step)))
            best_val_loss = cur_val_loss


        generate_sample(generator, params, writer, global_step, results_path, e)



    # generate final results
    generate_sample(generator, params, None, global_step, results_path, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()

    parser.add_argument('--exp', required=True, help='experiment name')
    parser.add_argument(
        '--frame_sizes', nargs='+', type=int, required=True,
        help='frame sizes in terms of the number of lower tier frames, \
              starting from the lowest RNN tier'
    )
    parser.add_argument(
        '--dataset', required=True,
        help='dataset name - name of a directory in the datasets path \
              (settable by --datasets_path)'
    )
    parser.add_argument(
        '--n_rnn', type=int, help='number of RNN layers in each tier'
    )
    parser.add_argument(
        '--dim', type=int, help='number of neurons in every RNN and MLP layer'
    )
    parser.add_argument(
        '--learn_h0', type=parse_bool,
        help='whether to learn the initial states of RNNs'
    )
    parser.add_argument(
        '--q_levels', type=int,
        help='number of bins in quantization of audio samples'
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='how many samples to include in each truncated BPTT pass'
    )
    parser.add_argument(
        '--weight_norm', type=parse_bool,
        help='whether to use weight normalization'
    )
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument(
        '--val_frac', type=float,
        help='fraction of data to go into the validation set'
    )
    parser.add_argument(
        '--test_frac', type=float,
        help='fraction of data to go into the test set'
    )
    parser.add_argument(
        '--keep_old_checkpoints', type=parse_bool,
        help='whether to keep checkpoints from past epochs'
    )
    parser.add_argument(
        '--datasets_path', help='path to the directory containing datasets'
    )
    parser.add_argument(
        '--results_path', help='path to the directory to save the results to'
    )
    parser.add_argument('--epoch_limit', help='how many epochs to run')
    parser.add_argument(
        '--resume', type=parse_bool, default=True,
        help='whether to resume training from the last checkpoint'
    )
    parser.add_argument(
        '--sample_rate', type=int,
        help='sample rate of the training data and generated sound'
    )
    parser.add_argument(
        '--n_samples', type=int,
        help='number of samples to generate in each epoch'
    )
    parser.add_argument(
        '--sample_length', type=int,
        help='length of each generated sample (in samples)'
    )
    parser.add_argument(
        '--loss_smoothing', type=float,
        help='smoothing parameter of the exponential moving average over \
              training loss, used in the log and in the loss plot'
    )
    parser.add_argument(
        '--cuda', type=parse_bool,
        help='whether to use CUDA'
    )
    parser.add_argument(
        '--gpu', type=str,
        help='which GPU to use'
    )
    parser.add_argument(
        '--dropout', type=float,
        help='dropout probability'
    )
    parser.add_argument(
        '--lr', type=float,
        help='learning rate'
    )
    parser.add_argument(
        '--load_model', required=False,
        help='Load a certain model with this path'
    )

    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
