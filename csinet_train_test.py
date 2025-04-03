import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
# from torchinfo import summary
from tqdm import tqdm
from scipy.io import savemat
import sys, datetime
from models.CsinetPlus import CsinetPlus
from utils.cal_nmse import cal_nmse
from data_feed.data_feed import DataFeed
from einops import rearrange

def train_model(
    train_loader,
    val_loader,
    test_loader,
    comment="unknown",
    encoded_dim=16,
    num_epoch=200,
    lr=1e-2,
    if_writer=False,
    model_path_save=None,
    model_path_load=None,
    load_model=False,
    save_model=False,
    Nc=100, # Number of subcarriers (delay bins)
    Nt=64,  # Number of antennas    (angle bins)
    n_refine_nets=5, # number of refine layers at the decoder
):
    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    
    print(f'Creating net with {n_refine_nets} refine nets at decoder side.')
    # instantiate the model and send to GPU
    net = CsinetPlus(encoded_dim, Nc, Nt, n_refine_nets=n_refine_nets)
    net.to(device)

    # path to save the model
    comment = comment + "_" + net.name
    
    if load_model and model_path_load:
        net.load_state_dict(torch.load(model_path_load))
    
    if save_model and not model_path_save:
        model_path_save = "checkpoint/" + comment + ".path"

    # print model summary
    # if_writer = False
    if if_writer:
        # summary(net, input_data=torch.zeros(16, 2, Nc, Nt).to(device))
        writer = SummaryWriter(log_dir="logs/" + comment)
        # writer =0

    # set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 90], gamma=0.5
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # training
    all_train_nmse = []
    all_val_nmse = []
    for epoch in range(num_epoch):
        net.train()
        running_loss = 0.0
        running_nmse = 0.0
        with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
            for i, data in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                # get the inputs
                input_channel, data_idx = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                encoded_vector, output_channel = net(input_channel)
                loss = criterion(output_channel, input_channel)

                nmse = torch.mean(cal_nmse(input_channel, output_channel), 0).item()

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_nmse = (nmse + i * running_nmse) / (i + 1)
                log = OrderedDict()
                log["loss"] = "{:.6e}".format(running_loss)
                log["nmse"] = running_nmse
                tepoch.set_postfix(log)
            scheduler.step()
        all_train_nmse.append(running_nmse)
        
        if val_loader is None:
            continue  # no validation is needed

        if epoch >= num_epoch - 50 or if_writer:
            # validation
            net.eval()
            with torch.no_grad():
                total = 0
                val_loss = 0
                val_nmse = 0

                for data in val_loader:
                    # get the inputs
                    input_channel, data_idx = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    encoded_vector, output_channel = net(input_channel)

                    val_loss += (
                        nn.MSELoss(reduction="mean")(
                            input_channel, output_channel
                        ).item()
                        * data_idx.shape[0]
                    )
                    val_nmse += torch.sum(cal_nmse(input_channel, output_channel), 0)
                    total += data_idx.shape[0]

                val_loss /= float(total)
                val_nmse /= float(total)
            all_val_nmse.append(val_nmse.item())
            print("val_loss={:.3e}".format(val_loss), flush=True)
            print("val_nmse={:.3f}".format(val_nmse), flush=True)
            if if_writer:
                writer.add_scalar("Loss/train", running_loss, epoch)
                writer.add_scalar("Loss/test", val_loss, epoch)
                writer.add_scalar("NMSE/train", running_nmse, epoch)
                writer.add_scalar("NMSE/test", val_nmse, epoch)

    if if_writer:
        writer.close()
    if model_path_save:
        torch.save(net.state_dict(), model_path_save)

    # test
    net.eval()
    with torch.no_grad():
        total = 0
        test_loss = 0
        test_nmse = 0

        for data in test_loader:
            # get the inputs
            input_channel, data_idx = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            encoded_vector, output_channel = net(input_channel)

            test_loss += (
                nn.MSELoss(reduction="mean")(input_channel, output_channel).item()
                * data_idx.shape[0]
            )
            test_nmse += torch.sum(cal_nmse(input_channel, output_channel), 0).item()
            total += data_idx.shape[0]

        test_loss /= float(total)
        test_nmse /= float(total)

        print("test_loss={:.6e}".format(test_loss), flush=True)
        print("test_nmse={:.6f}".format(test_nmse), flush=True)

        return {
            "all_train_nmse": all_train_nmse,
            "all_val_nmse": all_val_nmse,
            "test_loss": test_loss,
            "test_nmse": test_nmse,
            "model_path": model_path_save,
        }


def test_model(
    test_loader,
    net=None,
    model_path=None,
    encoded_dim=32,
    Nc=100,
    Nt=64,
    n_refine_nets=5,
):
    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    if device.type != 'cuda':
        print('not running on gpu!')

    # instantiate the model and send to GPU
    if model_path:
        net = CsinetPlus(encoded_dim, Nc, Nt, n_refine_nets=n_refine_nets)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    net.to(device)

    # test
    net.eval()
    with torch.no_grad():
        test_loss = []
        test_nmse = []
        test_data_idx = []
        inputs = []
        encoded_vects = []
        outputs = []
        for data in test_loader:
            # get the inputs
            input_channel, data_idx = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            encoded_vector, output_channel = net(input_channel)
            
            test_loss.append(nn.MSELoss(reduction="none")(input_channel, output_channel).mean((-1,-2,-3)).cpu().numpy())
            test_nmse.append(cal_nmse(input_channel, output_channel).cpu().numpy())
            test_data_idx.append(data_idx.cpu().numpy())            
            encoded_vects.append(encoded_vector.cpu().numpy())
            
            inputs_rearranged = rearrange(input_channel, 'Batch RealImag Nt Nc -> Batch Nt Nc RealImag')
            inputs.append(torch.view_as_complex(inputs_rearranged.contiguous()).cpu().numpy())
            
            outputs_rearranged = rearrange(output_channel, 'Batch RealImag Nt Nc -> Batch Nt Nc RealImag')
            outputs.append(torch.view_as_complex(outputs_rearranged.contiguous()).cpu().numpy())
            
        test_loss = np.concatenate(test_loss)
        test_nmse = np.concatenate(test_nmse)
        test_data_idx = np.concatenate(test_data_idx)
        inputs = np.concatenate(inputs)
        encoded_vects = np.concatenate(encoded_vects)
        outputs = np.concatenate(outputs)
        
        return {
            "test_loss_all": test_loss,
            "test_nmse_all": test_nmse,
            "test_data_idx": test_data_idx,
            "inputs": inputs,
            "encoded": encoded_vects,
            "outputs": outputs,
        }
    

def test_from_csv(csv_folder, csv_name, model_path=None, encoded_dim=32, Nc=100, Nt=64):
                          
    test_loader = DataLoader(DataFeed(csv_folder, csv_name, num_data_point=100000), 
                             batch_size=1024)
    
    test_results = test_model(test_loader=test_loader,
                              model_path=model_path,
                              encoded_dim=encoded_dim,
                              Nc=Nc,
                              Nt=Nt)
    return test_results


def test_AE(csv_folder, csv_name, model_path, encoded_dim=32, Nc=16, Nt=32):
    """
    csv_folder = "channel_datasets/1/" -> folder with a .mat with all the data and the CSV
    csv_name = "/train_data_idx.csv" -> file containing the indices for indexing all the data 
                                        (must be inside the csv folder)
    """
    return test_from_csv(csv_folder=csv_folder, csv_name=csv_name, 
                         model_path=model_path, encoded_dim=encoded_dim, Nc=Nc, Nt=Nt)

def train_model_loop(
    training_data_folder = "channel_datasets/1/",
    testing_data_folder = "channel_datasets/1/",
    train_csv = "/train_data_idx.csv",
    val_csv = "/test_data_idx.csv",
    test_csv = "/test_data_idx.csv",
    train_batch_size = 32,
    test_batch_size = 1024,
    n_runs = 1,
    list_number_training_datapoints = [5120],  # [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    
    # validation and testing parameters
    n_val_samples=10000,
    n_test_samples=10000,
    
    # train_model_params
    Nc=100,
    Nt=64,
    encoded_dim=32,
    num_epoch=110,
    tensorboard_writer=True,
    model_path_save=None,
    model_path_load=None,
    save_model=True,
    load_model=False,
    lr=1e-2,
    n_refine_nets=5, # number of refine layers at the decoder
    ):
    
    np.random.seed(10)
    seeds = np.random.randint(0, 10000, size=(1000,))

    all_avg_nmse = []
    for i in range(n_runs):
        all_nmse = []
        for num_train_data in list_number_training_datapoints:
            torch.manual_seed(seeds[i])
            train_loader = DataLoader(
                DataFeed(training_data_folder, train_csv, num_data_point=num_train_data, random_state=seeds[i]),
                batch_size=train_batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                DataFeed(training_data_folder, val_csv, num_data_point=n_val_samples, random_state=seeds[i]),
                batch_size=test_batch_size,
            )
            test_loader = DataLoader(
                DataFeed(testing_data_folder, test_csv, num_data_point=n_test_samples, random_state=seeds[i]),
                batch_size=test_batch_size,
            )

            now = datetime.datetime.now().strftime("%H_%M_%S")
            date = datetime.date.today().strftime("%y_%m_%d")
            comment = "_".join([now, date])

            print(f"Number of trainig    data points : {len(train_loader.dataset)}")
            print(f"Number of validation data points : {len(val_loader.dataset)}")
            print(f"Number of testing    data points : {len(test_loader.dataset)}")
            ret = train_model(
                train_loader,
                val_loader,
                test_loader,
                comment=comment,
                Nc=Nc,
                Nt=Nt,
                encoded_dim=encoded_dim,
                num_epoch=num_epoch,
                if_writer=tensorboard_writer,
                model_path_save=model_path_save,
                model_path_load=model_path_load,
                save_model=save_model,
                load_model=load_model,
                lr=lr,
                n_refine_nets=n_refine_nets
            )
            print(f'Returned results: {ret}')
            all_nmse.append(ret["all_val_nmse"])
        avg_nmse = np.asarray([np.asarray(nmse).mean() for nmse in all_nmse])
        all_avg_nmse.append(avg_nmse)
    all_avg_nmse = np.stack(all_avg_nmse, 0)

    print(all_avg_nmse)
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    savemat(results_folder + "/all_avg_nmse_train_on_synth.mat",
            {"all_avg_nmse_train_on_synth": all_avg_nmse})
    print("done")
    return ret # last ret