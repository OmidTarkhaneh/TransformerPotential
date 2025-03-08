def ModelTest(training, validation, data_test,species_order,energy_shifter, epochs, device):
        
       
        import torch
        import math
        import torch.utils.tensorboard
        import tqdm
        import numpy as np
      
        import torch.nn as nn
        import torch.optim as optim  
        from sklearn.metrics import mean_squared_error
        import pandas as pd
        from aev import AEVComputer, SpeciesAEV
        from units import hartree2kjoulemol
        from nn import TransformerModel, Sequential
        import os
        import pickle



        # device to run the training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        Rcr = 5.1000e+00
        Rca = 3.5000e+00
        EtaR = torch.tensor([19.7000000e+00], device=device)
        ShfR = torch.tensor([8.0000000e-01, 1.068800e+00, 1.337500e+00, 1.606300e+00, 1.8750000e+00, 2.143700e+00, 2.4125000e+00, 2.681300e+00, 2.9500000e+00, 3.218800e+00, 3.4875000e+00, 3.756200e+00, 4.0250000e+00, 4.293700e+00, 4.5625000e+00, 4.831300e+00], device=device)
        Zeta = torch.tensor([1.4100000e+01], device=device)
        ShfZ = torch.tensor([3.927000e-01, 1.17810000e+00, 1.9635000+00, 2.7489000e+00], device=device)
        EtaA = torch.tensor([1.25000000e+01], device=device)
        ShfA = torch.tensor([8.0000000e-01, 1.1375000e+00, 1.4750000e+00, 1.8125000e+00, 2.15000e+00, 2.48750e+00, 2.8250e+00, 3.1625000e+00], device=device)
        species_order = ['H','C', 'N','O', 'S', 'Cl']

        num_species = len(species_order)
        aev_computer = AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        # energy_shifter = EnergyShifter(None)
        print('Self atomic energies: ', energy_shifter.self_energies)






        ####################################################################
        aev_dim = aev_computer.aev_length

        import torch
        import torch.nn as nn





        #####################################
        ################################################
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset

        class TransformerEncoderMLP(nn.Module):
                def __init__(self, input_size, hidden_size, n_heads, n_layers, ff_hidden_size, mlp_hidden_size):
                        super(TransformerEncoderMLP, self).__init__()

                        # Transformer Encoder
                        self.lin1=nn.Linear(input_size, mlp_hidden_size)
                        self.elu=nn.GELU()
                        self.embedding = nn.Linear(mlp_hidden_size, hidden_size)
                        self.transformer_encoder_layer = nn.TransformerEncoderLayer(hidden_size, n_heads, dim_feedforward=ff_hidden_size)
                        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=n_layers)

                        # MLP
                        self.mlp = nn.Sequential(
                        nn.Linear(hidden_size, mlp_hidden_size),
                        nn.GELU(),
                        nn.Linear(mlp_hidden_size, mlp_hidden_size),
                        nn.GELU(),
                        nn.Linear(mlp_hidden_size, 1)
                        )

                def forward(self, x):
                        # Transformer Encoder
                        x=self.lin1(x)
                        x=self.elu(x)
                        
                        x_transformed = self.embedding(x)
                        x_transformed=torch.unsqueeze(x_transformed, dim=0)
                        x_transformed = self.transformer_encoder(x_transformed)
                        x_transformed=torch.squeeze(x_transformed, dim=0)

                        # MLP
                        output = self.mlp(x_transformed)
                        return output


        input_size = aev_dim
        hidden_size = 64
        output_size = 1

        # Set hyperparameters
        n_heads = 2
        n_layers = 1
        # transformer_dropout = 0.1
        ff_hidden_size = 128
        mlp_hidden_size=128

        # Instantiate the model
        H_network = TransformerEncoderMLP(input_size, hidden_size, n_heads, n_layers, ff_hidden_size, mlp_hidden_size)



        nn = TransformerModel(H_network)

        print(nn)

        ###############################################################################
        # Initialize the weights and biases.
        #
        # .. note::
        #   Pytorch default initialization for the weights and biases in linear layers
        #   is Kaiming uniform. See: `TORCH.NN.MODULES.LINEAR`_
        #   We initialize the weights similarly but from the normal distribution.
        #   The biases were initialized to zero.
        #
        # .. _TORCH.NN.MODULES.LINEAR:
        #   https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear


        def init_params(m):
                if isinstance(m, torch.nn.Linear):
                        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
                        torch.nn.init.zeros_(m.bias)


        nn.apply(init_params)

        ###############################################################################
        # Let's now create a pipeline of AEV Computer --> Neural Networks.
        # model = torchani.nn.Sequential(aev_computer, nn).to(device)
        model = Sequential(aev_computer, nn).to(device)
        AdamW = torch.optim.AdamW(model.parameters(), lr=0.001)




        # SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        ###############################################################################
        # Setting up a learning rate scheduler to do learning rate decay
        AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=50, threshold=0.0000001)
        # SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

        ###############################################################################
        # Train the model by minimizing the MSE loss, until validation RMSE no longer
        # improves during a certain number of steps, decay the learning rate and repeat
        # the same process, stop until the learning rate is smaller than a threshold.
        #
        # We first read the checkpoint files to restart training. We use `latest.pt`
        # to store current training state.
        latest_checkpoint = 'latest.pt'


        # if os.path.isfile(latest_checkpoint):
        #     checkpoint = torch.load(latest_checkpoint)
        #     nn.load_state_dict(checkpoint['nn'])
        #     AdamW.load_state_dict(checkpoint['AdamW'])
        #     AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])


        import torch
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()


        # During training, we need to validate on validation set and if validation error
        # is better than the best, then save the new best model to a checkpoint


        def validate():
        # run validation
                mse_sum = torch.nn.MSELoss(reduction='sum')
                total_mse = 0.0
                count = 0
                true_energies_1=[]
                predicted_energies_1=[]

                true_dftmain_energy=[]
                predicted_dftmain_energies=[]

                model.train(False)
                with torch.no_grad():
                        for properties in validation:
                                species = properties['species'].to(device)
                                coordinates = properties['coordinates'].to(device).float()
                                true_energies = properties['energies'].to(device)
                                _, predicted_energies = model((species, coordinates))
                                total_mse += mse_sum(predicted_energies, true_energies).item()
                                count += predicted_energies.shape[0]

                                # save predicted and true energy in list
                                predicted_energies_1.append(predicted_energies.detach().cpu().numpy())
                                true_energies_1.append(true_energies.detach().cpu().numpy())



                model.train(True)
                return hartree2kjoulemol(math.sqrt(total_mse / count)), predicted_energies_1, true_energies_1



        ##################################################################################
        """# Model Training"""
        ##################################################################################

        # We will also use TensorBoard to visualize our training process
        tensorboard = torch.utils.tensorboard.SummaryWriter()
        max_gradient_norm = 1.0

        rmse_val=[]
        best_rmse=[]
        lr_val=[]
        ###############################################################################
        # Finally, we come to the training loop.
        #
        # In this tutorial, we are setting the maximum epoch to a very small number,
        # only to make this demo terminate fast. For serious training, this should be
        # set to a much larger value
        mse = torch.nn.MSELoss(reduction='none')

        print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
        max_epochs = epochs
        early_stopping_learning_rate = 1.0E-7
        best_model_checkpoint = 'best.pt'

        for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
                rmse, predicted_energies_1, true_energies_1 = validate()
                print('RMSE (kJ/mol):', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

                learning_rate = AdamW.param_groups[0]['lr']

                if learning_rate < early_stopping_learning_rate:
                        break

                # checkpoint
                if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
                        torch.save(nn.state_dict(), best_model_checkpoint)

                AdamW_scheduler.step(rmse)
        

                tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
                rmse_val.append(rmse)
                tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
                best_val=AdamW_scheduler.best
                best_rmse.append(best_val)
                tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)
                lr_val.append(learning_rate)

                for i, properties in tqdm.tqdm(
                        enumerate(training),
                        total=len(training),
                        desc="epoch {}".format(AdamW_scheduler.last_epoch)
                ):
                        species = properties['species'].to(device)
                        coordinates = properties['coordinates'].to(device).float()
                        true_energies = properties['energies'].to(device).float()
                        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                        _, predicted_energies = model((species, coordinates))

                        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

                        AdamW.zero_grad()
                        # SGD.zero_grad()
                        loss.backward()
                        # Apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)

                        AdamW.step()
                        # SGD.step()

                        # write current batch loss to TensorBoard
                        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)


        

        def validate2():

                mse_sum = torch.nn.MSELoss(reduction='sum')
                total_mse = 0.0
                count = 0
                true_energies_1=[]
                predicted_energies_1=[]
                true_dft1main_energy=[]
                predicted_dft1main_energies=[]

                model.train(False)
                with torch.no_grad():
                        for properties in data_test:
                                species = properties['species'].to(device)
                                coordinates = properties['coordinates'].to(device).float()
                                true_energies = properties['energies'].to(device)
                                _, predicted_energies = model((species, coordinates))
                                total_mse += mse_sum(predicted_energies, true_energies).item()
                                count += predicted_energies.shape[0]

                                energy_shift = energy_shifter.sae(species)
                                true_dft1_energy = true_energies + energy_shift.to(device)
                                predicted_dft1_energies= predicted_energies + energy_shift.to(device)
                                
                                # save predicted and true energy in list 
                                predicted_energies_1.append(predicted_energies.detach().cpu().numpy())
                                true_energies_1.append(true_energies.detach().cpu().numpy())

                                true_dft1main_energy.append(true_dft1_energy.detach().cpu().numpy())
                                predicted_dft1main_energies.append(predicted_dft1_energies.detach().cpu().numpy())


                model.train(True)
                return hartree2kjoulemol(math.sqrt(total_mse / count)), predicted_energies_1, true_energies_1,true_dft1main_energy,predicted_dft1main_energies





        device='cpu'
        model=model.to(device)
        rmse_1, predicted_energies_111,true_energies_111,true_dft1main_energy,predicted_dft1main_energies = validate2()


        true_energies_22= np.hstack(true_energies_111)
        pred_energies_22= np.hstack(predicted_energies_111)

        mae=np.sum(np.abs(true_energies_22-pred_energies_22))

        mae_test=mae/(len(true_energies_22))

        print('overall MAE(kJ/mol)=',hartree2kjoulemol(mae_test))
        print('overall RMSE(kJ/mol)=',(rmse_1))

        mse=mean_squared_error(true_energies_22,pred_energies_22)

        rmse_test=np.sqrt(mse)
        # rmse_test=hartree2kjoulemol(rmse_test)
        print('overall rmse_test(kJ/mol)=',hartree2kjoulemol(rmse_test))
######################################################################################

        return  hartree2kjoulemol(rmse_test), hartree2kjoulemol(mae_test)


