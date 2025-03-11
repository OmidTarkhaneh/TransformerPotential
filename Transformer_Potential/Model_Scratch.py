def ModelTest_FromScratch(training, validation, data_test,species_order,energy_shifter, epochs, device):
        
       
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
        import torch.nn.functional as F



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

        print('Self atomic energies: ', energy_shifter.self_energies)
        aev_dim = aev_computer.aev_length


        ################################################

        class MultiHeadAttention(nn.Module):
            def __init__(self, hidden_size, n_heads):
                super(MultiHeadAttention, self).__init__()
                self.hidden_size = hidden_size
                self.n_heads = n_heads
                self.head_dim = hidden_size // n_heads

                # Ensure hidden_size is divisible by n_heads
                assert self.head_dim * n_heads == hidden_size

                # Q, K, V linear projections
                self.q_linear = nn.Linear(hidden_size, hidden_size)
                self.k_linear = nn.Linear(hidden_size, hidden_size)
                self.v_linear = nn.Linear(hidden_size, hidden_size)

                # Output projection
                self.out_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                batch_size, seq_len, hidden_size = x.size()

                # Linear projections
                q = self.q_linear(x)
                k = self.k_linear(x)
                v = self.v_linear(x)

                # Reshape for multi-head attention
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

                # softmax
                attention_weights = F.softmax(scores, dim=-1)

                #  attention weights to values
                context = torch.matmul(attention_weights, v)

                # Reshape back
                context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

                # Output projection
                output = self.out_proj(context)

                return output

        class FeedForward(nn.Module):
            def __init__(self, hidden_size, ff_hidden_size):
                super(FeedForward, self).__init__()
                self.linear1 = nn.Linear(hidden_size, ff_hidden_size)
                self.linear2 = nn.Linear(ff_hidden_size, hidden_size)
                self.gelu = nn.GELU()

            def forward(self, x):
                return self.linear2(self.gelu(self.linear1(x)))

        class LayerNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-12):
                super(LayerNorm, self).__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.bias = nn.Parameter(torch.zeros(hidden_size))
                self.eps = eps

            def forward(self, x):
                mean = x.mean(-1, keepdim=True)
                std = x.std(-1, keepdim=True)
                return self.weight * (x - mean) / (std + self.eps) + self.bias

        class TransformerEncoderLayer(nn.Module):
            def __init__(self, hidden_size, n_heads, dim_feedforward):
                super(TransformerEncoderLayer, self).__init__()

                # Multi-head attention
                self.self_attn = MultiHeadAttention(hidden_size, n_heads)

                # Feed-forward network
                self.feed_forward = FeedForward(hidden_size, dim_feedforward)

                # Layer normalization
                self.norm1 = LayerNorm(hidden_size)
                self.norm2 = LayerNorm(hidden_size)

            def forward(self, x):
                # Multi-head attention with residual connection and layer norm
                attn_output = self.self_attn(x)
                x = x + attn_output  
                x = self.norm1(x)  # Layer normalization

                # Feed-forward with residual connection and layer norm
                ff_output = self.feed_forward(x)
                x = x + ff_output  
                x = self.norm2(x)  

                return x

        class TransformerEncoder(nn.Module):
            def __init__(self, encoder_layer, num_layers):
                super(TransformerEncoder, self).__init__()
                self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class TransformerEncoderModel(nn.Module):
            def __init__(self, input_size, hidden_size, n_heads, n_layers, ff_hidden_size, mlp_hidden_size):
                super(TransformerEncoderModel, self).__init__()

                # Input processing
                self.lin1 = nn.Linear(input_size, mlp_hidden_size)
                self.gelu = nn.GELU()
                self.embedding = nn.Linear(mlp_hidden_size, hidden_size)
                # self.positional_encoding = PositionalEncoding(hidden_size)

                # custom transformer encoder layer
                encoder_layer = TransformerEncoderLayer(hidden_size, n_heads, ff_hidden_size)

                #  custom transformer encoder with multiple layers
                self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

              
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, mlp_hidden_size),
                    nn.GELU(),
                    nn.Linear(mlp_hidden_size, mlp_hidden_size),
                    nn.GELU(),
                    nn.Linear(mlp_hidden_size, 1)
                )

            def forward(self, x):
              
                x = self.lin1(x)
                x = self.gelu(x)

                # Embedding
                x_transformed = self.embedding(x)

                # Add batch dimension if not present
                if x_transformed.dim() == 2:
                    x_transformed = torch.unsqueeze(x_transformed, dim=0)

       
                # x_transformed = self.positional_encoding(x_transformed)
                x_transformed = self.transformer_encoder(x_transformed)

                # Remove batch dimension if it was added
                if x.dim() == 2:
                    x_transformed = torch.squeeze(x_transformed, dim=0)

          
                output = self.mlp(x_transformed)

                return output

 
        def create_model(aev_dim):
            input_size = aev_dim
            hidden_size = 64
            n_heads = 2
            n_layers = 1
            ff_hidden_size = 128
            mlp_hidden_size = 128

            H_network = TransformerEncoderModel(
                input_size, hidden_size, n_heads, n_layers, ff_hidden_size, mlp_hidden_size
            )

            return H_network


        H_network= create_model(aev_dim)
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


        import torch
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()



        def validate():
        # run validation
                mse_sum = torch.nn.MSELoss(reduction='sum')
                total_mse = 0.0
                count = 0
                true_energies_1=[]
                predicted_energies_1=[]


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
               
                        loss.backward()
                        # Apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)

                        AdamW.step()
               
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

        print('overall rmse_test(kJ/mol)=',hartree2kjoulemol(rmse_test))
######################################################################################

        return  hartree2kjoulemol(rmse_test), hartree2kjoulemol(mae_test)


