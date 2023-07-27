import torch

def batch_padding(train, target, mini_batch_size):
        """
        Add padding to input data during mini-batch training by randomly sample data points without replacement. This is generally applied to the last batch
        where the actual number of data points is smaller than the batch size. This can happen when the number of data points is not evenly divisible by the
        selected mini-batch size.

        M: mini-batch size

        Input:
            train: (n_dat x 2 x T) Torch tensor, where n_dat <= M
                Training data
            target: (n_dat x 2 x T) Torch tensor, where n_dat <= M
                Target data
        Output:
            train_new: (M x 2 x T) Torch tensor
                Training data, padded to match the mini-batch size
            target_new: (M x 2 x T) Torch tensor
                Target data, padded to match the mini-batch size
        """
        
        # shuffling
        diff = mini_batch_size - train.size(0)
        shuf_inds = torch.randperm(train.size(0))
        train_shuf = train[shuf_inds]
        target_shuf = target[shuf_inds]

        # draw without replacement and pad data
        train_pad = train_shuf[:diff]
        target_pad = target_shuf[:diff]
        train_new = torch.cat([train, train_pad], dim=0)
        target_new = torch.cat([target, target_pad], dim=0)
        
        return train_new, target_new