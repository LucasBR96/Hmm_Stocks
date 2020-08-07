import torch
import torch.nn as net
import torch.nn.functional as net_fun

import numpy
from torch_mod import next_state_pipeline
from random import shuffle

class idx_generator:

    def __init__( self , num_idx , batch_size ):

        '''
        this is a auxiliarry class that generates random numbers between 0 and num_idx
        in batches of equal(ish) size, where every batches have no values in common with 
        the others. When running out of batches, new ones are created following this rule
        '''

        self.num_idx = num_idx
        self.batch_size = batch_size
        self.values = []

    def pop( self ):

        if len( self.values ) == 0:

            values = list( range( self.num_idx ) )
            shuffle( values )

            for i in range( self.num_idx//self.batch_size ):

                a = i*self.batch_size
                b = min( ( i + 1 )*self.batch_size , self.num_idx ) #in case of leftover
                sample = tuple( values[ a:b ] )
                self.values.append( sample )
        return self.values.pop()
                 

class trainer:

    def __init__( self , num_states , layers ):
        '''
        the purpose of this class is to host and train a next_state_pipeline model.
        A few other things are hosted as well, such as:
        spitter -> just throws random gaussian noise
        ref -> small piece of data to test the model performance. More on that in the fit() 
               and manage_cost() methods
        iter_counter -> just in case if the training is stopped
        '''

        self.cuda = torch.cuda.is_available() # this is self explanatory

        mod = next_state_pipeline( num_states , layers )
        if self.cuda:
            mod = mod.cuda()
        self.mod = mod 

        self.spitter = torch.distributions.Normal( 0 , 1 )
        self.iter_counter = 0
        self.ref = None 
    
    # SELF EXPLANATORY METHODS -----------------------------------------------------------------------------------------------
    def reset_counter( self ):

        self.iter_counter = 0
        self.ref = None 

    def check_device( self , X ):

        a = X.is_cuda
        b = self.cuda

        if a and ( not b ):
            X = X.to( 'cpu')
        if b and ( not a ):
            X = X.to( 'cuda')
        return X
    
    def sample_noise( self , size = 1 ):

        if size < 1:
            raise ValueError

        data = self.spitter.sample_n( size )
        if self.cuda:
            return data.to( 'cuda' )
        return data
    
    def predict( self , X ):

        X = self.check_device( X )
        mu , sigma = self.mod( X )
        noise = self.sample_noise( X.shape[ 0 ] )
        return ( mu + sigma*noise ).detach()
    
    def cost( self , X , y ):

        y_hat = self.predict( X )
        y = self.check_device( y )
        return ( ( y - y_hat )**2 ).mean()
    
    #------------------------------------------------------------------------------------------------------------------------

    def manage_cost( self , verbose ):

        X , y = self.ref
        cost = self.cost( X , y )
        if self.cuda:
            cost = cost.cpu()
        cost = cost.numpy()

        pos = self.iter_counter
        if verbose:
            print( "cost at iter {}:".format( pos ), cost )
        
        return pos , cost

    def iter_step( self , X_sample , y_sample , noise , opm ):

        mu , sigma = self.mod( X_sample )
        y_hat = mu + sigma*noise

        sqrd_err = ( y_sample - y_hat )**2 
        kl_err = -( sigma + mu**2 - 1 - torch.log( sigma ) )/2
        loss = ( sqrd_err + kl_err ).mean()

        opm.zero_grad()
        loss.backward()
        opm.step()

    def fit( self , X , y , iters = 100 , batch_size = 20 , lr = 1e-3 , verbose = True , ticks = 10 ):

        X = self.check_device( X )
        y = self.check_device( y )

        h = X.shape[ 0 ]
        idx = idx_generator( h , batch_size )

        self.mod.train()
        
        opm = torch.optim.Adam( self.mod.parameters() , lr = lr )

        for it in range( iters ):

            X_list , y_list = [] , []
            for i in idx.pop():
                X_list.append( X[ i ].unsqueeze( 0 ) ) 
                y_list.append( y[ i ] )
            X_sample = torch.cat( X_list ) 
            y_sample = self.check_device( torch.tensor( y_list ) )
            noise = self.sample_noise( batch_size )

            self.iter_step( X_sample , y_sample , noise , opm )

            if it == 0:
                costs = []

            if self.iter_counter == 0:
                self.ref = ( X_sample , y_sample )
                
            self.iter_counter += 1
            if ( self.iter_counter%ticks == 0 or self.iter_counter == 1 ):
                pos , cost = self.manage_cost( verbose )
                costs.append(( pos , cost ) )
        
        return costs


