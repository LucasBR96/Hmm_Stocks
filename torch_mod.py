import torch
import torch.nn as net
import torch.nn.functional as net_fun

import numpy

class gaussian_Em_model( net.Module ):

    def __init__( self , s ):
        '''
        The emission model for the continuous hmm
        for every state a simple gaussian
        '''

        super( gaussian_Em_model , self ).__init__()

        self.num_state = s
        self.mu = net.Parameter( torch.rand( s ).unsqueeze( 0 ) )
        self.sigma = net.Parameter( torch.rand( s ).unsqueeze( 0 ) )
    
    def forward( self , M ):

        '''
        M -> tensor of floats , one dim
        returns P( x | mu[ t ] , sigma[ s ] ) for every x in M and s in num_state
        '''

        if M.ndim == 1:
            M = M.unsqueeze( 1 )

        z = ( M - self.mu )/torch.abs( self.sigma )
        norm = torch.distributions.Normal( 0 , 1 )

        # P( x | mu , sigma ) = P( ( x - mu )/sigma | 0 , 1 )
        return torch.exp( norm.log_prob( z ) )

        
class Tr_model( net.Module ):

    def __init__( self , s ):
        '''
        Transition Between two latent variables
        '''
        super( Tr_model , self ).__init__()

        self.num_state = s
        # A[ s , s' ] = P( S[t + 1 ] = s' | S[ t ] = s ) for any t in [ 0 : inf ]
        self.A = net.Parameter( torch.rand( s , s ) )
    
    def forward( self , Mat ):

        '''
        Mat -> Two dimentional ternsors
        Mat.shape[ 0 ] must be equal to self.num_state
        '''

        A = net_fun.softmax( self.A , dim = 1 )
        return Mat@A


class gaussian_hmm( net.Module ):

    def __init__( self , s ):

        super( gaussian_hmm , self ).__init__()

        self.num_state = s
        
        # f[ s ] = P( S[ t ] = s ) for any t
        self.f = net.Parameter( torch.rand( s ) )
        self.Em_mod = gaussian_Em_model( s )
        self.Tr_mod = Tr_model( s )

    def forward( self , H ):

        '''
        H -> two dim tensor
        every row is A sequence of values
        all sequences have the same lenght.

        returns alpha, where alpha( s , M ) is the probability of the last hidden state be S given the sequence M
        returns c , where c( M , t ) = P( M[ t ] | M[ t -1 ] , M[ t -2 ] , ..... M[ 0 ] )
        '''
        c_list = []
        for t in range( H.shape[ 1 ] ):

            B = self.Em_mod( H[ : , t ] )
            if t == 0:
                mul = net_fun.softmax( self.f , dim = 0)
            else:
                mul = self.Tr_mod( alpha )
            
            alpha_prime = mul*B
            c = alpha_prime.sum( axis = 1 , keepdim = True)
            alpha = alpha_prime/c
            c_list.append( c )
        return alpha , torch.cat( c_list , dim = 1 )

class next_state_pipeline( net.Module ):

    def __init__( self , s , hidden_layers ):

        super( next_state_pipeline , self ).__init__()

        self.num_state = s
        self.hmm = gaussian_hmm( s )

        # mods = net.ModuleList()
        # mods.append( net.Linear( s , hidden_layers[ 0 ] ) )
        # mods.append( net.ReLU() )
        # for i in range( len( hidden_layers ) ):

        #     a = hidden_layers[ i ]
        #     if i + 1 < len( hidden_layers ):
        #         b = hidden_layers[ i + 1 ]
        #     else:
        #         b = 2

        #     mods.append( net.Linear( a , b ) )
        #     if i + 1 < len( hidden_layers ):
        #         mods.append( net.ReLU() )
        assert( isinstance( hidden_layers , list ) )
        layers = [ s ] + hidden_layers + [ 2 ]
        n = len( layers )
        mods = net.ModuleList()
        for i in range( n - 1 ):

            a = layers[ i ]
            b = layers[ i + 1 ]
            mods.append( net.Linear( a , b ) )

            # if it is not the last layer
            if i != n - 2:
                mods.append( net.ReLU() )

        self.net = mods
    
    def forward( self , H ):

        '''
        H is the same input described in gaussian_hmm.foward()
        returns , for every sequence M in H , mu and sigma, used in the distribution of 
        probabilities for the next observation

        Explaining in bigger detail, lets consider one sequence M
        1 - M is passed through the hmm. if len( M ) = n , then alpha( s ) = P( S[ n ] = s | M )
        2 - alpha is passed through the trans model, now alpha( s ) = P( S[ n ] = s | M )
        3 - finally, alpha is fed to the neural network, resulting  in mu and sigma
        '''

        alpha , c  = self.hmm( H ) #1
        A = self.hmm.Tr_mod #2
        alpha = A( alpha )
        for mod in self.net:#3
            alpha = mod( alpha )
        param = alpha.transpose( 0 , 1 )

        #          mu      ,   sigma
        return param[ 0 ] , torch.abs( param[ 1 ] )

if __name__ == "__main__":

    A = next_state_pipeline( 5 , [ 5 ] )
    print( *A.hmm.parameters() )
    # print()
    # print( *A.net.parameters() )

