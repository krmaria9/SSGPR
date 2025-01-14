import casadi as ca
import numpy as np
from model.tbf import TBF
from optimizer.minimize import minimize
import numpy.linalg as LA
import pickle
import os
import matplotlib.pyplot as plt

class SSGPR:
    """
    Sparse Spectrum Gaussian Process Regression (SSGPR) object.

    Parameters
	----------
	num_basis_functions : int
	    Number of trigonometric basis functions used to construct the design matrix.

	optimize_freq : bool
	    If true, the spectral points are optimized.
    """
    def __init__(self, num_basis_functions=100, optimize_freq=True):
        self.m             = num_basis_functions  # number of basis functions to use
        self.optimize_freq = optimize_freq        # optimise spectral frequencies?

    def add_data(self, X_train, Y_train, X_test=None, Y_test=None):
        """
        Add data to the SSGPR object.

        Parameters
        ----------
        X_train : numpy array of shape (N, D)
            Training data inputs where N is the number of data training points and D is the
            dimensionality of the training data.

        Y_train : numpy array of shape (N, 1)
            Training data targets where N is the number of data training points.

        X_test : numpy array of shape (N, D) - default is None
            Test data inputs where N is the number of data test points and D is the
            dimensionality of the test data.

        Y_test : numpy array of shape (N, 1) - default is None
            Test data inputs where N is the number of data test points.
        """
        assert X_train.ndim == 2, "X_train must be 2-dimensional of shape (N, D)"
        assert Y_train.ndim == 2, "Y_train must be 2-dimensional of shape (N, 1)"
        self.N, self.D  = X_train.shape
        self.X          = X_train
        # self.Y_mean     = Y_train.mean()
        self.Y          = Y_train # - self.Y_mean # subtract target mean to make zero mean targets (added back later)
        if (X_test is not None) and (Y_test is not None):
            assert X_test.ndim == 2, "X_test must be 2-dimensional of shape (N, D)"
            assert Y_test.ndim == 2, "Y_test must be 2-dimensional of shape (N, 1)"
            self.X_test     = X_test
            self.Y_test     = Y_test
        self.tbf = TBF(self.D, self.m)          # initialize the Trigonometric Basis Function object
        self.optimized  = False

    def update_parameters(self, params):
        """
        Updates the SSGPR and TBF object parameters.

        Parameters
        ----------
        params : numpy array of shape (n,)
            n is length D + 2 + num_basis_functions. D is the data dimensionality
            params[0:D] - TBF lengthscales
            params[D]   - TBF amplitude
            params[D+1] - SSGPR noise variance
            params[D+2] - TBF spectral frequencies
        """
        self.tbf.update_lengthscales(np.exp(params[:self.D]))  # update TBF lengthscales
        self.tbf.update_amplitude(np.exp(2*params[self.D]))    # update TBF amplitude
        self.var_n = np.exp(2*params[self.D + 1])              # update noise variance
        self.tbf.update_frequencies(params[self.D + 2:])       # update the TBF spectral frequencies

    # function to make predictions on training points x (x must be in array format)
    def predict(self, Xs, Ys=None):
        phi = self.tbf.design_matrix(self.X)
        phi_star = self.tbf.design_matrix(Xs)

        # Use phi, y to compute alpha
        if Ys is None:
            A = (self.tbf.var_0/self.m) * phi.T @ phi + self.var_n * np.eye(2*self.m)
            R = LA.cholesky(A).T

            alpha = self.tbf.var_0 / self.m * LA.solve(R, LA.solve(R.T, phi.T @ self.Y))
            var =(self.var_n * (1 + self.tbf.var_0/self.m * np.sum((phi_star @ LA.inv(R))**2, axis=1))).reshape(-1,1)
            stddev = np.sqrt(var) # predictive std dev

        # Use phi_star, ys to compute alpha
        else:
            A = (self.tbf.var_0/self.m) * phi_star.T @ phi_star + self.var_n * np.eye(2*self.m)
            R = LA.cholesky(A).T

            alpha = self.tbf.var_0 / self.m * LA.solve(R, LA.solve(R.T, phi_star.T @ Ys))
            var =(self.var_n * (1 + self.tbf.var_0/self.m * np.sum((phi_star @ LA.inv(R))**2, axis=1))).reshape(-1,1)
            stddev = np.sqrt(var) # predictive std dev
    
        mu = (phi_star @ alpha).reshape(-1,1) # predictive mean

        return mu, stddev, alpha
    
    def evaluate_prediction(self, Xs, alpha):
        phi_star = self.tbf.design_matrix(Xs)

        mu = (phi_star @ alpha).reshape(-1,1) # predictive mean

        return mu

    def predict_symbolic(self, Xs, type_function=False, num_samples=1):
        """
        Make symbolic predictions based on input Xs.

        Parameters are analogous to the predict() function.
        """

        # Calculate some useful constants
        phi = self.tbf.design_matrix_symbolic(self.X)
        phi_star = self.tbf.design_matrix_symbolic(Xs)
        A = (self.tbf.var_0/self.m) * ca.mtimes(phi.T, phi) + self.var_n * ca.SX.eye(2*self.m)
        R = ca.chol(A) # equivalent to LA.cholesky(A).T
        
        alpha = self.tbf.var_0 / self.m * ca.solve(R, ca.solve(R.T, ca.mtimes(phi.T, self.Y)))
        # mu = ca.mtimes(phi_star, alpha) + self.Y_mean  # predictive mean
        mu = ca.mtimes(phi_star, alpha)  # predictive mean

        # Instead of matrix inversion, we use solve to compute the solution of a system of linear equations
        invR_phi_star = ca.mtimes(phi_star, ca.solve(R, ca.SX.eye(2*self.m)))
        var_components = invR_phi_star**2
        var_summed = ca.sum2(var_components)
        var = self.var_n * (1 + self.tbf.var_0/self.m * var_summed)
        stddev = ca.sqrt(var)

        if type_function:
            f_mu = ca.Function('f5', [Xs], [mu])
            f_stddev = ca.Function('f6', [Xs], [stddev])
            return f_mu, f_stddev
        else:
            return mu, stddev
        
    # function that computes the marginal likelihood
    def negative_marginal_log_likelihood(self, params):
        """
        Calculates the negative marginal log likelihood.

        Parameters
        ----------
        params : numpy array of shape (n,)
            n is length D + 2 + num_basis_functions. D is the data dimensionality
            params[0:D] - TBF lengthscales
            params[D]   - TBF amplitude
            params[D+1] - SSGPR noise variance
            params[D+2] - TBF spectral frequencies

        Return
        ------
        nmll : numpy array of shape (1, 1)
            negative marginal log likelihood
        """
        self.update_parameters(params)
        phi = self.tbf.design_matrix(self.X)
        R = LA.cholesky((self.tbf.var_0/self.m) * phi.T @ phi + self.var_n * np.eye(2*self.m)).T
        Rtiphity = LA.solve(R.T, phi.T @ self.Y)

        #negative marginal log likelihood
        nmll = (self.Y.T @ self.Y - (self.tbf.var_0/self.m)*np.sum(Rtiphity**2)) / (2*self.var_n)
        nmll += np.log(np.diag(R)).sum() + ((self.N/2) - self.m) * np.log(self.var_n)
        nmll += (self.N/2)*np.log(2*np.pi)

        return nmll

    # function that computes gradients
    def gradients(self, params):
        """
        Calculates the gradients of the negative marginal log likelihood with respect to
        the lengthscales, amplitude, noise variance and spectral frequencies.

        Parameters
        ----------
        params : numpy array of shape (n,)
            n is of length D + 2 + num_basis_functions. D is the data dimensionality.
            params[0:D] - TBF lengthscales
            params[D]   - TBF amplitude
            params[D+1] - SSGPR noise variance
            params[D+2] - TBF spectral frequencies

        Return
        ------
        grad : numpy array of shape (D + 2 + num_basis_functions, )
            gradients of the negative marginal log likelihood with respect to
            the lengthscales, amplitude, noise variance and spectral frequencies.
        """
        self.update_parameters(params)
        grad = np.zeros((self.D + 2 + self.D*self.m, 1))
        phi = self.tbf.design_matrix(self.X)
        R = LA.cholesky((self.tbf.var_0 / self.m) * phi.T @ phi + self.var_n * np.eye(2 * self.m)).T

        PhiRi = phi @ LA.inv(R)
        RtiPhit = PhiRi.T
        Rtiphity = RtiPhit @ self.Y

        A = np.concatenate(((self.Y/self.var_n - (self.tbf.var_0/(self.var_n*self.m)) * PhiRi @ Rtiphity).reshape(-1,1),
                            np.sqrt((self.tbf.var_0/(self.var_n*self.m)))*PhiRi),axis=1)
        diagfact = -(1/self.var_n) + np.sum(A**2, axis=1) 
        Aphi = A.T @ phi
        B = A @ Aphi[:,:self.m]*phi[:,self.m:] - A @ Aphi[:,self.m:]*phi[:,:self.m]                                    
        
        #DERIVATIVES START
        # derivatives wrt the lengthscales
        for d in range(self.D):
            grad[d] = -(0.5*2*self.tbf.var_0/self.m) * (self.X[:,d].T @ B @ self.tbf.W[:,d])

        # derivative wrt signal power hyperparameter
        grad[self.D]= (0.5*2*self.tbf.var_0/self.m) * (((self.N * self.m)/self.var_n) - np.sum(Aphi**2))

        # derivative wrt noise power hyperparameter
        grad[self.D+1] = -0.5*2*np.sum(diagfact)*self.var_n

        if self.optimize_freq:
            # derivatives wrt the representative frequencies
            for d in range(self.D):
                grad[self.D+2+d*self.m:self.D+2+(d+1)*self.m] = \
                    (0.5*2*(self.tbf.var_0/self.m)*(self.X[:,d].T @ B)/self.tbf.l[d,0]).reshape(-1,1)
        else:
            grad[self.D+2+d*self.m:self.D+2+(d+1)*self.m] = 0

        return grad[:,0]

    def objective_function(self, params):
        """
        Wrapper function for the negative marginal log likelihood and the gradients.

        Parameters
        ----------
        params : numpy array of shape (n,)
            n is of length D + 2 + num_basis_functions. D is the data dimensionality.
            params[0:D] - TBF lengthscales
            params[D]   - TBF amplitude
            params[D+1] - SSGPR noise variance
            params[D+2] - TBF spectral frequencies

        Return
        ------
        nmll : numpy array of shape (1, 1)
            negative marginal log likelihood

        grad : numpy array of shape (D + 2 + num_basis_functions, )
            gradients of the negative marginal log likelihood with respect to
            the lengthscales, amplitude, noise variance and spectral frequencies.

        """
        nmll = self.negative_marginal_log_likelihood(params)
        grad = self.gradients(params)
        return nmll, grad

    def optimize(self, save_dir, restarts=3, maxiter=1000, verbose=True):
        """
        Optimize the marginal log likelihood with conjugate gradients minimization.

        Parameters
        ----------
        restarts : int
            The number of restarts for the minimization process. Defaults to 3.
            - The first minimization attempt is initialized with:
                - lengthscales: half of the ranges of the input dimensions
                - amplitude: variance of the targets
                - noise variance: variance of the targets divided by four
                - spectral points: choose the best from 100 random initializations
            - Subsequent restarts have random initialization.

        maxiter : int
            The maximum number of line searches for the minimization process.
            Defaults to 1000.

        verbose : bool
            If True, prints minimize progress.

        Return
        ------
        Xs : numpy array - Shape : (D + 2 + num_basis_functions, 1)
		    The found solution.

	    global_opt : numpy array - Shape : (i, 1 + D + 2 + num_basis_functions)
            Convergence information from the best restart. The first column is the negative marginal log
            likelihood returned by the function being minimized. The next D + 2 + num_basis_functions
            columns are the guesses during the minimization process. i is the number of
            linesearches performed.
        """
        if verbose:
            print('***************************************************')
            print('*              Optimizing parameters              *')
            print('***************************************************')

        global_opt = np.inf

        # initialise hyper-parameters from data
        lengthscales = np.log((np.max(self.X, 0) - np.min(self.X, 0)).T / 2)
        lengthscales[lengthscales < -1e2] = -1e2
        amplitude = 0.5*np.log(np.var(self.Y))
        noise_variance = 0.5*np.log(np.var(self.Y) / 4)

        for restart in range(restarts):

            # initialise the spectral points: try 100 inits and keep best
            nmll=np.inf
            for i in range(100):
                spectral_sample = np.random.normal(0,1,size=(self.m*self.D))
                params = np.hstack((lengthscales, amplitude, noise_variance, spectral_sample))
                nmllc = self.negative_marginal_log_likelihood(params)
                if nmllc < nmll:
                    spectral_points = spectral_sample
                    nmll = nmllc
                    if verbose:
                        print("Selecting new spectral points ...")
            if verbose:
                print("Selected spectral points.")

            # minimize
            X0 = np.hstack((lengthscales, amplitude, noise_variance, spectral_points)).reshape(-1,1)
            Xs, convergence, _, fX = minimize(self.objective_function, X0, length=maxiter, verbose=verbose, concise=True)
            self.plot_convergence(save_dir + f"/conv_{restart}.png", fX)

            # check if the local optimum beat the current global optimum
            if convergence < global_opt:
                global_opt = convergence # update the global optimum
                self.Xs = Xs                   # save best solution
                which_restart = restart

            # print out optimization result if the user wants
            if verbose:
                print('restart # %i, negative log-likelihood = %.5f' %(restart+1,convergence))

            if restart < restarts-1: # randomize parameters for next iteration
                self.tbf.update_amplitude(np.random.normal())
                self.tbf.update_lengthscales(np.random.normal(size=self.D))
                self.noise  = np.random.normal()

        self.update_parameters(self.Xs)
        self.optimized = True

        if verbose:
            print("Using restart # %i results:" % (which_restart+1))
            print("Negative log-likelihood: %.5f" % global_opt)
            self.print_hyperparams()

        return self.Xs, global_opt

    def print_hyperparams(self):
        """
        Prints the current values of the hyperparameters.
        """
        print("lengthscales: ", self.tbf.l)
        print("amplitude: ", self.tbf.var_0)
        print("noise variance: ", self.var_n)

    def evaluate_performance(self, save_dir, restarts=3, maxiter=1000):
        """
        Evaluates the performance of the predictive mean by calculating the
        normalized mean squared error (NMSE) and the mean negative log
        probability (MNLP) of the predictive mean against the test data.

        If optimize has not previously been called, it is called in this
        function.

        Test data must first be loaded with the add_data method.

        Parameters
        ----------
        restarts : int
            The number of restarts for the minimization process.
            - The first minimization attempt is initialized with:
                - lengthscales: half of the ranges of the input dimensions
                - amplitude: variance of the targets
                - noise variance: variance of the targets divided by four
                - spectral points: choose the best from 100 random initializations
            - Subsequent restarts have random initialization.

        Return
        ------
        NMSE : numpy.float64
            Normalized mean squared error (NMSE)

        MNLP : numpy.float64
            Mean negative log probability (MNLP)
        """
        if self.X_test is None or self.Y_test is None:
            raise Exception("No test data loaded in add_data!")

        if not self.optimized:
            self.optimize(save_dir, restarts=restarts, maxiter=maxiter, verbose=False)

        mu, stddev, _ = self.predict(self.X_test) # predict on the test points

        NMSE = ((mu - self.Y_test) ** 2).mean() / ((self.Y.mean() - self.Y_test) ** 2).mean()
        MNLP = -0.5 * (-((mu - self.Y_test) ** 2.) / stddev ** 2 - np.log(2 * np.pi) - np.log(stddev ** 2)).mean()

        return NMSE, MNLP

    def save(self, filename):
        """
        Save the current SSGPR instance to a pickle file.
        
        Parameters:
        - filename (str): The path of the file to save to.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

        # Save also scaled frequencies
        self.tbf.save_scaled_frequencies(filename.replace(".pkl", "_freqs.csv"))

    @staticmethod
    def load(filename):
        """
        Load an SSGPR instance from a pickle file.
        
        Parameters:
        - filename (str): The path of the file to load from.
        
        Returns:
        - SSGPR instance
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The specified file '{filename}' does not exist.")

        with open(filename, 'rb') as file:
            return pickle.load(file)

    def plot_convergence(self, path, convergence):
        N = convergence.shape[0]
        X = np.arange(N)
        Y = convergence[:,0]
        plt.figure()
        plt.plot(X, Y)
        plt.grid()
        plt.ylabel("Negative marginal log likelihood")
        plt.xlabel("Number of iterations")
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(path, dpi=300)
        plt.close()