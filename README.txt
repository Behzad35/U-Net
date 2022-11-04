(Building and running the program) 
--------------------------------------------------------------------
To build the program, call 'make'.
The excutable is called 'unet'.

To run the program, 3 command line arguments are needed:
$ ./unet [imgsize] [batchsize] [num_of_batches]
For example:
$ ./unet 512 8 16


(Variables to modify in main.cpp)
--------------------------------------------------------------------
To enable verbose mode, set verbose to 'true' in the main file.

Other parameters such as number of layers, channel size, number of epochs, learning rate, etc. can be modified in the main file accordingly. (Note: number of layers must be at least 2)


(Repository structure)
--------------------------------------------------------------------
(src): source files

(Weights): kernel weights extracted from the python code

(training_data): images and annotations in .png and .csv formats (.csv files were used to run the program on the cluster)

(backup_kernel_csv): folder to store kernel weights (for debugging, or if computations need to be stopped and carried on later.)





