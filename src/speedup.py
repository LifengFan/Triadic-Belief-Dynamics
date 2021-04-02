import torch.multiprocessing


# This model object contains the loaded ML model.
# Each Python child process will have its independent copy of the Model object
model = None

def init_worker(GPUs, model_param_path):
    """ This gets called when the worker process is created
    Here we initialize the worker process by loading the ML model into GPU memory.
    Each worker process will pull a GPU ID from a queue of available IDs (e.g. [0, 1, 2, 3]) and load the ML model to that GPU
    This ensures that multiple GPUs are consumed evenly."""
    global model
    if not gpus.empty():
        gpu_id = gpus.get()
        logger.info("Using GPU {} on pid {}".format(gpu_id, os.getpid()))
        ctx = mx.gpu(gpu_id)
    else:
        logger.info("Using CPU only on pid {}".format(os.getpid()))
        ctx = mx.cpu()
    model = MXNetModel(param_path=model_param_path,
                       input_shapes=INPUT_SHAPE,
                       context=ctx)


def process_input_by_worker_process(input_file_path):
    """ The work to be done by one worker for one input.
    Performs input transformation then runs inference with the ML model local to the worker process. """
    # input transformation (read from file, resize, swap axis, etc.) this happens on CPU
    transformed_input = transform_input(input_file_path, reshape=(IMAGE_DIM, IMAGE_DIM))
    # run inference (forward pass on neural net) on the ML model loaded on GPU
    output = model.infer(transformed_input)
    return {"file": input_file_path,
            "result": output }


def run_inference_in_process_pool(model_param_path, input_paths, num_process, output_path):
    """ Create a process pool and submit inference tasks to be processed by the pool """
    # If GPUs are available, create a queue that loops through the GPU IDs.
    # For example, if there are four worker processes and 4 GPUs, the queue contains [0, 1, 2, 3]
    # If there are four worker processes and 2 GPUs the queue contains [0, 1, 0, 1]
    # This can be done when GPU is underutilized, and there's enough memory to hold multiple copies of the model.
    gpus = detect_cores() # If there are n GPUs, this returns list [0,1,...,n-1].
    gpu_ids = multiprocessing.Queue()
    if len(gpus) > 0:
        gpu_id_cycle_iterator = cycle(gpus)
        for i in range(num_process):
            gpu_ids.put(next(gpu_id_cycle_iterator))

    # Initialize process pool
    process_pool = multiprocessing.Pool(processes=num_process, initializer=init_worker, initargs=(gpu_ids, model_param_path))

    # Feed inputs process pool to do transform and inference
    pool_output = process_pool.map(process_input_by_worker_process, input_paths)

