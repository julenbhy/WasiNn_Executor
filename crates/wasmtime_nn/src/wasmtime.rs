use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;


use common::{ActionCapabilities, WasmRuntime};
use wasmtime::{ Engine, Linker, Module, Store, InstancePre, Instance };
use common::nn_utils::{ NnWasmCtx, link_host_functions, create_store, 
    pass_input, retrieve_result, download_model, pass_model, download_inputs };

pub struct WasmtimeRuntime {
    pub engine: Engine,
    pipelines: Arc<Mutex<HashMap<String, WasiNnPipeline>>>,
}

impl Default for WasmtimeRuntime {
    fn default() -> Self {
        WasmtimeRuntime {
            engine: Engine::default(),
            pipelines: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Clone for WasmtimeRuntime {
    fn clone(&self) -> Self {
        WasmtimeRuntime {
            engine: self.engine.clone(),
            pipelines: Arc::clone(&self.pipelines),
        }
    }
}

impl WasmRuntime for WasmtimeRuntime {

    /// Receives a container ID, capabilities, and a precompiled module,
    /// Capabilities include model URLs for downloading the multiple model parts of a model
    /// A pipeline is created. Each model part is executed on a separate thread.
    /// The threads will keep running for serving /run requests.
    /// Each pipeline is stored in a HashMap indexed by the container ID.
    fn initialize(
        &self,
        container_id: String,
        capabilities: ActionCapabilities,
        module: Vec<u8>,
    ) -> anyhow::Result<()> {

        // Deserialize WASM module
        let module = unsafe { Module::deserialize(&self.engine, &module)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize module {}: {}", container_id, e))? };

        // Link host functions
        let mut linker: Linker<NnWasmCtx> = Linker::new(&self.engine);
        link_host_functions(&mut linker)?;

        // Pre-instantiate
        let instance_pre = linker.instantiate_pre(&module)?;

        // Get model URLs
        let model_urls = capabilities.model_urls.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model URLs required"))?;

        // Setup pipeline
        let pipeline = WasiNnPipeline::setup_pipeline(&self.engine, instance_pre, model_urls)?;

        // Store pipeline
        self.pipelines.lock().unwrap().insert(container_id, pipeline);
        Ok(())
    }

    /// Sends input parameters to the first stage of the pipeline, which is already running.
    /// The parameters are passed to the first stage of the pipeline.
    /// Collects the output from the last stage and returns it as a JSON value.
    fn run(
        &self,
        container_id: &str,
        mut parameters: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {

        let pipelines = self.pipelines.lock().unwrap();
        let pipeline = pipelines.get(container_id)
            .ok_or_else(|| anyhow::anyhow!("Pipeline not found: {}", container_id))?;

        download_inputs(&mut parameters)?;
        let result = pipeline.run_pipeline(parameters)?;
        Ok(result)
    }

    /// Destroys the pipeline associated with the given container ID.
    /// This function is called when the container is no longer needed.
    /// It cleans up resources and stops any running threads.
    fn destroy(
        &self,
        container_id: &str,
    ) {
        if let Some(pipeline) = self.pipelines.lock().unwrap().remove(container_id) {
            pipeline.stop_pipeline();
        }
        println!("Destroyed: {}", container_id);
    }
}

// Input and output channels for each model stage
// The first stage receives JSON input, the last stage outputs JSON,
// and the middle stages receive and output tensors.
use std::{sync::mpsc::{channel, Receiver as StdReceiver, Sender as StdSender}, thread::JoinHandle};
use wasmtime_wasi_nn::Tensor;
enum InputChannel {
    Json(StdReceiver<serde_json::Value>),
    Tensor(StdReceiver<Tensor>),
}
enum OutputChannel {
    Json(StdSender<serde_json::Value>),
    Tensor(StdSender<Tensor>),
}


/// Launches threads for each model stage.
/// Each thread will run a model stage, passing the output of one stage as input to the next.
/// The function performs the following steps:
/// 1. Extract the number of model stages from the parameters.
/// 2. Spawn a thread for each model stage, passing the input and output channels.
/// 3. Each thread will instantiate the wasm module for its model stage.
/// 4. Each step will keep listening on its input channel, processing, and sending output to its output channel.
/// Uses channels to communicate between threads and handles the input and output of each stage.
/// The function is designed to handle multiple model stages, where each stage can run in its own thread.
/// The input to the first stage is JSON, and the output of the last stage is also JSON.
/// The middle stages receive tensors and output tensors to the next stage.
pub struct WasiNnPipeline {
    steps: Vec<JoinHandle<anyhow::Result<()>>>,
    input_tx: StdSender<serde_json::Value>,
    output_rx: StdReceiver<serde_json::Value>,
}
impl WasiNnPipeline {
    pub fn setup_pipeline(
        engine: &Engine,
        instance_pre: InstancePre<NnWasmCtx>,
        model_urls: &[String]
    ) -> anyhow::Result<WasiNnPipeline> {
        println!("\x1b[31mWASMTIME\x1b[0m Pipeline: {:?}", model_urls);

        let (initial_tx, initial_rx) = channel::<serde_json::Value>();
        let mut prev_rx_json = Some(initial_rx);
        let (final_tx, final_rx) = channel::<serde_json::Value>();
        let mut prev_tx_json = Some(final_tx);

        let mut prev_rx_tensor: Option<StdReceiver<Tensor>> = None;
        let mut handles = Vec::new();

        for (i, url) in model_urls.iter().enumerate() {
            let (in_chan, out_chan, next_rx_tensor) = setup_stage_channels(
                i, model_urls.len(), &mut prev_rx_json, &mut prev_tx_json, prev_rx_tensor.take()
            )?;

            let engine_c = engine.clone();
            let inst_c = instance_pre.clone();
            let url_c = url.clone();
            let handle = std::thread::spawn(move || {
                setup_pipeline_step(i, &engine_c, inst_c, url_c, in_chan, out_chan)
            });
            handles.push(handle);
            prev_rx_tensor = Some(next_rx_tensor);
        }

        Ok(WasiNnPipeline { steps: handles, input_tx: initial_tx, output_rx: final_rx })
    }

    pub fn run_pipeline(&self, params: serde_json::Value) -> anyhow::Result<serde_json::Value> {
        self.input_tx.send(params)?;
        let out = self.output_rx.recv()?;
        Ok(out)
    }

    pub fn stop_pipeline(self) {
        drop(self.input_tx);
        for handle in self.steps {
            if let Err(e) = handle.join() {
                println!("Error stopping pipeline thread: {:?}", e);
            }
        }
    }
}


/// Runs on a thread for each model stage
/// This function is responsible for instantiating the WASM module,
/// passing the model, setting up the wasi_nn context.
/// Then it runs the pipeline logic by calling run_pipeline_stage.
fn setup_pipeline_step(
    stage_idx: usize,
    engine: &Engine,
    instance_pre: InstancePre<NnWasmCtx>,
    model_url: String,
    input_channel: InputChannel,
    output_channel: OutputChannel,
) -> anyhow::Result<()> {

    // Create a store for the WASM instance
    let mut store = create_store(&engine)?;

    // Obtain the instance from the preinstance
    let instance = instance_pre.instantiate(&mut store)
        .map_err(|e| anyhow::anyhow!("Step {}: Failed to instantiate WASM module: {}", stage_idx, e))?;

    // Download the model from the provided URL
    let model_bytes = download_model(&model_url)
        .map_err(|e| anyhow::anyhow!("Step {}: Failed to download model from {}: {}", stage_idx, model_url, e))?;

    // Write the model to the WASM memory
    pass_model(&instance, &mut store, model_bytes)
        .map_err(|e| anyhow::anyhow!("Step {}: Failed to pass model to WASM instance: {}", stage_idx, e))?;

    // Call the "build_context" function in the WASM instance
    instance.get_typed_func::<(), u32>(&mut store, "build_context")
        .map_err(|_| anyhow::anyhow!("Step {}: Failed to get build_context from model: {}", stage_idx, model_url))?
        .call(&mut store, ())?;

    println!("\x1b[31mWASMTIME\x1b[0m Step {}: Context built successfully", stage_idx);

    // Run the model stage with the input and output channels
    run_model_stage(stage_idx, &instance, &mut store, input_channel, output_channel)
        .map_err(|e| anyhow::anyhow!("Step {}: Failed to run model stage: {}", stage_idx, e))?;

    Ok(())
}


/// The functions performs the following steps:
/// 1. Listen for input on the input channel.
///    If it is the first stage, listen on the initial JSON channel and write the input to the WASM memory.
///    If it is not the first stage, listen for input tensors from the previous stage and write them to the WasiNnCtx.
/// 2. Call the "inference" function in the WASM instance.
/// 3. If it is the last stage, retrieve the inference output from the WASM memory and send it as JSON output.
///    If it is not the last stage, retrieve the output tensors from the WasiNnCtx and send to the next stage's input channel.
fn run_model_stage(
    stage_idx: usize,
    instance: &Instance,
    mut store: &mut Store<NnWasmCtx>,
    mut input_channel: InputChannel,
    output_channel: OutputChannel,
) -> anyhow::Result<()> {
    // Main inference loop
    loop {
        // 3. Receive input for this stage
        match &mut input_channel {
            InputChannel::Json(receiver) => { // The first stage receives JSON input
                println!("\x1b[31mWASMTIME\x1b[0m Step {}: Waiting for JSON input", stage_idx);
                match receiver.recv() {
                    Ok(mut value) => { // Received JSON input
                        println!("\x1b[31mWASMTIME\x1b[0m Step {}: Received JSON input", stage_idx);

                        pass_input(instance, &mut store, &value)?;
                        
                        if let Err(e) = instance.get_typed_func::<(), u32>(&mut store, "preprocess_export")?
                            .call(&mut store, ()) {
                            eprintln!("preprocess failed: {:?}", e);
                            break;
                        }
                        
                    }
                    Err(e) => { // Input channel closed → end of processing
                        println!("\x1b[31mWASMTIME\x1b[0m Step {}: Input channel closed: {:?}", stage_idx, e);
                        break;
                    }
                }
            },
            InputChannel::Tensor(receiver) => { // Middle and last stages receive tensor input
                println!("\x1b[31mWASMTIME\x1b[0m Step {}: Waiting for tensor input", stage_idx);
                match receiver.recv() {
                    Ok(tensor) => { // Received tensor input
                        println!("\x1b[31mWASMTIME\x1b[0m Step {}: Received tensor input", stage_idx);
                        store.data_mut().wasi_nn().set_input_tensor(0, 0, tensor)?;
                    }
                    Err(e) => { // Input channel closed → end of processing
                        println!("\x1b[31mWASMTIME\x1b[0m Step {}: Input channel closed: {:?}", stage_idx, e);
                        break;
                    }
                }
            },
        }

        // 4. Run inference
        println!("\x1b[31mWASMTIME\x1b[0m Step {}: Running inference", stage_idx);

        // Same but break the loop in case of an error
        if let Err(e) = instance.get_typed_func::<(), u32>(&mut store, "compute")?
            .call(&mut store, ()) {
            eprintln!("STAGE {}: Inference failed: {:?}", stage_idx, e);
            break;
        }


        // 5. Send output to next stage
        match output_channel {
            OutputChannel::Tensor(ref sender) => { // Extract tensor from WASI-NN and send it to the next stage
                println!("\x1b[31mWASMTIME\x1b[0m Step {}: Sending tensor output", stage_idx);
                let output_tensor = store.data_mut().wasi_nn().get_output_tensor(0, 0)?;
                sender.send(output_tensor)?;
            }
            OutputChannel::Json(ref sender) => { // Retrieve final JSON output and send it to the final channel
                println!("\x1b[31mWASMTIME\x1b[0m Step {}: Sending JSON output", stage_idx);

                // The parameters are also passed to the instance in case the user needs
                // any of them in the postprocess function.
                // The inputs have been cleared to avoid passing them to the postprocess function
                //parameters["model_index"] = Value::Number(serde_json::Number::from(stage_idx));
                //pass_input(instance, &mut store, &parameters)?;
                
                // Call the postprocess function to finalize the output
                instance.get_typed_func::<(), u32>(&mut store, "postprocess_export")?
                    .call(&mut store, ())?;

                // Retrieve the final output JSON from the WASI-NN context
                let output_json = retrieve_result(instance, &mut store)?;

                sender.send(output_json)?;
            }
        }
    }

    drop(output_channel); // Close the output channel to propagate end of processing
    println!("\x1b[31mWASMTIME\x1b[0m Step {}: Finished processing", stage_idx);

    Ok(()) 
}




/// Sets up the input and output channels for each stage of the model pipeline.
/// This function determines the input and output channels based on the stage index.
/// It creates a new channel for the next stage and returns the appropriate input and output channels.
fn setup_stage_channels(
    stage_idx: usize,
    num_stages: usize,
    initial_receiver: &mut Option<StdReceiver<serde_json::Value>>,
    final_sender: &mut Option<StdSender<serde_json::Value>>,
    previous_receiver: Option<StdReceiver<Tensor>>,
) -> anyhow::Result<(InputChannel, OutputChannel, StdReceiver<Tensor>)> {

    // Special case: only one stage
    if num_stages == 1 {
        let input = InputChannel::Json(initial_receiver.take().unwrap());
        let output = OutputChannel::Json(final_sender.take().unwrap());
        // We still need to return a receiver, but it's unused.
        let (_dummy_sender, dummy_receiver) = channel::<Tensor>();
        return Ok((input, output, dummy_receiver));
    }
    
    // Create a new channel for the next stage
    let (next_sender, next_receiver) = channel::<Tensor>();

    // Determine input and output channels based on the stage index
    let (input_channel, output_channel) = match stage_idx {
        0 => ( // The first stage receives JSON input and outputs tensors
            InputChannel::Json(initial_receiver.take().unwrap()),
            OutputChannel::Tensor(next_sender),
        ),
        n if n == num_stages - 1 => ( // The last stage receives tensors and outputs JSON
            InputChannel::Tensor(previous_receiver.unwrap()),
            OutputChannel::Json(final_sender.take().unwrap()),
        ),
        _ => ( // Middle stages receive tensors and output tensors
            InputChannel::Tensor(previous_receiver.unwrap()),
            OutputChannel::Tensor(next_sender),
        ),
    };

    Ok((input_channel, output_channel, next_receiver))
}

