use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;


use common::{ActionCapabilities, WasmRuntime};
use wasmtime::{ Engine, Linker, Module, Store, InstancePre, Instance };
use common::nn_utils::{ NnWasmCtx, link_host_functions, create_store, 
    pass_input, retrieve_result, download_model, pass_model, download_inputs };

pub struct WasmtimeRuntime {
    pub engine: Engine,
    pipelines: Arc<Mutex<HashMap<String, Arc<WasiNnPipeline>>>>,
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
        let pipeline = Arc::new(WasiNnPipeline::new(&self.engine, instance_pre, model_urls)?);

        // Store pipeline
        self.pipelines.lock().unwrap().insert(container_id.clone(), pipeline);

        println!("\x1b[31mWASMTIME\x1b[0m Initialized pipeline for container id '{}'", container_id);
        Ok(())
    }


    /// Sends input parameters to the first step of the pipeline, which is already running.
    /// The parameters are passed to the first step of the pipeline.
    /// Collects the output from the last step and returns it as a JSON value.
    fn run(
        &self,
        container_id: &str,
        mut parameters: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {


        println!("\x1b[31mWASMTIME\x1b[0m Running pipeline for container id '{}'", container_id);

        println!("\x1b[31mWASMTIME\x1b[0m Locking pipelines HashMap for container id '{}'", container_id);

        let pipelines = self.pipelines.lock().unwrap();

        println!("\x1b[31mWASMTIME\x1b[0m Getting pipeline for container id '{}'", container_id);

        let pipeline = pipelines.get(container_id)
            .ok_or_else(|| anyhow::anyhow!("Pipeline not found: {}", container_id))?
            .clone(); // Clone the Arc, not the pipeline itself

        println!("\x1b[31mWASMTIME\x1b[0m Downloading inputs for container id '{}'", container_id);

        download_inputs(&mut parameters)?;

        println!("\x1b[31mWASMTIME\x1b[0m Running pipeline for container id '{}'", container_id);
        let result = pipeline.run_pipeline(parameters)?;

        println!("\x1b[31mWASMTIME\x1b[0m Finished running pipeline for container id '{}'", container_id);
        Ok(result)
    }


    /// Destroys the pipeline associated with the given container ID.
    /// This function is called when the container is no longer needed.
    /// It cleans up resources and stops any running threads.
    fn destroy(
        &self,
        container_id: &str,
    ) {
        println!("\x1b[31mWASMTIME\x1b[0m Destroying pipeline for container id '{}'", container_id);
        if let Some(_pipeline) = self.pipelines.lock().unwrap().remove(container_id) {
            //pipeline.stop_pipeline();
        }
        println!("\x1b[31mWASMTIME\x1b[0m Destroyed pipeline for container id '{}'", container_id);
    }
}

// Input and output channels for each model step
// The first step receives JSON input, the last step outputs JSON,
// and the middle steps receive and output tensors.
use std::thread::JoinHandle;
use crossbeam::channel::{ Sender, Receiver, unbounded as channel };
use wasmtime_wasi_nn::Tensor;
pub(crate) enum InputChannel {
    Json(Receiver<serde_json::Value>),
    Tensor(Receiver<Tensor>),
}
pub(crate) enum OutputChannel {
    Json(Sender<serde_json::Value>),
    Tensor(Sender<Tensor>),
}




/// Launches threads for each model step.
/// Each thread will run a model step, passing the output of one step as input to the next.
/// The function performs the following steps:
/// 1. Extract the number of model steps from the parameters.
/// 2. Spawn a thread for each model step, passing the input and output channels.
/// 3. Each thread will instantiate the wasm module for its model step.
/// 4. Each step will keep listening on its input channel, processing, and sending output to its output channel.
/// Uses channels to communicate between threads and handles the input and output of each step.
/// The function is designed to handle multiple model steps, where each step can run in its own thread.
/// The input to the first step is JSON, and the output of the last step is also JSON.
/// The middle steps receive tensors and output tensors to the next step.
pub struct WasiNnPipeline {
    _steps: Vec<WasiNnPipelineStep>,
    input_tx: Sender<serde_json::Value>,
    output_rx: Receiver<serde_json::Value>,
}
impl WasiNnPipeline {
    pub fn new(
        engine: &Engine,
        instance_pre: InstancePre<NnWasmCtx>,
        model_urls: &[String]
    ) -> anyhow::Result<WasiNnPipeline> {
        println!("\x1b[31mWASMTIME\x1b[0m Pipeline: {:?}", model_urls);

        let (initial_tx, initial_rx) = channel::<serde_json::Value>();
        let mut prev_rx_json = Some(initial_rx);
        let (final_tx, final_rx) = channel::<serde_json::Value>();
        let mut prev_tx_json = Some(final_tx);

        let mut prev_rx_tensor: Option<Receiver<Tensor>> = None;
        let mut steps = Vec::new();

        for (i, url) in model_urls.iter().enumerate() {
            let (in_chan, out_chan, next_rx_tensor) = Self::setup_step_channels(
                i, model_urls.len(), &mut prev_rx_json, &mut prev_tx_json, prev_rx_tensor.take()
            )?;

            let engine_c = engine.clone();
            let instance_pre_c = instance_pre.clone();
            let url_c = url.clone();
            let step = WasiNnPipelineStep::new(
                i, engine_c, instance_pre_c, url_c, in_chan, out_chan
            );
            steps.push(step);
            prev_rx_tensor = Some(next_rx_tensor);
        }

        Ok(WasiNnPipeline { _steps: steps, input_tx: initial_tx, output_rx: final_rx })
    }


    pub fn run_pipeline(&self, params: serde_json::Value) -> anyhow::Result<serde_json::Value> {
        self.input_tx.send(params)?;
        let out = self.output_rx.recv()?;
        Ok(out)
    }


    pub fn _stop_pipeline(self) {
        drop(self.input_tx);
        for step in self._steps {
            step.join();
        }
    }


    /// Sets up the input and output channels for each step of the model pipeline.
    /// This function determines the input and output channels based on the step index.
    /// It creates a new channel for the next step and returns the appropriate input and output channels.
    fn setup_step_channels(
        step_idx: usize,
        num_steps: usize,
        initial_receiver: &mut Option<Receiver<serde_json::Value>>,
        final_sender: &mut Option<Sender<serde_json::Value>>,
        previous_receiver: Option<Receiver<Tensor>>,
    ) -> anyhow::Result<(InputChannel, OutputChannel, Receiver<Tensor>)> {

        // Special case: only one step
        if num_steps == 1 {
            let input = InputChannel::Json(initial_receiver.take().unwrap());
            let output = OutputChannel::Json(final_sender.take().unwrap());
            // We still need to return a receiver, but it's unused.
            let (_dummy_sender, dummy_receiver) = channel::<Tensor>();
            return Ok((input, output, dummy_receiver));
        }
        
        // Create a new channel for the next step
        let (next_sender, next_receiver) = channel::<Tensor>();

        // Determine input and output channels based on the step index
        let (input_channel, output_channel) = match step_idx {
            0 => ( // The first step receives JSON input and outputs tensors
                InputChannel::Json(initial_receiver.take().unwrap()),
                OutputChannel::Tensor(next_sender),
            ),
            n if n == num_steps - 1 => ( // The last step receives tensors and outputs JSON
                InputChannel::Tensor(previous_receiver.unwrap()),
                OutputChannel::Json(final_sender.take().unwrap()),
            ),
            _ => ( // Middle steps receive tensors and output tensors
                InputChannel::Tensor(previous_receiver.unwrap()),
                OutputChannel::Tensor(next_sender),
            ),
        };

        Ok((input_channel, output_channel, next_receiver))
    }
}






pub struct WasiNnPipelineStep {
    pub step_idx: usize,
    //pub model_url: String,
    //pub input_channel: InputChannel,
    //pub output_channel: OutputChannel,
    //pub instance: Option<Instance>,
    //pub store: Option<Store<NnWasmCtx>>,
    pub handle: Option<JoinHandle<anyhow::Result<()>>>,
}
impl WasiNnPipelineStep {
    /// Runs on a thread for each model step
    /// This function is responsible for instantiating the WASM module,
    /// passing the model, setting up the wasi_nn context.
    /// Then it runs the pipeline logic by calling run_pipeline_step.
    pub fn new(
        step_idx: usize,
        engine: Engine,
        instance_pre: InstancePre<NnWasmCtx>,
        model_url: String,
        input_channel: InputChannel,
        output_channel: OutputChannel,
    ) -> WasiNnPipelineStep {

        let handle = std::thread::spawn(move || {
            // 1. Crear el Store
            let mut store = create_store(&engine)?;

            // 2. Instanciar el módulo
            let instance = instance_pre.instantiate(&mut store)
                .map_err(|e| anyhow::anyhow!("Step {}: Failed to instantiate WASM module: {}", step_idx, e))?;

            // 3. Descargar el modelo
            let model_bytes = download_model(&model_url)
                .map_err(|e| anyhow::anyhow!("Step {}: Failed to download model: {}", step_idx, e))?;

            // 4. Pasar modelo al WASM
            pass_model(&instance, &mut store, model_bytes)
                .map_err(|e| anyhow::anyhow!("Step {}: Failed to pass model: {}", step_idx, e))?;

            // 5. build_context
            instance.get_typed_func::<(), u32>(&mut store, "build_context")?
                .call(&mut store, ())?;

            println!("\x1b[31mWASMTIME\x1b[0m Step {}: Context built", step_idx);

            // 6. Ejecutar el paso
            Self::run_model_step(step_idx, &instance, &mut store, input_channel, output_channel)
                .map_err(|e| anyhow::anyhow!("Step {}: Failed to run model step: {}", step_idx, e))?;

            Ok(())
        });

        WasiNnPipelineStep {
            step_idx,
            //model_url,
            //input_channel,
            //output_channel,
            //instance: None, // The thread has the ownership of the instance
            //store: None, // The thread has the ownership of the store
            handle: Some(handle), 
        }
    }


    pub fn join(self) {
        if let Some(handle) = self.handle {
            if let Err(e) = handle.join() {
                println!("Step {}: Error joining thread: {:?}", self.step_idx, e);
            }
        }
    }




    /// The functions performs the following steps:
    /// 1. Listen for input on the input channel.
    ///    If it is the first step, listen on the initial JSON channel and write the input to the WASM memory.
    ///    If it is not the first step, listen for input tensors from the previous step and write them to the WasiNnCtx.
    /// 2. Call the "inference" function in the WASM instance.
    /// 3. If it is the last step, retrieve the inference output from the WASM memory and send it as JSON output.
    ///    If it is not the last step, retrieve the output tensors from the WasiNnCtx and send to the next step's input channel.
    fn run_model_step(
        step_idx: usize,
        instance: &Instance,
        mut store: &mut Store<NnWasmCtx>,
        mut input_channel: InputChannel,
        output_channel: OutputChannel,
    ) -> anyhow::Result<()> {
        // Main inference loop
        loop {
            // 3. Receive input for this step
            match &mut input_channel {
                InputChannel::Json(receiver) => { // The first step receives JSON input
                    println!("\x1b[31mWASMTIME\x1b[0m Step {}: Waiting for JSON input", step_idx);
                    match receiver.recv() {
                        Ok(mut value) => { // Received JSON input
                            println!("\x1b[31mWASMTIME\x1b[0m Step {}: Received JSON input", step_idx);

                            value["model_index"] = serde_json::Value::Number(serde_json::Number::from(step_idx));


                            pass_input(instance, &mut store, &value)
                                .map_err(|e| {
                                    eprintln!("Failed to pass input to WASM instance: {:?}", e);
                                    anyhow::anyhow!("Step {}: Failed to pass input to WASM instance: {}", step_idx, e)
                                })?;
                                

                            let preprocess_fn = instance.get_typed_func::<(), u32>(&mut store, "run_preprocess")
                                .map_err(|e| {
                                    eprintln!("Failed to get run_preprocess from model: {:?}", e);
                                    anyhow::anyhow!("Step {}: Failed to get run_preprocess from model: {}", step_idx, e)
                                })?;

                            if let Err(e) = preprocess_fn.call(&mut store, ()) {
                                eprintln!("Step {}: preprocess failed: {:?}", step_idx, e);
                                break;
                            }
                        }
                        Err(e) => { // Input channel closed → end of processing
                            println!("\x1b[31mWASMTIME\x1b[0m Step {}: Input channel closed: {:?}", step_idx, e);
                            break;
                        }
                    }
                },
                InputChannel::Tensor(receiver) => { // Middle and last steps receive tensor input
                    println!("\x1b[31mWASMTIME\x1b[0m Step {}: Waiting for tensor input", step_idx);
                    match receiver.recv() {
                        Ok(tensor) => { // Received tensor input
                            println!("\x1b[31mWASMTIME\x1b[0m Step {}: Received tensor input", step_idx);
                            store.data_mut().wasi_nn().set_input_tensor(0, 0, tensor)?;
                        }
                        Err(e) => { // Input channel closed → end of processing
                            println!("\x1b[31mWASMTIME\x1b[0m Step {}: Input channel closed: {:?}", step_idx, e);
                            break;
                        }
                    }
                },
            }

            // 4. Run inference
            println!("\x1b[31mWASMTIME\x1b[0m Step {}: Running inference", step_idx);
            let compute_fn = instance.get_typed_func::<(), u32>(&mut store, "compute")
                .map_err(|e| {
                    eprintln!("Failed to get compute from model: {:?}", e);
                    anyhow::anyhow!("Step {}: Failed to get compute from model: {}", step_idx, e)
                })?;

            if let Err(e) = compute_fn.call(&mut store, ()) {
                eprintln!("Step {}: compute failed: {:?}", step_idx, e);
                break;
            }

            // 5. Send output to next step
            match output_channel {
                OutputChannel::Tensor(ref sender) => { // Extract tensor from WASI-NN and send it to the next step
                    let output_tensor = store.data_mut().wasi_nn().get_output_tensor(0, 0)?;
                    println!("\x1b[31mWASMTIME\x1b[0m Step {}: Sending tensor output", step_idx);
                    sender.send(output_tensor)?;
                }
                OutputChannel::Json(ref sender) => { // Retrieve final JSON output and send it to the final channel
                    println!("\x1b[31mWASMTIME\x1b[0m Step {}: Postprocessing", step_idx);

                    // The parameters are also passed to the instance in case the user needs
                    // any of them in the postprocess function.
                    // The inputs have been cleared to avoid passing them to the postprocess function
                    // Generate a new JSON object with the parameters
                    let mut parameters = serde_json::Value::Object(serde_json::Map::new());
                    parameters["model_index"] = serde_json::Value::Number(serde_json::Number::from(step_idx));
                    pass_input(instance, &mut store, &parameters)?;

                    println!("\x1b[31mWASMTIME\x1b[0m Step {}: Running postprocess", step_idx);
                    
                    // Call the postprocess function to finalize the output
                    let postprocess_fn = instance.get_typed_func::<(), u32>(&mut store, "run_postprocess")
                        .map_err(|e| {
                            eprintln!("Failed to get run_postprocess from model: {:?}", e);
                            anyhow::anyhow!("Step {}: Failed to get run_postprocess from model: {}", step_idx, e)
                        })?;

                    if let Err(e) = postprocess_fn.call(&mut store, ()) {
                        eprintln!("Step {}: postprocess failed: {:?}", step_idx, e);
                        break;
                    }

                    // Retrieve the final output JSON from the WASI-NN context
                    let output_json = retrieve_result(instance, &mut store)?;

                    println!("\x1b[31mWASMTIME\x1b[0m Step {}: Sending JSON output", step_idx);
                    sender.send(output_json)?;
                }
            }
        }

        drop(output_channel); // Close the output channel to propagate end of processing
        println!("\x1b[31mWASMTIME\x1b[0m Step {}: Finished processing", step_idx);

        Ok(()) 
    }

}
