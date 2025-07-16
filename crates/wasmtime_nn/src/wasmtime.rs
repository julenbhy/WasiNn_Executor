use common::{ActionCapabilities, WasmRuntime};

use wasmtime::{ Engine, Linker, Module, Store, InstancePre, Instance };
use common::nn_utils::{ NnWasmCtx, link_host_functions, create_store, 
    pass_input, retrieve_result, download_model, pass_model };

#[derive(Clone)]
pub struct WasmtimeRuntime {
    pub engine: Engine,
}

impl Default for WasmtimeRuntime {
    fn default() -> Self {
        WasmtimeRuntime {
            engine: Engine::default(),
        }
    }
}

impl WasmRuntime for WasmtimeRuntime {

    fn initialize(
        &self,
        container_id: String,
        capabilities: ActionCapabilities,
        module: Vec<u8>,
    ) -> anyhow::Result<()> {

        // Deserialize the precompiled module
        let module = unsafe{ Module::deserialize(&self.engine, &module).map_err(|e| {
            anyhow::anyhow!("Failed to deserialize module for container {}: {}", container_id, e)
        })?};

        // Create a linker with wasi_p1 and wasi_nn contexts
        let mut linker: Linker<NnWasmCtx> = Linker::new(&self.engine);
        link_host_functions(&mut linker)?;

        // Generate an instance_pre to avoid repeated type checks
        // in the multiple threads that will run the model.
        let instance_pre = linker.instantiate_pre(&module)?;

        // Get the list of model URLs
        let model_urls = capabilities.model_urls.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Model URLs are required to set up the pipeline")
        })?;

        setup_pipeline(&self.engine, instance_pre, model_urls)?;


        Ok(())
    }

    fn run(
        &self,
        _container_id: &str,
        _parameters: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {

        Ok(serde_json::json!({ "result": "success" }))
    }


    fn destroy(
        &self,
        container_id: &str
    ) {
        println!("Destroying container: {}", container_id);
    }
}

// Input and output channels for each model stage
// The first stage receives JSON input, the last stage outputs JSON,
// and the middle stages receive and output tensors.
use std::sync::mpsc::{Sender as StdSender, Receiver as StdReceiver, channel};
use wasmtime_wasi_nn::Tensor;
enum InputChannel {
    Json(StdReceiver<serde_json::Value>),
    Tensor(StdReceiver<Tensor>),
}
enum OutputChannel {
    Json(StdSender<serde_json::Value>),
    Tensor(StdSender<Tensor>),
}
struct WasiNnPipelineStep {
    input_channel: InputChannel,
    output_channel: OutputChannel,
    instance: Instance,
    store: Store<NnWasmCtx>
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
fn setup_pipeline(
    engine: &Engine,
    instance_pre: InstancePre<NnWasmCtx>,
    model_urls: &[String]
) -> anyhow::Result<()> {
    
    println!("\x1b[31mWASMTIME\x1b[0m Setting up pipeline with model URLs: {:?}", model_urls);

    // Create initial and final channels for communication between threads
    let (initial_sender, initial_receiver_raw) = channel::<serde_json::Value>();
    let mut initial_receiver = Some(initial_receiver_raw);

    // Final channel to receive JSON output (last stage)
    let (final_sender_raw, final_receiver) = channel::<serde_json::Value>();
    let mut final_sender = Some(final_sender_raw);

    let mut receiver = None;
    let mut handles = vec![];

    // Iterate through each model URL to set up the pipeline
    for (stage_idx, model_url) in model_urls.iter().enumerate() {
        println!("\x1b[31mWASMTIME\x1b[0m Setting up stage {} with model URL: {}", stage_idx, model_url);

        // Set up the input and output channels for this stage
        let (input_channel, output_channel, next_receiver) = setup_stage_channels(
            stage_idx,
            model_urls.len(),
            &mut initial_receiver,
            &mut final_sender,
            receiver,
        )?;

        let engine_clone = engine.clone();
        let instance_pre_clone = instance_pre.clone();
        let model_url_clone = model_url.clone();

        // Spawn the thread for this stage
        let handle = std::thread::spawn(move || {
            setup_pipeline_step(stage_idx, &engine_clone, instance_pre_clone, model_url_clone, input_channel, output_channel)
        });
        
        // Store the handle for later use
        handles.push(handle);

        // Move the receiver to the next stage
        receiver = Some(next_receiver);
    }


    Ok(())
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
        .map_err(|e| anyhow::anyhow!("Failed to instantiate WASM module: {}", e))?;

    // Download the model from the provided URL
    let model_bytes = download_model(&model_url)
        .map_err(|e| anyhow::anyhow!("Failed to download model from {}: {}", model_url, e))?;

    // Write the model to the WASM memory
    pass_model(&instance, &mut store, model_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to pass model to WASM instance: {}", e))?;

    // Call the "build_context" function in the WASM instance
    instance.get_typed_func::<(), u32>(&mut store, "build_context")
        .map_err(|_| anyhow::anyhow!("EXECUTOR ERROR: Failed to get build_context from model: {}", model_url))?
        .call(&mut store, ())?;

    // Run the model stage with the input and output channels
    run_model_stage(stage_idx, &instance, &mut store, input_channel, output_channel)
        .map_err(|e| anyhow::anyhow!("Failed to run model stage: {}", e))?;

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
                //log::info!("STAGE {}: Waiting for JSON input", index);
                println!("\x1b[31mWASMTIME\x1b[0m STAGE: Waiting for JSON input");
                match receiver.recv() {
                    Ok(mut value) => { // Received JSON input
                        //log::info!("STAGE {}: Received JSON input", index);
                        println!("\x1b[31mWASMTIME\x1b[0m STAGE: Received JSON input");

                        //value["model_index"] = serde_json::Value::Number(serde_json::Number::from(index));
                        pass_input(instance, &mut store, &value)?;
                        
                        /* 
                        instance.get_typed_func::<(), u32>(&mut store, "preprocess_export")?
                            .call(&mut store, ())
                            .map_err(|e| anyhow::anyhow!("Failed to call preprocess_export: {}", e))?;
                        */
                        if let Err(e) = instance.get_typed_func::<(), u32>(&mut store, "preprocess_export")?
                            .call(&mut store, ()) {
                            eprintln!("preprocess failed: {:?}", e);
                            break;
                        }
                        
                    }
                    Err(_) => { // Input channel closed → end of processing
                        //log::info!("STAGE {}: Input channel closed", index);
                        println!("\x1b[31mWASMTIME\x1b[0m STAGE: Input channel closed");
                        break;
                    }
                }
            },
            InputChannel::Tensor(receiver) => { // Middle and last stages receive tensor input
                //log::info!("STAGE {}: Waiting for tensor input", index);
                println!("\x1b[31mWASMTIME\x1b[0m STAGE: Waiting for tensor input");
                match receiver.recv() {
                    Ok(tensor) => { // Received tensor input
                        //log::info!("STAGE {}: Received tensor input", index);
                        println!("\x1b[31mWASMTIME\x1b[0m STAGE: Received tensor input");
                        // Set input tensor for WASI-NN context
                        store.data_mut().wasi_nn().set_input_tensor(0, 0, tensor)?;
                    }
                    Err(_) => { // Input channel closed → end of processing
                        //log::info!("STAGE {}: Input channel closed", index);
                        println!("\x1b[31mWASMTIME\x1b[0m STAGE: Input channel closed");
                        break;
                    }
                }
            },
        }

        // 4. Run inference
        //log::info!("STAGE {}: Running inference", stage_idx);
        println!("\x1b[31mWASMTIME\x1b[0m STAGE: Running inference");

        //instance.get_typed_func::<(), u32>(&mut store, "compute")?
        //    .call(&mut store, ())?;
        // Same but break the loop in case of an error
        if let Err(e) = instance.get_typed_func::<(), u32>(&mut store, "compute")?
            .call(&mut store, ()) {
            eprintln!("STAGE {}: Inference failed: {:?}", stage_idx, e);
            break;
        }


        // 5. Send output to next stage
        match output_channel {
            OutputChannel::Tensor(ref sender) => { // Extract tensor from WASI-NN and send it to the next stage
                //log::info!("STAGE {}: Sending tensor output", stage_idx);
                println!("\x1b[31mWASMTIME\x1b[0m STAGE: Sending tensor output");

                let output_tensor = store.data_mut().wasi_nn().get_output_tensor(0, 0)?;

                sender.send(output_tensor)?;
            }
            OutputChannel::Json(ref sender) => { // Retrieve final JSON output and send it to the final channel
                //log::info!("STAGE {}: Sending JSON output", stage_idx);
                println!("\x1b[31mWASMTIME\x1b[0m STAGE: Sending JSON output");

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
    //log::info!("STAGE {}: Finished processing", stage_idx);
    println!("\x1b[31mWASMTIME\x1b[0m STAGE: Finished processing");

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
