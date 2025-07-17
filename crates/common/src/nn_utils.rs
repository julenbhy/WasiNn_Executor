use wasmtime::{ Engine, Linker, Store, Instance };
use wasmtime_wasi::preview1::{ self, WasiP1Ctx };
use wasmtime_wasi::p2::WasiCtxBuilder;
use wasmtime_wasi::{ DirPerms, FilePerms };
use wasmtime_wasi_nn::witx::WasiNnCtx;
use reqwest;

/// Represents the WebAssembly context for WASI and WASI-NN.
pub struct NnWasmCtx {
    wasi: WasiP1Ctx,
    wasi_nn: WasiNnCtx,
}
impl NnWasmCtx {
    pub fn new(wasi: WasiP1Ctx, wasi_nn: WasiNnCtx) -> Self {
        Self { wasi, wasi_nn }
    }
    pub fn wasi(&mut self) -> &mut WasiP1Ctx { &mut self.wasi }
    pub fn wasi_nn(&mut self) -> &mut WasiNnCtx { &mut self.wasi_nn }
}

/// Links host functions to the provided linker for the WebAssembly context.
/// This includes functions for WASI Preview 1 and WASI-NN.
pub fn link_host_functions(
    linker: &mut Linker<NnWasmCtx>
) -> anyhow::Result<()> {
    preview1::add_to_linker_sync(linker, NnWasmCtx::wasi)?;
    wasmtime_wasi_nn::witx::add_to_linker(linker, NnWasmCtx::wasi_nn)?;
    Ok(())
}

//use wasmtime_wasi_nn::backend::pytorch::PytorchBackend;
/// Creates a new WebAssembly store with the provided engine and initializes
/// a custom context (`WasmCtx`) that includes WASI and WASI-NN support.
pub fn create_store(
    engine: &Engine
) -> anyhow::Result<Store<NnWasmCtx>> {

    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_stderr()
        .preopened_dir("/tmp", "/tmp", DirPerms::all(), FilePerms::all()).unwrap()
        .build_p1();

    //let graph = vec![("pytorch".to_string(), "models".to_string())]; // Convert to Vec<(String, String)>
    let (backends, registry) = wasmtime_wasi_nn::preload(&[]).unwrap();
    let wasi_nn = WasiNnCtx::new(backends, registry);

    let wasm_ctx = NnWasmCtx::new(wasi, wasi_nn);

    Ok(Store::new(engine, wasm_ctx))
}


/// Passes input parameters from the host environment to the WebAssembly instance memory.
/// The module must export a function called `set_input` that returns a pointer to write input
pub fn pass_input(
    instance: &wasmtime::Instance,
    store: &mut Store<NnWasmCtx>,
    parameters: &serde_json::Value
) -> anyhow::Result<()> {

    let input = parameters.to_string();
    // Access the WASM memory
    let memory = instance.get_memory(&mut *store, "memory")
        .ok_or_else(|| anyhow::anyhow!("EXECUTOR ERROR: Failed to get WASM memory"))?;

    // Obtain the pointer to the input with set_input
    let set_input = instance.get_typed_func::<u32, u32>(&mut *store, "set_input")
        .map_err(|_| anyhow::anyhow!("EXECUTOR ERROR: Failed to get set_input"))?;

    let input_ptr = set_input.call(&mut *store, input.len() as u32)? as usize;

    // Write the input to the WASM memory
    let content = input.as_bytes();
    memory.data_mut(&mut *store)[input_ptr..(input_ptr + content.len())].copy_from_slice(content);

    Ok(())
}

/// Retrieves the result from the WebAssembly instance memory and returns it as a JSON value.
/// The module must export a function called `get_result_len` that returns the length of the result,
/// and a function called `get_result` that returns a pointer to the result.
pub fn retrieve_result(
    instance: &wasmtime::Instance,
    store: &mut Store<NnWasmCtx>
) -> anyhow::Result<serde_json::Value> {
    // Access the WASM memory
    let memory = instance.get_memory(&mut *store, "memory")
        .ok_or_else(|| anyhow::anyhow!("EXECUTOR ERROR: Failed to get WASM memory"))?;

    // Get the length of the result with get_result_len
    let get_result_len = instance.get_typed_func::<(), u32>(&mut *store, "get_result_len")
        .map_err(|_| anyhow::anyhow!("EXECUTOR ERROR: Failed to get get_result_len"))?;

    let length = get_result_len.call(&mut *store, ())
        .map_err(|_| anyhow::anyhow!("EXECUTOR ERROR: Failed to call get_result_len"))?
        as usize;

    // Get the pointer to the result with get_result
    let get_result = instance.get_typed_func::<(), u32>(&mut *store, "get_result")
        .map_err(|_| anyhow::anyhow!("EXECUTOR ERROR: Failed to get get_result"))?;

    let content_ptr = get_result.call(&mut *store, ())
        .map_err(|_| anyhow::anyhow!("EXECUTOR ERROR: Failed to call get_result"))?
        as usize;

    // Read the result from the WASM memory
    let content = memory.data(&store)[content_ptr..(content_ptr + length)].to_vec();

    let result = String::from_utf8(content)?;
    let json_result: serde_json::Value = serde_json::from_str(&result)?;

    Ok(json_result)
}


/// Downloads the model from the provided URL and returns its bytes.
pub fn download_model(
    model_url: &str,
) -> anyhow::Result<Vec<u8>> {
    // Download the model from the URL
    let response = reqwest::blocking::get(model_url)
        .map_err(|e| anyhow::anyhow!("Failed to download model from {}: {}", model_url, e))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("Failed to download model: HTTP {}", response.status()));
    }

    // Read the response body as bytes
    let model_bytes = response.bytes()
        .map_err(|e| anyhow::anyhow!("Failed to read model bytes: {}", e))?
        .to_vec();

    Ok(model_bytes)
}


/// Writes the model bytes to the WASM memory.
/// Requires the WASM instance to export a `set_model` function
/// that allocates memory for the model and returns a pointer to it.
pub fn pass_model(
    instance: &Instance,
    store: &mut Store<NnWasmCtx>,
    model_bytes: Vec<u8>,
) -> anyhow::Result<()> {
    // Access the WASM memory
    let memory = instance.get_memory(&mut *store, "memory")
        .ok_or_else(|| anyhow::anyhow!("EXECUTOR ERROR: Failed to get WASM memory"))?;

    // Obtain the pointer to the model with set_model
    let set_model = instance.get_typed_func::<u32, u32>(&mut *store, "set_model")
        .map_err(|_| anyhow::anyhow!("EXECUTOR ERROR: Failed to get set_model"))?;

    let model_ptr = set_model.call(&mut *store, model_bytes.len() as u32)? as usize;

    // Write the model to the WASM memory
    memory.data_mut(&mut *store)[model_ptr..(model_ptr + model_bytes.len())].copy_from_slice(&model_bytes);

    Ok(())
}


/// Downloads inputs from URLs or S3 URLs and encodes them as base64.
/// the inputs are saved in the "inputs" field of the parameters.
pub fn download_inputs(
    parameters: &mut serde_json::Value
) -> anyhow::Result<()> {

    let replace_method = parameters["download_method"].as_str();

    match replace_method {
        Some("URL") => download_inputs_urls_parallel(parameters)?,
        //Some("S3") => download_inputs_s3_parallel(parameters)?,
        //Some("MinIO") => download_inputs_minio_async(parameters)?,
        Some(_) => {
            eprintln!("Invalid download method provided. Skipping input download.");
        },
        None => {
            eprintln!("No 'download_method' key found. Skipping input download.");
        }
    }
    Ok(())
}
use std::thread;
use std::sync::{ Arc, Mutex };
use anyhow::anyhow;
use base64::{engine::general_purpose::STANDARD, Engine as _};

fn download_inputs_urls_parallel(
    parameters: &mut serde_json::Value
) -> anyhow::Result<()> {
    println!("\x1b[31mWASMTIME\x1b[0m Downloading inputs from URLs in parallel...");
    
    let input_urls = parameters["input_urls"].as_array()
        .ok_or_else(|| anyhow!("EXECUTOR ERROR: 'input_urls' must be a list"))?;

    // Create a vector of Mutex<Option<Value>> wrapped in Arc for shared access
    let encoded_inputs: Arc<Vec<Mutex<Option<serde_json::Value>>>> = Arc::new(
        (0..input_urls.len()).map(|_| Mutex::new(None)).collect()
    );

    let mut handles = Vec::with_capacity(input_urls.len());

    for (index, input_url) in input_urls.iter().enumerate() {
        let url = input_url.as_str()
            .ok_or_else(|| anyhow!("EXECUTOR ERROR: 'input_urls' contains a non-string value"))?
            .to_string();

        let encoded_inputs = Arc::clone(&encoded_inputs);

        let handle = thread::spawn(move || -> std::result::Result<(), anyhow::Error> {
            let bytes = reqwest::blocking::get(&url)?
                .error_for_status()?
                .bytes()?.to_vec();

            let encoded = serde_json::Value::String(STANDARD.encode(&bytes));
            let mut slot = encoded_inputs[index].lock().unwrap();
            *slot = Some(encoded);

            Ok(())
        });

        handles.push(handle);
    }

    // Wait for all threads to finish
    for handle in handles {
        handle.join().map_err(|_| anyhow!("Thread error"))??;
    }

    // Extract the results into a Vec<serde_json::Value>
    let results: Vec<serde_json::Value> = encoded_inputs.iter()
        .map(|mutex_opt| {
            mutex_opt.lock().unwrap()
                .clone()
                .ok_or_else(|| anyhow!("Missing encoded input"))
        })
        .collect::<std::result::Result<_, anyhow::Error>>()?;

    parameters["inputs"] = serde_json::json!(results);

    Ok(())
}