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