use tide::{Request, StatusCode, Response};
use tide::log::{info, error};
use serde_json::json;

use common::{utils, ActivationContext, ActivationInit, WasmRuntime};
use tokio::task;

/// POST /:container_id/init
pub async fn init(
    mut req: Request<impl WasmRuntime>
) -> tide::Result {

    let container_id = extract_container_id(&req)?;
    let activation_init: ActivationInit = req.body_json().await
        .map_err(|e| tide::Error::from_str(StatusCode::BadRequest, format!("Invalid JSON body: {}", e)))?;

    info!("WASI_NN_SERVER /init '{}' with id '{}'", activation_init.value.name, container_id);

    let module_bytes = utils::b64_decode(activation_init.value.code)
        .map_err(|e| tide::Error::from_str(StatusCode::BadRequest, format!("Base64 decode failed: {}", e)))?;

    let module = utils::unzip(module_bytes)
        .map_err(|e| tide::Error::from_str(StatusCode::BadRequest, format!("Unzip failed: {}", e)))?;

    req.state()
        .initialize(container_id, activation_init.value.annotations, module)
        .map_err(|e| tide::Error::from_str(StatusCode::InternalServerError, e.to_string()))?;

    Ok(StatusCode::Ok.into())
}

/// POST /:container_id/run
pub async fn run(
    mut req: Request<impl WasmRuntime + Send + Sync + 'static>,
) -> tide::Result {

    let container_id = extract_container_id(&req)?;
    let activation_context: ActivationContext = req.body_json().await
        .map_err(|e| tide::Error::from_str(StatusCode::BadRequest, format!("Invalid JSON body: {}", e)))?;

    info!("WASI_NN_SERVER /run '{}' with id '{}'", activation_context.action_name, container_id);

    /*
    // Launch the runtime in a blocking task
    let runtime = req.state().clone();

    let result = task::spawn_blocking(move || {
        runtime.run(&container_id, activation_context.value)
    }).await;

    let response = match result {
        Ok(Ok(output)) => tide::Response::builder(StatusCode::Ok)
            .body(json!({ "result": output }))
            .build(),

        Ok(Err(err)) => {
            error!("Runtime error: {}", err);
            Response::builder(StatusCode::InternalServerError)
                .body(json!({ "error": err.to_string() }))
                .build()
        }

        Err(join_err) => {
            error!("Thread join error: {}", join_err);
            Response::builder(StatusCode::InternalServerError)
                .body(json!({ "error": join_err.to_string() }))
                .build()
        }
    };
*/

    // Launch the runtime without being a task
    let result = req.state().run(&container_id, activation_context.value);
    let response = match result {
        Ok(output) => Response::builder(StatusCode::Ok)
            .body(json!({ "result": output }))
            .build(),

        Err(err) => {
            error!("Runtime error: {}", err);
            Response::builder(StatusCode::InternalServerError)
                .body(json!({ "error": err.to_string() }))
                .build()
        }
    };

    /*
    // Simulating the response for now
    let fake_result = json!({
        "result": "simulated output",
        "action": activation_context.action_name,
        "container": container_id
    });

    let response = Response::builder(StatusCode::Ok)
        .body(fake_result)
        .build();
    */
    Ok(response)
}

/// POST /:container_id/destroy
pub async fn destroy(
    req: Request<impl WasmRuntime>
) -> tide::Result {
    let container_id = extract_container_id(&req)?;
    info!("WASI_NN_SERVER /destroy: removing container '{}'", container_id);

    req.state().destroy(&container_id);

    Ok(StatusCode::Ok.into())
}

/// Extracts container ID from path or returns BadRequest
fn extract_container_id(
    req: &Request<impl WasmRuntime>
) -> tide::Result<String> {
    req.param("container_id")
        .map(|s| s.to_string())
        .map_err(|_| tide::Error::from_str(StatusCode::BadRequest, "Missing container_id"))
}