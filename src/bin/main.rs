use wasi_nn_server::handlers::{init, destroy, run};
use wasmtime_nn::WasmtimeRuntime as SelectedRuntime;

use tide::log;

static ADDRESS: &str = "127.0.0.1:9000";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tide::log::start(); // Enable logging via env (RUST_LOG)


    let runtime = SelectedRuntime::default();
    let mut app = tide::with_state(runtime);

    app.at("/:container_id/init").post(init);
    app.at("/:container_id/run").post(run);
    app.at("/:container_id/destroy").post(destroy);

    log::info!("Listening on: {}", ADDRESS);
    app.listen(ADDRESS).await?;

    Ok(())
}
