use anyhow::{Result};
use wasi_nn::{self, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext, Graph};
use base64;
use image::load_from_memory;
use image::imageops::{resize, FilterType};


// Although we only need the execution context,
// we need to make sure the graph is not freed from memory
static mut GRAPH: *mut Graph = std::ptr::null_mut();
static mut CONTEXT: *mut GraphExecutionContext = std::ptr::null_mut();

#[no_mangle]
pub extern "C" fn build_context() -> anyhow::Result<()> {
    println!("\x1b[32mFROM WASM:\x1b[0m Building context for PyTorch model...");
    let model_bytes = unsafe { std::slice::from_raw_parts(MODEL, MODEL_LEN) };
    unsafe {
        // Create the graph and assign it to the global variable
        let graph = Box::new(GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::CPU)
            .build_from_bytes(&[&model_bytes])?);
        GRAPH = Box::into_raw(graph);

        // Create the execution context and assign it to the global variable
        let context = (*GRAPH).init_execution_context()?;
        CONTEXT = Box::into_raw(Box::new(context));
    }
    Ok(())
}



// If this is the first model part, the input tensor should be loaded from the payload json
// The input tensor will be preprocessed and set as the input tensor for the model
// If this is not the first model part, the intermediate tensor will already be set as the input tensor
// from the embedder
#[no_mangle]
pub fn preprocess(json: Value) -> anyhow::Result<()> {
    println!("\x1b[32mFROM WASM:\x1b[0m Preprocessing input tensor...");
    
    // Get the context from the global variable
    let context = unsafe { &mut *CONTEXT };

    //println!("FROM WASM: First step, preprocessing input image");
    let tensor_data = preprocess_images(
        json["inputs"].as_array().ok_or_else(|| anyhow::anyhow!("WASM ERROR: Invalid or missing 'inputs' key"))?,
        224,
        224,
        &[0.485, 0.456, 0.406],
        &[0.229, 0.224, 0.225],
    ).map_err(|e| {
        anyhow::anyhow!("WASM ERROR: Failed to preprocess images: {}", e)
    })?;

    // get the input_shape for the current model part using the model_index
    let model_index = json["model_index"].as_i64()
        .ok_or_else(|| anyhow::anyhow!("WASM ERROR: Missing 'model_index' key or not a i64"))?;

    let mut input_shape = json["input_shapes"]
        .get((model_index) as usize)
        .and_then(|shape| shape.as_array())
        .ok_or_else(|| anyhow::anyhow!("WASM ERROR: Invalid or missing 'input_shapes' for model_index"))?
        .iter()
        .map(|val| val.as_u64().unwrap() as usize)
        .collect::<Vec<usize>>();
    
    // Add the batch_size to the input_shape
    let batch_size = json["batch_size"].as_u64().unwrap(); 
    input_shape[0] = batch_size as usize;

    // Set the input tensor. PyTorch models do not use ports, so it is set to 0 here. 
    // Tensors are passed to the model, and the model's forward method processes these tensors.
    let precision = wasi_nn::TensorType::F32;
    context.set_input(0, precision, &input_shape, &tensor_data)?;

    Ok(())
}





// If this is the last model part, get the output tensor and postprocess it.
// If this is not the last model part, return a mock result
// The embedder will get the intermediate tensor and set it as the input tensor for the next model part
#[no_mangle]
pub fn postprocess(json: Value) -> Result<serde_json::Value, anyhow::Error> {

    let model_index = json["model_index"].as_i64()
        .ok_or_else(|| anyhow::anyhow!("WASM ERROR: Missing 'model_index' key or not a i64"))?;


    // Get the context from the global variable
    let context = unsafe { &mut *CONTEXT };

    //println!("FROM WASM: Last step, postprocessing output tensor");
    // Get the output_shape (the input_shape for the next model part)
    let mut output_shape = json["input_shapes"]
        .get((model_index + 1) as usize)
        .and_then(|shape| shape.as_array())
        .ok_or_else(|| anyhow::anyhow!("WASM ERROR: Invalid output shape for model_index"))?
        .iter()
        .map(|val| val.as_u64().unwrap() as usize)
        .collect::<Vec<usize>>();
    
    // Prepare the output buffer
    let batch_size = json["batch_size"].as_u64().unwrap(); 
    output_shape[0] = batch_size as usize;
    let output_size = output_shape.iter().copied().product(); // Multiply all elements of the shape
    let mut output_buffer = vec![0f32; output_size];

    context.get_output(0, &mut output_buffer[..])?;

    // Postprocess the output tensor
    let class_labels = json["class_labels"].as_array().unwrap();
    let result_json = postprocess_images(output_buffer, class_labels, 3);
    Ok(result_json)
}


/* PREPROCESS RELATED FUNCTIONS */

// Transform a list of images into a batch tensor
fn preprocess_images(
    images_base64: &[serde_json::Value],
    height: u32,
    width: u32,
    mean: &[f32],
    std: &[f32],
) -> Result<Vec<f32>, anyhow::Error> {

    let mut batch_tensors = Vec::new();

    for image_base64 in images_base64 {
        // Get the image as a base64 string
        let image_base64_str = image_base64.as_str().ok_or_else(|| {
            anyhow::anyhow!("WASM ERROR: 'images' should be a base64 string")
        })?;
        let image_bytes = base64::decode(image_base64_str)?;

        // Preprocess the image and add it to the batch
        let tensor_data = preprocess_one(image_bytes, width, height, mean, std);
        batch_tensors.extend(tensor_data);
    }

    Ok(batch_tensors)
}


fn preprocess_one(
    image: Vec<u8>, 
    height: u32, 
    width: u32, 
    mean: &[f32],
    std: &[f32]
) -> Vec<f32> {
    // Load and resize the image
    let img = load_from_memory(&image).unwrap().to_rgb8();
    let resized_img = resize(&img, height, width, FilterType::Triangle);

    // Normalize the image
    let mut normalized_img: Vec<f32> = Vec::new();
    for rgb in resized_img.pixels() {
        normalized_img.push((rgb[0] as f32 / 255. - mean[0]) / std[0]);
        normalized_img.push((rgb[1] as f32 / 255. - mean[1]) / std[1]);
        normalized_img.push((rgb[2] as f32 / 255. - mean[2]) / std[2]);
    }
    // Convert the f32 values to u8 bytes
    let bytes_required = normalized_img.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];
    for c in 0..3 {
        for i in 0..(normalized_img.len() / 3) {
            // Read the number as a f32 and break it into u8 bytes
            let u8_f32: f32 = normalized_img[i * 3 + c] as f32;
            let u8_bytes = u8_f32.to_ne_bytes();

            for j in 0..4 {
                u8_f32_arr[((normalized_img.len() / 3 * c + i) * 4) + j] = u8_bytes[j];
            }
        }
    }
    let tensor_data_f32 = u8_to_f32_conversion(&u8_f32_arr);
    tensor_data_f32
}

fn u8_to_f32_conversion(u8_vec: &[u8]) -> Vec<f32> {
    let mut f32_vec = Vec::with_capacity(u8_vec.len() / 4);  // 4 bytes for each f32
    for chunk in u8_vec.chunks_exact(4) {
        // Convert each 4 bytes into a f32 value
        let f32_val = f32::from_ne_bytes(chunk.try_into().expect("Chunk size must be 4"));
        f32_vec.push(f32_val);
    }
    f32_vec
}



/* POSTPROCESS RELATED FUNCTIONS */
fn postprocess_images(
    output_tensor: Vec<f32>,
    class_labels: &Vec<serde_json::Value>,
    top_k: usize,
) -> serde_json::Value {
    let num_classes = class_labels.len();

    let results = output_tensor
        .chunks(num_classes)
        .enumerate()
        .map(|(i, output)| {
            let sorted_results = sort_results(output);
            let batch_results = sorted_results[..top_k]
                .iter()
                .map(|InferenceResult(class, prob)| {
                    let label = class_labels.get(*class).cloned();
                    serde_json::json!({
                        "class": class,
                        "probability": prob,
                        "label": label
                    })
                })
                .collect::<Vec<serde_json::Value>>();

            (i.to_string(), serde_json::json!(batch_results))
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();

    let json_output = serde_json::json!(results);

    //println!("RESULTS: {:?}", json_output);

    json_output
}


fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}



#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);
