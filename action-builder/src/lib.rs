
#[macro_export]
macro_rules! memory_nn_tensors_method {
    ($pre_fn:ident, $post_fn:ident) => {

        use serde_json::Value;
        use std::ptr;
        use std::alloc::{alloc, Layout};

        static mut INPUT: *mut u8 = ptr::null_mut();
        static mut INPUT_LEN: usize = 0;
        static mut RESULT: Option<String> = None;
        static mut MODEL: *mut u8 = ptr::null_mut();
        static mut MODEL_LEN: usize = 0;

        #[no_mangle]
        pub extern "C" fn alloc_input(size: usize) -> *mut u8 {
            unsafe {
                INPUT = alloc(Layout::from_size_align(size, 1).unwrap());
                INPUT_LEN = size;
                INPUT
            }
        }

        #[no_mangle]
        pub extern "C" fn get_result() -> *const u8 {
            unsafe {
                    RESULT.as_ref().unwrap().as_ptr()
            }
        }

        #[no_mangle]
        pub extern "C" fn get_result_len() -> usize {
            unsafe {
                    RESULT.as_ref().unwrap().len()
            }
        }

        #[no_mangle]
        pub extern "C" fn set_model(size: usize) -> *mut u8 {
            unsafe {
                MODEL = alloc(Layout::from_size_align(size, 1).unwrap());
                MODEL_LEN = size;
                MODEL
            }
        }

        #[no_mangle]
        pub fn run_preprocess() -> anyhow::Result<()>{
            unsafe {
                // Parse the input JSON
                let input_slice = std::slice::from_raw_parts(INPUT, INPUT_LEN);
                let input_str = std::str::from_utf8(input_slice).unwrap();
                let json: Value = serde_json::from_str(input_str).unwrap();

                // Call the function
                $pre_fn(json)?;
            }
            Ok(())
        }

        #[no_mangle]
        pub fn run_postprocess() -> anyhow::Result<()>{
            unsafe {
                // Parse the input JSON
                let input_slice = std::slice::from_raw_parts(INPUT, INPUT_LEN);
                let input_str = std::str::from_utf8(input_slice).unwrap();
                let json: Value = serde_json::from_str(input_str).unwrap();

                // Call the function
                let result_json = $post_fn(json)?;
                
                // Save the result as a string
                RESULT = Some(result_json.to_string());
            }
            Ok(())
        }


        // Compute the inference
        // This is the only step that is executed for all model parts
        #[no_mangle]
        pub fn compute() -> anyhow::Result<()> {
            let context = unsafe { &mut *CONTEXT };
            context.compute()?;
            Ok(())
        }

        // Dummy main function to satisfy the Rust compiler
        pub fn main() {}
        
    };
}
