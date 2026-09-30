// reinforcement_learning/tests/ffi_boundary_test.rs
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::env;

/// Helper function to initialize the embedded Python interpreter and inject
/// the local virtual environment's site-packages into the runtime path.
fn setup_python_env(py: Python<'_>) {
    let sys = py.import("sys").expect("FAIL: Core sys module broken");
    let path = sys.getattr("path").expect("FAIL: Could not access sys.path");
    
    let current_dir = env::current_dir()
        .expect("FAIL: Could not determine current workspace directory");
    
    let parent_dir = current_dir.parent()
        .expect("FAIL: Could not find workspace root parent directory");
    
    let absolute_site_packages = format!(
        "{}/.venv/lib/python3.11/site-packages", 
        parent_dir.to_string_lossy()
    );
    
    path.call_method1("insert", (0, &absolute_site_packages))
        .expect("FAIL: Could not inject custom site-packages path");
}

// =========================================================================
// CATEGORY 1: Environment & Initialization Test
// =========================================================================
#[test]
fn test_embedded_gymnasium_initialization() {
    Python::with_gil(|py| {
        setup_python_env(py);

        let gym = py.import("gymnasium");
        if gym.is_err() {
            let sys = py.import("sys").unwrap();
            let path = sys.getattr("path").unwrap();
            let active_paths: Vec<String> = path.extract().unwrap();
            println!("--- DEBUG RUNTIME PYTHON PATHS LOOKED AT ---");
            for p in active_paths { println!("{}", p); }
            panic!("FAIL: Local .venv environment is missing the Gymnasium package!");
        }
        
        let env = gym.unwrap().call_method1("make", ("CartPole-v1",));
        assert!(env.is_ok(), "FAIL: Could not instantiate CartPole over the local FFI boundary!");
    });
}

// =========================================================================
// CATEGORY 2: NumPy Array & Tensor Boundary Test (Memory Layout)
// =========================================================================
#[test]
fn test_numpy_array_boundary_and_data_extraction() {
    Python::with_gil(|py| {
        setup_python_env(py);
        
        // Import numpy directly via the injected site-packages
        let np = py.import("numpy").expect("FAIL: NumPy package missing from .venv");
        
        // Create a mocked 1D array mimicking a simple observation or force tensor
        let sample_data = vec![0.5, -1.2, 3.14, 0.09];
        let py_array = np.call_method1("array", (sample_data.clone(),))
            .expect("FAIL: Could not construct NumPy array in Python runtime");
            
        // Verify we can extract shape and extract the native data back cleanly over FFI
        let shape: Vec<usize> = py_array.getattr("shape")
            .expect("FAIL: Could not read array shape")
            .extract()
            .unwrap();
            
        let extracted_data: Vec<f64> = py_array.call_method0("tolist")
            .expect("FAIL: Could not cast array back to list")
            .extract()
            .unwrap();
            
        assert_eq!(shape, vec![4], "FAIL: NumPy array shape altered across boundary");
        assert_eq!(extracted_data, sample_data, "FAIL: Floating-point precision data corruption across FFI");
    });
}

// =========================================================================
// CATEGORY 3: State Management & GIL Deadlock Test
// =========================================================================
#[test]
fn test_gil_release_and_reacquisition_safety() {
    Python::with_gil(|py| {
        setup_python_env(py);
        let gym = py.import("gymnasium").unwrap();
        let env = gym.call_method1("make", ("CartPole-v1",)).unwrap();
        
        // Reset the environment to get a valid initial state inside the lock
        let _initial_state = env.call_method0("reset").unwrap();
        
        // Simulating dropping the GIL to let long-running native Rust code execute 
        // without blocking Python runtime concurrency or threads
        py.allow_threads(|| {
            // Spin a heavy native Rust arithmetic calculation while GIL is completely released
            let mut _acc = 0;
            for i in 0..10_000 { _acc += i; }
        });
        
        // Re-entering the GIL implicitly loop ensures we can cleanly re-acquire tokens 
        // and interact with our instantiated Python state objects safely.
        let step_result = env.call_method1("step", (1,)); 
        assert!(step_result.is_ok(), "FAIL: FFI layer deadlocked or corrupted state during GIL release cycle");
    });
}

// =========================================================================
// CATEGORY 4: Gymnasium Abstract Lifecycle Test
// =========================================================================
#[test]
fn test_gymnasium_lifecycle_flow() {
    Python::with_gil(|py| {
        setup_python_env(py);
        let gym = py.import("gymnasium").unwrap();
        let env = gym.call_method1("make", ("CartPole-v1",)).unwrap();
        
        // --- 1. RESET CYCLE ---
        let reset_res = env.call_method0("reset")
            .expect("FAIL: env.reset() execution threw an exception");
        
        let reset_tuple = reset_res.downcast_into::<PyTuple>()
            .expect("FAIL: env.reset() did not return a standard tuple response layout");
            
        let obs_item = reset_tuple.get_item(0).unwrap();
        let observation_init: Vec<f64> = obs_item.extract().unwrap();
        
        let info_item = reset_tuple.get_item(1).unwrap();
        let info_init = info_item.downcast_into::<PyDict>();
        
        assert_eq!(observation_init.len(), 4, "FAIL: CartPole observation space length mismatch");
        assert!(info_init.is_ok(), "FAIL: Reset info element is not a standard PyDict mapping");

        // --- 2. STEP CYCLE ---
        let step_res = env.call_method1("step", (1,))
            .expect("FAIL: env.step() execution threw an exception");
            
        // Use downcast_into to securely cast type representation
        let step_tuple = step_res.downcast_into::<PyTuple>()
            .expect("FAIL: env.step() response layout structure is invalid");
            
        assert_eq!(step_tuple.len(), 5, "FAIL: Standard Gymnasium step() tuple must contain exactly 5 elements");

        let _obs_next: Vec<f64> = step_tuple.get_item(0).unwrap().extract().unwrap();
        let reward: f64 = step_tuple.get_item(1).unwrap().extract().unwrap();
        let terminated: bool = step_tuple.get_item(2).unwrap().extract().unwrap();
        let truncated: bool = step_tuple.get_item(3).unwrap().extract().unwrap();

        assert!(reward > 0.0, "FAIL: CartPole step did not reward a surviving time step");
        assert!(!terminated, "FAIL: Environment terminated on the absolute first step");
        assert!(!truncated, "FAIL: Environment reached truncation limits unexpectedly");
    });
}
