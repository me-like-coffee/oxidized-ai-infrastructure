// reinforcement_learning/src/pyo3_gateways/continuous.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use ndarray::Array1;
use crate::core::environment::Environment;

/// A type-safe wrapper around a dynamic Python Gymnasium environment
/// configured for continuous vector observations and discrete actions.
pub struct PyO3ContinuousEnvironment {
    gym_env: PyObject,
}

impl PyO3ContinuousEnvironment {
    /// Creates a new continuous observation environment instance.
    pub fn new(env_name: &str, kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Python::with_gil(|py| {
            let gym = py.import("gymnasium")
                .expect("FAIL: Failed to import Python 'gymnasium' package. Is your .venv active?");
                
            let env = gym.call_method("make", (env_name,), kwargs)
                .expect("FAIL: Failed to call gymnasium.make() over the FFI boundary");
                
            Self { gym_env: env.into() }
        })
    }

    /// Internal fallback constructor to build the wrapper directly from a raw Python reference pointer.
    pub fn from_raw(obj: PyObject) -> Self {
        Self { gym_env: obj }
    }
}

impl Environment for PyO3ContinuousEnvironment {
    type Observation = Array1<f64>;
    type Action = usize;

    fn reset(&mut self) -> Self::Observation {
        Python::with_gil(|py| {
            let result = self.gym_env.call_method0(py, "reset")
                .expect("FAIL: env.reset() execution threw an exception");
                
            // Bind the owned PyObject to the GIL, then downcast it to a PyTuple
            let tuple = result.bind(py).downcast::<PyTuple>()
                .expect("FAIL: env.reset() did not return a standard (obs, info) tuple");
            
            let raw_vec: Vec<f64> = tuple.get_item(0).unwrap().extract()
                .expect("FAIL: Could not extract continuous observation space to Vec<f64>");
                
            Array1::from_vec(raw_vec)
        })
    }

    fn step(&mut self, action: Self::Action) -> (Self::Observation, f64, bool, bool) {
        Python::with_gil(|py| {
            let result = self.gym_env.call_method1(py, "step", (action,))
                .expect("FAIL: env.step() execution threw an exception");
                
            // Apply the same bind and downcast chain here
            let tuple = result.bind(py).downcast::<PyTuple>()
                .expect("FAIL: env.step() response layout structure is invalid");
            
            let raw_vec: Vec<f64> = tuple.get_item(0).unwrap().extract()
                .expect("FAIL: Could not extract next continuous observation space to Vec<f64>");
            let obs = Array1::from_vec(raw_vec);
            
            let reward: f64 = tuple.get_item(1).unwrap().extract().unwrap();
            let terminated: bool = tuple.get_item(2).unwrap().extract().unwrap();
            let truncated: bool = tuple.get_item(3).unwrap().extract().unwrap();
            
            (obs, reward, terminated, truncated)
        })
    }
}
