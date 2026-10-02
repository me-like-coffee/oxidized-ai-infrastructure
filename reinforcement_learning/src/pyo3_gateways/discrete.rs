// reinforcement_learning/src/pyo3_gateways/discrete.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use crate::core::environment::Environment;

/// A type-safe wrapper around a dynamic Python Gymnasium environment
/// configured for discrete observations and discrete actions.
pub struct PyO3DiscreteEnvironment {
    gym_env: PyObject,
}

impl PyO3DiscreteEnvironment {
    /// Creates a new discrete environment instance.
    /// 
    /// Allows passing optional configuration kwargs (e.g., `is_slippery: false`).
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

impl Environment for PyO3DiscreteEnvironment {
    type Observation = usize;
    type Action = usize;

    fn reset(&mut self) -> Self::Observation {
        Python::with_gil(|py| {
            let result = self.gym_env.call_method0(py, "reset")
                .expect("FAIL: env.reset() execution threw an exception");
                
            let tuple = result.bind(py).downcast::<PyTuple>()
                .expect("FAIL: env.reset() did not return a standard (obs, info) tuple");
                
            tuple.get_item(0).unwrap().extract().unwrap()
        })
    }

    fn step(&mut self, action: Self::Action) -> (Self::Observation, f64, bool, bool) {
        Python::with_gil(|py| {
            let result = self.gym_env.call_method1(py, "step", (action,))
                .expect("FAIL: env.step() execution threw an exception");
                
            let tuple = result.bind(py).downcast::<PyTuple>()
                .expect("FAIL: env.step() response layout structure is invalid");
            
            let obs: usize = tuple.get_item(0).unwrap().extract().unwrap();
            let reward: f64 = tuple.get_item(1).unwrap().extract().unwrap();
            let terminated: bool = tuple.get_item(2).unwrap().extract().unwrap();
            let truncated: bool = tuple.get_item(3).unwrap().extract().unwrap();
            
            (obs, reward, terminated, truncated)
        })
    }
}
